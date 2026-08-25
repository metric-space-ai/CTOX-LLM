//! Model-specific artifact graph composition.
//!
//! This module wires the frozen Qwen tensor names to backend operations. It is
//! intentionally not a generic neural-network runtime. The first executable
//! slice covers embedding, the exact residual MLP subgraph, final norm, and
//! LM head. Token mixers are added separately and must pass the pinned Qwen
//! oracle before a complete executor can be promoted.

use crate::backend::{Activation, Backend};
use crate::loader::ModelArtifact;
use crate::reference::{
    apply_partial_rope, grouped_query_attention, rms_norm_1p_weight, sigmoid_gate, swiglu,
};
use crate::{EngineError, Qwen38Config, Result};

#[derive(Debug)]
pub struct FullAttentionState {
    key_heads: Vec<Vec<f32>>,
    value_heads: Vec<Vec<f32>>,
    head_dim: usize,
    maximum_tokens: usize,
    tokens: usize,
}

impl FullAttentionState {
    pub fn new(key_value_heads: usize, head_dim: usize, maximum_tokens: usize) -> Result<Self> {
        if key_value_heads == 0 || head_dim == 0 || maximum_tokens == 0 {
            return Err(EngineError::Shape(
                "full-attention state dimensions must be non-zero".into(),
            ));
        }
        Ok(Self {
            key_heads: (0..key_value_heads).map(|_| Vec::new()).collect(),
            value_heads: (0..key_value_heads).map(|_| Vec::new()).collect(),
            head_dim,
            maximum_tokens,
            tokens: 0,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn reset(&mut self) {
        self.key_heads.iter_mut().for_each(Vec::clear);
        self.value_heads.iter_mut().for_each(Vec::clear);
        self.tokens = 0;
    }

    fn append(&mut self, position: usize, key: &[f32], value: &[f32]) -> Result<()> {
        let expected = self
            .key_heads
            .len()
            .checked_mul(self.head_dim)
            .ok_or_else(|| EngineError::Shape("full-attention state shape overflows".into()))?;
        if position != self.tokens
            || self.tokens >= self.maximum_tokens
            || key.len() != expected
            || value.len() != expected
        {
            return Err(EngineError::Shape(
                "full-attention append position or shape differs".into(),
            ));
        }
        for head in 0..self.key_heads.len() {
            let start = head * self.head_dim;
            self.key_heads[head].extend_from_slice(&key[start..start + self.head_dim]);
            self.value_heads[head].extend_from_slice(&value[start..start + self.head_dim]);
        }
        self.tokens += 1;
        Ok(())
    }

    fn flattened_key(&self) -> Vec<f32> {
        self.key_heads.iter().flatten().copied().collect()
    }

    fn flattened_value(&self) -> Vec<f32> {
        self.value_heads.iter().flatten().copied().collect()
    }
}

pub struct ArtifactDecoder<'a, B: Backend> {
    artifact: &'a ModelArtifact,
    backend: &'a B,
    rms_epsilon: f32,
}

impl<'a, B: Backend> ArtifactDecoder<'a, B> {
    pub fn new(artifact: &'a ModelArtifact, backend: &'a B, rms_epsilon: f32) -> Result<Self> {
        if !rms_epsilon.is_finite() || rms_epsilon <= 0.0 {
            return Err(EngineError::Shape(
                "decoder RMS epsilon must be finite and positive".into(),
            ));
        }
        Ok(Self {
            artifact,
            backend,
            rms_epsilon,
        })
    }

    pub fn embedding(&self, token_id: u32) -> Result<Vec<f32>> {
        let matrix = self
            .artifact
            .recovered_matrix("model.language_model.embed_tokens.weight")?;
        let row = matrix.row_operation(token_id as usize)?;
        self.backend.recovered_row(&row)
    }

    pub fn projection(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let matrix = self.artifact.recovered_matrix(name)?;
        if input.len() != matrix.matrix.columns {
            return Err(EngineError::Shape(format!(
                "projection {name} received {} values, expected {}",
                input.len(),
                matrix.matrix.columns
            )));
        }
        self.backend
            .fused_matvec(&matrix.operation(input, Activation::Identity)?)
    }

    pub fn rms_norm(&self, name: &str, hidden: &[f32]) -> Result<Vec<f32>> {
        let weight = self.artifact.float_tensor(name)?.to_f32_vec()?;
        rms_norm_1p_weight(hidden, 1, hidden.len(), &weight, self.rms_epsilon)
    }

    /// Execute Qwen's post-token-mixer residual MLP for one token:
    /// `x + down(silu(gate(norm(x))) * up(norm(x)))`.
    pub fn decoder_mlp_residual(&self, layer_prefix: &str, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.is_empty() || hidden.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::Shape(
                "decoder MLP hidden state is empty or non-finite".into(),
            ));
        }
        let normalized = self.rms_norm(
            &format!("{layer_prefix}.post_attention_layernorm.weight"),
            hidden,
        )?;
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let gate = self.projection(&format!("{mlp_prefix}.gate_proj.weight"), &normalized)?;
        let up = self.projection(&format!("{mlp_prefix}.up_proj.weight"), &normalized)?;
        let activated = swiglu(&gate, &up)?;
        let down = self.projection(&format!("{mlp_prefix}.down_proj.weight"), &activated)?;
        if down.len() != hidden.len() {
            return Err(EngineError::Shape(format!(
                "decoder MLP produced {} values, expected {}",
                down.len(),
                hidden.len()
            )));
        }
        Ok(hidden
            .iter()
            .zip(down)
            .map(|(residual, value)| residual + value)
            .collect())
    }

    /// Execute one cached full-attention token mixer and its residual. This is
    /// the exact scalar composition oracle around backend quantized
    /// projections; production backends replace the attention/state internals
    /// with paged fused kernels.
    pub fn full_attention_residual(
        &self,
        layer_prefix: &str,
        hidden: &[f32],
        position: usize,
        state: &mut FullAttentionState,
        config: &Qwen38Config,
    ) -> Result<Vec<f32>> {
        if hidden.len() != config.hidden_size
            || state.key_heads.len() != config.num_key_value_heads
            || state.head_dim != config.head_dim
        {
            return Err(EngineError::Shape(
                "full-attention hidden state or cache geometry differs".into(),
            ));
        }
        let normalized =
            self.rms_norm(&format!("{layer_prefix}.input_layernorm.weight"), hidden)?;
        let prefix = format!("{layer_prefix}.self_attn");
        let query_width = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| EngineError::Shape("query width overflows".into()))?;
        let key_value_width = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| EngineError::Shape("KV width overflows".into()))?;
        let query_gate = self.projection(&format!("{prefix}.q_proj.weight"), &normalized)?;
        if query_gate.len() != query_width * 2 {
            return Err(EngineError::Shape(
                "full-attention query/gate projection shape differs".into(),
            ));
        }
        let mut query = Vec::with_capacity(query_width);
        let mut gate = Vec::with_capacity(query_width);
        for head in query_gate.chunks_exact(config.head_dim * 2) {
            query.extend_from_slice(&head[..config.head_dim]);
            gate.extend_from_slice(&head[config.head_dim..]);
        }
        let mut key = self.projection(&format!("{prefix}.k_proj.weight"), &normalized)?;
        let value = self.projection(&format!("{prefix}.v_proj.weight"), &normalized)?;
        if key.len() != key_value_width || value.len() != key_value_width {
            return Err(EngineError::Shape(
                "full-attention key/value projection shape differs".into(),
            ));
        }
        let q_norm = self
            .artifact
            .float_tensor(&format!("{prefix}.q_norm.weight"))?
            .to_f32_vec()?;
        let k_norm = self
            .artifact
            .float_tensor(&format!("{prefix}.k_norm.weight"))?
            .to_f32_vec()?;
        query = rms_norm_1p_weight(
            &query,
            config.num_attention_heads,
            config.head_dim,
            &q_norm,
            self.rms_epsilon,
        )?;
        key = rms_norm_1p_weight(
            &key,
            config.num_key_value_heads,
            config.head_dim,
            &k_norm,
            self.rms_epsilon,
        )?;
        apply_partial_rope(
            &mut query,
            &mut key,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rotary_dim,
            position as u64,
            config.rope_theta,
        )?;
        state.append(position, &key, &value)?;
        let key = state.flattened_key();
        let value = state.flattened_value();
        let mut attention = grouped_query_attention(
            &query,
            &key,
            &value,
            config.num_attention_heads,
            config.num_key_value_heads,
            1,
            state.tokens(),
            config.head_dim,
            position,
        )?;
        sigmoid_gate(&mut attention, &gate)?;
        let projected = self.projection(&format!("{prefix}.o_proj.weight"), &attention)?;
        if projected.len() != hidden.len() {
            return Err(EngineError::Shape(
                "full-attention output projection shape differs".into(),
            ));
        }
        Ok(hidden
            .iter()
            .zip(projected)
            .map(|(residual, value)| residual + value)
            .collect())
    }

    pub fn final_logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let normalized = self.rms_norm("model.language_model.norm.weight", hidden)?;
        self.projection("lm_head.weight", &normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::format::{ArtifactBuilder, PackedTensor, TensorDType, DEFAULT_ALIGNMENT};
    use crate::loader::ChecksumPolicy;
    use crate::quant::{Q2Block64, BLOCK_LEN};
    use half::f16;

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn recovered_matrix(name: &str, rows: usize, columns: usize, value: f32) -> Vec<PackedTensor> {
        assert!(columns.is_multiple_of(BLOCK_LEN));
        let mut weights = Vec::new();
        for _ in 0..rows * (columns / BLOCK_LEN) {
            weights.extend_from_slice(&Q2Block64::quantize(&[value; BLOCK_LEN]).unwrap().encode());
        }
        vec![
            PackedTensor {
                name: name.into(),
                dtype: TensorDType::Q2B64,
                shape: vec![rows as u64, columns as u64],
                bytes: weights,
            },
            PackedTensor {
                name: format!("{name}.s_in"),
                dtype: TensorDType::F16,
                shape: vec![columns as u64],
                bytes: f16_bytes(&vec![1.0; columns]),
            },
            PackedTensor {
                name: format!("{name}.s_out"),
                dtype: TensorDType::F16,
                shape: vec![rows as u64],
                bytes: f16_bytes(&vec![1.0; rows]),
            },
        ]
    }

    #[test]
    fn mmap_graph_executes_embedding_mlp_norm_and_lm_head() {
        let hidden = BLOCK_LEN;
        let intermediate = BLOCK_LEN;
        let layer = "model.language_model.layers.0";
        let mut tensors = Vec::new();
        tensors.extend(recovered_matrix(
            "model.language_model.embed_tokens.weight",
            2,
            hidden,
            0.5,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.mlp.gate_proj.weight"),
            intermediate,
            hidden,
            1.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.mlp.up_proj.weight"),
            intermediate,
            hidden,
            2.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.mlp.down_proj.weight"),
            hidden,
            intermediate,
            1.0 / intermediate as f32,
        ));
        let head_dim = 32;
        let query_heads = 2;
        let key_value_heads = 1;
        let query_width = query_heads * head_dim;
        let key_value_width = key_value_heads * head_dim;
        tensors.extend(recovered_matrix(
            &format!("{layer}.self_attn.q_proj.weight"),
            query_width * 2,
            hidden,
            1.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.self_attn.k_proj.weight"),
            key_value_width,
            hidden,
            1.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.self_attn.v_proj.weight"),
            key_value_width,
            hidden,
            2.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.self_attn.o_proj.weight"),
            hidden,
            query_width,
            1.0 / query_width as f32,
        ));
        tensors.extend(recovered_matrix(
            "lm_head.weight",
            2,
            hidden,
            1.0 / hidden as f32,
        ));
        for name in [
            format!("{layer}.input_layernorm.weight"),
            format!("{layer}.post_attention_layernorm.weight"),
            "model.language_model.norm.weight".into(),
        ] {
            tensors.push(PackedTensor {
                name,
                dtype: TensorDType::F16,
                shape: vec![hidden as u64],
                bytes: f16_bytes(&vec![0.0; hidden]),
            });
        }
        for name in [
            format!("{layer}.self_attn.q_norm.weight"),
            format!("{layer}.self_attn.k_norm.weight"),
        ] {
            tensors.push(PackedTensor {
                name,
                dtype: TensorDType::F16,
                shape: vec![head_dim as u64],
                bytes: f16_bytes(&vec![0.0; head_dim]),
            });
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("decoder-slice.ctoxq");
        ArtifactBuilder {
            model: "test/qwen".into(),
            revision: "0123456789abcdef".into(),
            target: "cpu-test".into(),
            alignment: DEFAULT_ALIGNMENT,
            tensors,
        }
        .write_new(&path)
        .unwrap();
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors).unwrap();
        let backend = CpuBackend::scalar_verifier();
        let decoder = ArtifactDecoder::new(&artifact, &backend, 1e-6).unwrap();

        let embedding = decoder.embedding(0).unwrap();
        assert_eq!(embedding.len(), hidden);
        assert!(embedding.iter().all(|value| (*value - 0.5).abs() < 1e-6));
        assert!(decoder.embedding(2).is_err());

        let residual = vec![1.0_f32; hidden];
        let attention_config = Qwen38Config {
            hidden_size: hidden,
            num_attention_heads: query_heads,
            num_key_value_heads: key_value_heads,
            head_dim,
            rotary_dim: 16,
            rope_theta: 10_000.0,
            ..Qwen38Config::default()
        };
        let mut attention_state = FullAttentionState::new(key_value_heads, head_dim, 2).unwrap();
        let attention = decoder
            .full_attention_residual(layer, &residual, 0, &mut attention_state, &attention_config)
            .unwrap();
        let normalized = (1.0_f32 + 1e-6).sqrt().recip();
        let expected_attention = 1.0 + 2.0 * normalized / (1.0 + (-normalized).exp());
        assert!(attention
            .iter()
            .all(|value| (*value - expected_attention).abs() < 2e-4));
        assert_eq!(attention_state.tokens(), 1);
        assert!(decoder
            .full_attention_residual(layer, &residual, 2, &mut attention_state, &attention_config,)
            .is_err());
        attention_state.reset();
        assert_eq!(attention_state.tokens(), 0);

        let output = decoder.decoder_mlp_residual(layer, &residual).unwrap();
        let inverse = (1.0_f32 + 1e-6).sqrt().recip();
        let gate = inverse;
        let up = 2.0 * inverse;
        let activated = gate / (1.0 + (-gate).exp()) * up;
        let expected = 1.0 + activated;
        assert!(output.iter().all(|value| (*value - expected).abs() < 2e-4));

        let logits = decoder.final_logits(&output).unwrap();
        assert_eq!(logits.len(), 2);
        assert!(logits.iter().all(|value| value.is_finite()));
        assert!((logits[0] - logits[1]).abs() < 1e-6);
    }
}
