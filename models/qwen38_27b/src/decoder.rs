//! Model-specific artifact graph composition.
//!
//! This module wires the frozen Qwen tensor names to backend operations. It is
//! intentionally not a generic neural-network runtime. The first executable
//! slice covers embedding, the exact residual MLP subgraph, final norm, and
//! LM head. Token mixers are added separately and must pass the pinned Qwen
//! oracle before a complete executor can be promoted.

use crate::backend::{Activation, Backend};
use crate::config::LayerKind;
use crate::loader::ModelArtifact;
use crate::reference::{
    apply_partial_rope, causal_conv_silu_update, grouped_query_attention,
    recurrent_gated_delta_step, rms_norm_1p_weight, rms_norm_gated, sigmoid_gate, swiglu,
};
use crate::{EngineError, Qwen38Config, Result};

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn repeat_heads(
    values: &[f32],
    heads: usize,
    dimension: usize,
    repeats: usize,
) -> Result<Vec<f32>> {
    if heads == 0 || dimension == 0 || repeats == 0 || values.len() != heads * dimension {
        return Err(EngineError::Shape(
            "linear-attention head repetition shape differs".into(),
        ));
    }
    let mut output = Vec::with_capacity(values.len() * repeats);
    for head in values.chunks_exact(dimension) {
        for _ in 0..repeats {
            output.extend_from_slice(head);
        }
    }
    Ok(output)
}

#[derive(Debug)]
pub struct FullAttentionState {
    key_heads: Vec<Vec<f32>>,
    value_heads: Vec<Vec<f32>>,
    head_dim: usize,
    maximum_tokens: usize,
    tokens: usize,
}

#[derive(Debug)]
pub struct LinearAttentionState {
    convolution: Vec<f32>,
    recurrent: Vec<f32>,
    channels: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel: usize,
    maximum_tokens: usize,
    tokens: usize,
}

impl LinearAttentionState {
    pub fn new(config: &Qwen38Config, maximum_tokens: usize) -> Result<Self> {
        let key_width = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)
            .ok_or_else(|| EngineError::Shape("linear key width overflows".into()))?;
        let value_width = config
            .linear_num_value_heads
            .checked_mul(config.linear_value_head_dim)
            .ok_or_else(|| EngineError::Shape("linear value width overflows".into()))?;
        let channels = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or_else(|| EngineError::Shape("linear convolution width overflows".into()))?;
        let convolution_values = channels
            .checked_mul(config.linear_conv_kernel_dim)
            .ok_or_else(|| EngineError::Shape("linear convolution state overflows".into()))?;
        let recurrent_values = config
            .linear_num_value_heads
            .checked_mul(config.linear_key_head_dim)
            .and_then(|values| values.checked_mul(config.linear_value_head_dim))
            .ok_or_else(|| EngineError::Shape("linear recurrent state overflows".into()))?;
        if maximum_tokens == 0
            || config.linear_num_key_heads == 0
            || config.linear_num_value_heads == 0
            || !config
                .linear_num_value_heads
                .is_multiple_of(config.linear_num_key_heads)
        {
            return Err(EngineError::Shape(
                "linear-attention state geometry differs".into(),
            ));
        }
        Ok(Self {
            convolution: vec![0.0; convolution_values],
            recurrent: vec![0.0; recurrent_values],
            channels,
            value_heads: config.linear_num_value_heads,
            key_dim: config.linear_key_head_dim,
            value_dim: config.linear_value_head_dim,
            kernel: config.linear_conv_kernel_dim,
            maximum_tokens,
            tokens: 0,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn allocated_bytes(&self) -> usize {
        (self.convolution.len() + self.recurrent.len()) * std::mem::size_of::<f32>()
    }

    pub fn reset(&mut self) {
        self.convolution.fill(0.0);
        self.recurrent.fill(0.0);
        self.tokens = 0;
    }

    fn validate_step(&self, position: usize) -> Result<()> {
        if position != self.tokens || self.tokens >= self.maximum_tokens {
            return Err(EngineError::Shape(
                "linear-attention position exceeds or differs from state".into(),
            ));
        }
        Ok(())
    }
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

    pub fn allocated_bytes(&self) -> usize {
        self.key_heads
            .iter()
            .chain(&self.value_heads)
            .map(|head| head.capacity() * std::mem::size_of::<f32>())
            .sum()
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

#[derive(Debug)]
pub enum DecoderLayerState {
    Linear(LinearAttentionState),
    Full(FullAttentionState),
}

#[derive(Debug)]
pub struct DecoderState {
    layers: Vec<DecoderLayerState>,
    maximum_tokens: usize,
    position: usize,
}

impl DecoderState {
    pub fn new(config: &Qwen38Config, maximum_tokens: usize) -> Result<Self> {
        if maximum_tokens == 0 || maximum_tokens > config.max_position_embeddings {
            return Err(EngineError::Shape(
                "decoder state capacity is zero or exceeds model context".into(),
            ));
        }
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            layers.push(
                match config.layer_kind(layer).ok_or_else(|| {
                    EngineError::Shape(format!("decoder layer {layer} is outside config"))
                })? {
                    LayerKind::LinearAttention => DecoderLayerState::Linear(
                        LinearAttentionState::new(config, maximum_tokens)?,
                    ),
                    LayerKind::FullAttention => DecoderLayerState::Full(FullAttentionState::new(
                        config.num_key_value_heads,
                        config.head_dim,
                        maximum_tokens,
                    )?),
                },
            );
        }
        Ok(Self {
            layers,
            maximum_tokens,
            position: 0,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn allocated_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| match layer {
                DecoderLayerState::Linear(state) => state.allocated_bytes(),
                DecoderLayerState::Full(state) => state.allocated_bytes(),
            })
            .sum()
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            match layer {
                DecoderLayerState::Linear(state) => state.reset(),
                DecoderLayerState::Full(state) => state.reset(),
            }
        }
        self.position = 0;
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

    /// Execute one cached Qwen GatedDeltaNet token mixer and residual using
    /// the pinned recurrent rule. Production backends replace the scalar
    /// convolution/recurrent core with fused platform kernels while retaining
    /// this exact tensor order and state transition.
    pub fn linear_attention_residual(
        &self,
        layer_prefix: &str,
        hidden: &[f32],
        position: usize,
        state: &mut LinearAttentionState,
        config: &Qwen38Config,
    ) -> Result<Vec<f32>> {
        state.validate_step(position)?;
        let key_width = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_width = config.linear_num_value_heads * config.linear_value_head_dim;
        let convolution_width = key_width * 2 + value_width;
        if hidden.len() != config.hidden_size
            || state.channels != convolution_width
            || state.value_heads != config.linear_num_value_heads
            || state.key_dim != config.linear_key_head_dim
            || state.value_dim != config.linear_value_head_dim
            || state.kernel != config.linear_conv_kernel_dim
        {
            return Err(EngineError::Shape(
                "linear-attention hidden state or cache geometry differs".into(),
            ));
        }
        let normalized =
            self.rms_norm(&format!("{layer_prefix}.input_layernorm.weight"), hidden)?;
        let prefix = format!("{layer_prefix}.linear_attn");
        let mixed_qkv = self.projection(&format!("{prefix}.in_proj_qkv.weight"), &normalized)?;
        let z = self.projection(&format!("{prefix}.in_proj_z.weight"), &normalized)?;
        let a = self.projection(&format!("{prefix}.in_proj_a.weight"), &normalized)?;
        let b = self.projection(&format!("{prefix}.in_proj_b.weight"), &normalized)?;
        if mixed_qkv.len() != convolution_width
            || z.len() != value_width
            || a.len() != config.linear_num_value_heads
            || b.len() != config.linear_num_value_heads
        {
            return Err(EngineError::Shape(
                "linear-attention projection shape differs".into(),
            ));
        }
        let convolution_weight = self
            .artifact
            .float_tensor(&format!("{prefix}.conv1d.weight"))?
            .to_f32_vec()?;
        let convolved = causal_conv_silu_update(
            &mixed_qkv,
            &mut state.convolution,
            &convolution_weight,
            convolution_width,
            config.linear_conv_kernel_dim,
        )?;
        let query = &convolved[..key_width];
        let key = &convolved[key_width..key_width * 2];
        let value = &convolved[key_width * 2..];
        let repeats = config.linear_num_value_heads / config.linear_num_key_heads;
        let query = repeat_heads(
            query,
            config.linear_num_key_heads,
            config.linear_key_head_dim,
            repeats,
        )?;
        let key = repeat_heads(
            key,
            config.linear_num_key_heads,
            config.linear_key_head_dim,
            repeats,
        )?;
        let beta: Vec<f32> = b.iter().map(|value| 1.0 / (1.0 + (-value).exp())).collect();
        let a_log = self
            .artifact
            .float_tensor(&format!("{prefix}.A_log"))?
            .to_f32_vec()?;
        let dt_bias = self
            .artifact
            .float_tensor(&format!("{prefix}.dt_bias"))?
            .to_f32_vec()?;
        if a_log.len() != config.linear_num_value_heads
            || dt_bias.len() != config.linear_num_value_heads
        {
            return Err(EngineError::Shape(
                "linear-attention decay parameter shape differs".into(),
            ));
        }
        let log_decay: Vec<f32> = a
            .iter()
            .zip(a_log)
            .zip(dt_bias)
            .map(|((a, a_log), dt_bias)| -a_log.exp() * softplus(a + dt_bias))
            .collect();
        let core = recurrent_gated_delta_step(
            &query,
            &key,
            value,
            &log_decay,
            &beta,
            &mut state.recurrent,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        )?;
        let norm_weight = self
            .artifact
            .float_tensor(&format!("{prefix}.norm.weight"))?
            .to_f32_vec()?;
        let core = rms_norm_gated(
            &core,
            &z,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            &norm_weight,
            self.rms_epsilon,
        )?;
        let projected = self.projection(&format!("{prefix}.out_proj.weight"), &core)?;
        if projected.len() != hidden.len() {
            return Err(EngineError::Shape(
                "linear-attention output projection shape differs".into(),
            ));
        }
        state.tokens += 1;
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

    /// Complete target-model decode for one input token. The method is a
    /// correctness composition over the frozen 64-layer topology; optimized
    /// executors batch/fuse these same transitions per backend.
    pub fn decode_target_token(
        &self,
        token_id: u32,
        state: &mut DecoderState,
        config: &Qwen38Config,
    ) -> Result<Vec<f32>> {
        if state.layers.len() != config.num_hidden_layers || state.position >= state.maximum_tokens
        {
            return Err(EngineError::Shape(
                "decoder state topology or context capacity differs".into(),
            ));
        }
        let position = state.position;
        let mut hidden = self.embedding(token_id)?;
        if hidden.len() != config.hidden_size {
            return Err(EngineError::Shape(
                "embedding width differs from decoder hidden size".into(),
            ));
        }
        for (layer, layer_state) in state.layers.iter_mut().enumerate() {
            let prefix = format!("model.language_model.layers.{layer}");
            hidden = match (config.layer_kind(layer), layer_state) {
                (Some(LayerKind::LinearAttention), DecoderLayerState::Linear(linear_state)) => self
                    .linear_attention_residual(&prefix, &hidden, position, linear_state, config)?,
                (Some(LayerKind::FullAttention), DecoderLayerState::Full(full_state)) => {
                    self.full_attention_residual(&prefix, &hidden, position, full_state, config)?
                }
                _ => {
                    return Err(EngineError::InvalidState(format!(
                        "decoder layer {layer} state kind differs from config"
                    )))
                }
            };
            hidden = self.decoder_mlp_residual(&prefix, &hidden)?;
        }
        let logits = self.final_logits(&hidden)?;
        state.position += 1;
        Ok(logits)
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
        let linear_key_heads = 1;
        let linear_value_heads = 2;
        let linear_key_dim = 32;
        let linear_value_dim = 32;
        let linear_kernel = 2;
        let linear_key_width = linear_key_heads * linear_key_dim;
        let linear_value_width = linear_value_heads * linear_value_dim;
        let linear_convolution_width = linear_key_width * 2 + linear_value_width;
        let linear = format!("{layer}.linear_attn");
        tensors.extend(recovered_matrix(
            &format!("{linear}.in_proj_qkv.weight"),
            linear_convolution_width,
            hidden,
            1.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{linear}.in_proj_z.weight"),
            linear_value_width,
            hidden,
            1.0 / hidden as f32,
        ));
        for projection in ["in_proj_a", "in_proj_b"] {
            tensors.extend(recovered_matrix(
                &format!("{linear}.{projection}.weight"),
                linear_value_heads,
                hidden,
                1.0 / hidden as f32,
            ));
        }
        tensors.extend(recovered_matrix(
            &format!("{linear}.out_proj.weight"),
            hidden,
            linear_value_width,
            1.0 / linear_value_width as f32,
        ));
        let convolution_weights: Vec<f32> = (0..linear_convolution_width)
            .flat_map(|_| [0.0_f32, 1.0_f32])
            .collect();
        tensors.push(PackedTensor {
            name: format!("{linear}.conv1d.weight"),
            dtype: TensorDType::F16,
            shape: vec![linear_convolution_width as u64, 1, linear_kernel as u64],
            bytes: f16_bytes(&convolution_weights),
        });
        for name in ["A_log", "dt_bias"] {
            tensors.push(PackedTensor {
                name: format!("{linear}.{name}"),
                dtype: TensorDType::F32,
                shape: vec![linear_value_heads as u64],
                bytes: vec![0_u8; linear_value_heads * 4],
            });
        }
        tensors.push(PackedTensor {
            name: format!("{linear}.norm.weight"),
            dtype: TensorDType::F16,
            shape: vec![linear_value_dim as u64],
            bytes: f16_bytes(&vec![1.0; linear_value_dim]),
        });
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

        let linear_config = Qwen38Config {
            hidden_size: hidden,
            linear_num_key_heads: linear_key_heads,
            linear_num_value_heads: linear_value_heads,
            linear_key_head_dim: linear_key_dim,
            linear_value_head_dim: linear_value_dim,
            linear_conv_kernel_dim: linear_kernel,
            ..Qwen38Config::default()
        };
        let mut linear_state = LinearAttentionState::new(&linear_config, 2).unwrap();
        assert_eq!(linear_state.allocated_bytes(), 9_216);
        let linear_output = decoder
            .linear_attention_residual(layer, &residual, 0, &mut linear_state, &linear_config)
            .unwrap();
        assert_eq!(linear_output.len(), hidden);
        assert!(linear_output.iter().all(|value| value.is_finite()));
        assert_eq!(linear_state.tokens(), 1);
        assert!(decoder
            .linear_attention_residual(layer, &residual, 2, &mut linear_state, &linear_config)
            .is_err());
        linear_state.reset();
        let repeated = decoder
            .linear_attention_residual(layer, &residual, 0, &mut linear_state, &linear_config)
            .unwrap();
        assert_eq!(linear_output, repeated);

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

        let linear_decode_config = Qwen38Config {
            num_hidden_layers: 1,
            full_attention_interval: 4,
            max_position_embeddings: 2,
            ..linear_config.clone()
        };
        let mut target_state = DecoderState::new(&linear_decode_config, 2).unwrap();
        assert_eq!(target_state.allocated_bytes(), 9_216);
        let target_logits = decoder
            .decode_target_token(0, &mut target_state, &linear_decode_config)
            .unwrap();
        assert_eq!(target_logits.len(), 2);
        assert_eq!(target_state.position(), 1);
        target_state.reset();
        assert_eq!(target_state.position(), 0);
        assert_eq!(
            decoder
                .decode_target_token(0, &mut target_state, &linear_decode_config)
                .unwrap(),
            target_logits
        );

        let full_decode_config = Qwen38Config {
            num_hidden_layers: 1,
            full_attention_interval: 1,
            max_position_embeddings: 2,
            ..attention_config
        };
        let mut full_target_state = DecoderState::new(&full_decode_config, 2).unwrap();
        let full_logits = decoder
            .decode_target_token(0, &mut full_target_state, &full_decode_config)
            .unwrap();
        assert_eq!(full_logits.len(), 2);
        assert!(full_logits.iter().all(|value| value.is_finite()));
        assert_eq!(full_target_state.position(), 1);
    }
}
