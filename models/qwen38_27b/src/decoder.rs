//! Model-specific artifact graph composition.
//!
//! This module wires the frozen Qwen tensor names to backend operations. It is
//! intentionally not a generic neural-network runtime. The first executable
//! slice covers embedding, the exact residual MLP subgraph, final norm, and
//! LM head. Token mixers are added separately and must pass the pinned Qwen
//! oracle before a complete executor can be promoted.

use crate::backend::{Activation, Backend};
use crate::loader::ModelArtifact;
use crate::reference::{rms_norm_1p_weight, swiglu};
use crate::{EngineError, Result};

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
        tensors.extend(recovered_matrix(
            "lm_head.weight",
            2,
            hidden,
            1.0 / hidden as f32,
        ));
        for name in [
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
