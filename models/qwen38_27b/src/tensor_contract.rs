//! Exact tensor topology for the frozen Qwen3.8 text plus MTP graph.

use std::collections::BTreeMap;

use crate::config::{LayerKind, MODEL_ID};
use crate::format::{ModelManifest, TensorDType};
use crate::{EngineError, Qwen38Config, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorClass {
    QuantizedMatrix,
    F16,
    F32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub class: TensorClass,
    pub shape: Vec<u64>,
}

fn insert(
    tensors: &mut BTreeMap<String, TensorSpec>,
    name: impl Into<String>,
    class: TensorClass,
    shape: impl Into<Vec<u64>>,
) {
    let name = name.into();
    assert!(
        tensors
            .insert(
                name.clone(),
                TensorSpec {
                    class,
                    shape: shape.into(),
                },
            )
            .is_none(),
        "duplicate Qwen tensor contract {name}"
    );
}

fn quantized(
    tensors: &mut BTreeMap<String, TensorSpec>,
    name: impl Into<String>,
    rows: usize,
    columns: usize,
) {
    let name = name.into();
    insert(
        tensors,
        name.clone(),
        TensorClass::QuantizedMatrix,
        vec![rows as u64, columns as u64],
    );
    insert(
        tensors,
        format!("{name}.s_in"),
        TensorClass::F16,
        vec![columns as u64],
    );
    insert(
        tensors,
        format!("{name}.s_out"),
        TensorClass::F16,
        vec![rows as u64],
    );
}

fn full_attention(tensors: &mut BTreeMap<String, TensorSpec>, prefix: &str, config: &Qwen38Config) {
    let query = config.num_attention_heads * config.head_dim;
    let key_value = config.num_key_value_heads * config.head_dim;
    quantized(
        tensors,
        format!("{prefix}.q_proj.weight"),
        query * 2,
        config.hidden_size,
    );
    quantized(
        tensors,
        format!("{prefix}.k_proj.weight"),
        key_value,
        config.hidden_size,
    );
    quantized(
        tensors,
        format!("{prefix}.v_proj.weight"),
        key_value,
        config.hidden_size,
    );
    quantized(
        tensors,
        format!("{prefix}.o_proj.weight"),
        config.hidden_size,
        query,
    );
    insert(
        tensors,
        format!("{prefix}.q_norm.weight"),
        TensorClass::F16,
        vec![config.head_dim as u64],
    );
    insert(
        tensors,
        format!("{prefix}.k_norm.weight"),
        TensorClass::F16,
        vec![config.head_dim as u64],
    );
}

fn mlp(tensors: &mut BTreeMap<String, TensorSpec>, prefix: &str, config: &Qwen38Config) {
    for projection in ["gate_proj", "up_proj"] {
        quantized(
            tensors,
            format!("{prefix}.{projection}.weight"),
            config.intermediate_size,
            config.hidden_size,
        );
    }
    quantized(
        tensors,
        format!("{prefix}.down_proj.weight"),
        config.hidden_size,
        config.intermediate_size,
    );
}

fn decoder_norms(tensors: &mut BTreeMap<String, TensorSpec>, prefix: &str, config: &Qwen38Config) {
    for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
        insert(
            tensors,
            format!("{prefix}.{name}"),
            TensorClass::F16,
            vec![config.hidden_size as u64],
        );
    }
}

pub fn expected_tensor_contract(config: &Qwen38Config) -> BTreeMap<String, TensorSpec> {
    let mut tensors = BTreeMap::new();
    quantized(
        &mut tensors,
        "model.language_model.embed_tokens.weight",
        config.vocab_size,
        config.hidden_size,
    );
    quantized(
        &mut tensors,
        "lm_head.weight",
        config.vocab_size,
        config.hidden_size,
    );
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{layer}");
        decoder_norms(&mut tensors, &prefix, config);
        mlp(&mut tensors, &format!("{prefix}.mlp"), config);
        match config.layer_kind(layer).expect("layer is in range") {
            LayerKind::FullAttention => {
                full_attention(&mut tensors, &format!("{prefix}.self_attn"), config);
            }
            LayerKind::LinearAttention => {
                let prefix = format!("{prefix}.linear_attn");
                let key_width = config.linear_num_key_heads * config.linear_key_head_dim;
                let value_width = config.linear_num_value_heads * config.linear_value_head_dim;
                let convolution_width = key_width * 2 + value_width;
                quantized(
                    &mut tensors,
                    format!("{prefix}.in_proj_qkv.weight"),
                    convolution_width,
                    config.hidden_size,
                );
                quantized(
                    &mut tensors,
                    format!("{prefix}.in_proj_z.weight"),
                    value_width,
                    config.hidden_size,
                );
                for projection in ["in_proj_a", "in_proj_b"] {
                    quantized(
                        &mut tensors,
                        format!("{prefix}.{projection}.weight"),
                        config.linear_num_value_heads,
                        config.hidden_size,
                    );
                }
                quantized(
                    &mut tensors,
                    format!("{prefix}.out_proj.weight"),
                    config.hidden_size,
                    value_width,
                );
                insert(
                    &mut tensors,
                    format!("{prefix}.conv1d.weight"),
                    TensorClass::F16,
                    vec![
                        convolution_width as u64,
                        1,
                        config.linear_conv_kernel_dim as u64,
                    ],
                );
                for name in ["A_log", "dt_bias"] {
                    insert(
                        &mut tensors,
                        format!("{prefix}.{name}"),
                        TensorClass::F32,
                        vec![config.linear_num_value_heads as u64],
                    );
                }
                insert(
                    &mut tensors,
                    format!("{prefix}.norm.weight"),
                    TensorClass::F16,
                    vec![config.linear_value_head_dim as u64],
                );
            }
        }
    }
    insert(
        &mut tensors,
        "model.language_model.norm.weight",
        TensorClass::F16,
        vec![config.hidden_size as u64],
    );

    quantized(
        &mut tensors,
        "mtp.fc.weight",
        config.hidden_size,
        config.hidden_size * 2,
    );
    let mtp_layer = "mtp.layers.0";
    decoder_norms(&mut tensors, mtp_layer, config);
    mlp(&mut tensors, &format!("{mtp_layer}.mlp"), config);
    full_attention(&mut tensors, &format!("{mtp_layer}.self_attn"), config);
    for name in [
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ] {
        insert(
            &mut tensors,
            name,
            TensorClass::F16,
            vec![config.hidden_size as u64],
        );
    }
    tensors
}

pub fn validate_tensor_contract(manifest: &ModelManifest, config: &Qwen38Config) -> Result<()> {
    if manifest.model != MODEL_ID {
        return Err(EngineError::InvalidArtifact(format!(
            "tensor graph expects {MODEL_ID}, got {}",
            manifest.model
        )));
    }
    let expected = expected_tensor_contract(config);
    if manifest.tensors.len() != expected.len() {
        return Err(EngineError::InvalidArtifact(format!(
            "Qwen tensor graph has {} tensors, expected {}",
            manifest.tensors.len(),
            expected.len()
        )));
    }
    for tensor in &manifest.tensors {
        let Some(spec) = expected.get(&tensor.name) else {
            return Err(EngineError::InvalidArtifact(format!(
                "Qwen tensor graph contains unexpected {}",
                tensor.name
            )));
        };
        if tensor.shape != spec.shape {
            return Err(EngineError::Shape(format!(
                "Qwen tensor {} has shape {:?}, expected {:?}",
                tensor.name, tensor.shape, spec.shape
            )));
        }
        let dtype_matches = match spec.class {
            TensorClass::QuantizedMatrix => matches!(
                tensor.dtype,
                TensorDType::Q2B64 | TensorDType::Q4B64 | TensorDType::MixedQ2Q4B64
            ),
            TensorClass::F16 => tensor.dtype == TensorDType::F16,
            TensorClass::F32 => tensor.dtype == TensorDType::F32,
        };
        if !dtype_matches {
            return Err(EngineError::InvalidArtifact(format!(
                "Qwen tensor {} has {:?}, expected {:?}",
                tensor.name, tensor.dtype, spec.class
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::TensorEntry;

    fn manifest() -> ModelManifest {
        let config = Qwen38Config::default();
        let tensors = expected_tensor_contract(&config)
            .into_iter()
            .map(|(name, spec)| TensorEntry {
                name,
                dtype: match spec.class {
                    TensorClass::QuantizedMatrix => TensorDType::Q2B64,
                    TensorClass::F16 => TensorDType::F16,
                    TensorClass::F32 => TensorDType::F32,
                },
                shape: spec.shape,
                offset: 0,
                length: 0,
                sha256: "0".repeat(64),
                segments: Vec::new(),
            })
            .collect();
        ModelManifest {
            format: "ctox.q2q4.v1".into(),
            model: MODEL_ID.into(),
            revision: "1d4bf0f".into(),
            alignment: 256,
            target: "logical".into(),
            recovery: None,
            tensors,
        }
    }

    #[test]
    fn complete_text_and_mtp_graph_has_exact_counts() {
        let expected = expected_tensor_contract(&Qwen38Config::default());
        assert_eq!(expected.len(), 1_878);
        assert_eq!(
            expected
                .values()
                .filter(|spec| spec.class == TensorClass::QuantizedMatrix)
                .count(),
            506
        );
        validate_tensor_contract(&manifest(), &Qwen38Config::default()).unwrap();
    }

    #[test]
    fn missing_or_wrong_tensor_fails_closed() {
        let mut missing = manifest();
        missing
            .tensors
            .retain(|tensor| tensor.name != "mtp.layers.0.self_attn.q_proj.weight");
        assert!(validate_tensor_contract(&missing, &Qwen38Config::default()).is_err());

        let mut wrong = manifest();
        wrong
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "model.language_model.layers.0.linear_attn.A_log")
            .unwrap()
            .shape = vec![47];
        assert!(validate_tensor_contract(&wrong, &Qwen38Config::default()).is_err());
    }
}
