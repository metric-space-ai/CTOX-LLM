use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::config::{LayerKind, Qwen38Config};

pub const INDEPENDENT_FANOUT_POLICY: &str = "independent";
pub const QWEN38_FANOUT_POLICY: &str = "qwen38_fanout_s_in_v1";
pub const QWEN38_FANOUT_GROUPS: usize = 130;
pub const QWEN38_FANOUT_LOGICAL_S_IN: usize = 373;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FanoutGroup {
    pub kind: &'static str,
    pub prefix: String,
    pub scale_names: Vec<String>,
}

fn group(kind: &'static str, prefix: String, projections: &[&str]) -> FanoutGroup {
    FanoutGroup {
        kind,
        scale_names: projections
            .iter()
            .map(|projection| format!("{prefix}.{projection}.weight.s_in"))
            .collect(),
        prefix,
    }
}

/// Mirrors `training/fanout_recovery.py` byte-for-byte. The digest is part of
/// the logical checkpoint identity shared by every backend pack.
pub fn qwen38_fanout_groups(config: &Qwen38Config) -> Vec<FanoutGroup> {
    let mut groups = Vec::new();
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{layer}");
        groups.push(group(
            "mlp_gate_up",
            format!("{prefix}.mlp"),
            &["gate_proj", "up_proj"],
        ));
        match config
            .layer_kind(layer)
            .expect("layer is inside the frozen configuration")
        {
            LayerKind::FullAttention => groups.push(group(
                "full_attention_qkv",
                format!("{prefix}.self_attn"),
                &["q_proj", "k_proj", "v_proj"],
            )),
            LayerKind::LinearAttention => groups.push(group(
                "linear_attention_inputs",
                format!("{prefix}.linear_attn"),
                &["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"],
            )),
        }
    }
    groups.push(group(
        "mlp_gate_up",
        "mtp.layers.0.mlp".into(),
        &["gate_proj", "up_proj"],
    ));
    groups.push(group(
        "full_attention_qkv",
        "mtp.layers.0.self_attn".into(),
        &["q_proj", "k_proj", "v_proj"],
    ));
    groups.sort_by(|left, right| {
        (left.kind, left.prefix.as_str()).cmp(&(right.kind, right.prefix.as_str()))
    });
    groups
}

pub fn fanout_group_sha256(groups: &[FanoutGroup]) -> String {
    let encoded = serde_json::to_vec(groups).expect("fan-out group serialization cannot fail");
    format!("{:x}", Sha256::digest(encoded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_groups_match_python_recovery_contract() {
        let groups = qwen38_fanout_groups(&Qwen38Config::default());
        assert_eq!(groups.len(), QWEN38_FANOUT_GROUPS);
        assert_eq!(
            groups
                .iter()
                .map(|group| group.scale_names.len())
                .sum::<usize>(),
            QWEN38_FANOUT_LOGICAL_S_IN
        );
        assert_eq!(
            fanout_group_sha256(&groups),
            "5cac430d5da6762c8a3525c658f310daebdff0197bb3ca28b3c5cb067d5d35f0"
        );
    }
}
