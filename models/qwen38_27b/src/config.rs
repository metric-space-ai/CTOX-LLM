use serde::{Deserialize, Serialize};

pub const MODEL_ID: &str = "Qwen/Qwen3.8-27B";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen38Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_epsilon: f32,
    pub max_position_embeddings: usize,
    pub full_attention_interval: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub mtp_num_hidden_layers: usize,
    pub vision_is_separate: bool,
}

impl Default for Qwen38Config {
    fn default() -> Self {
        Self {
            vocab_size: 248_320,
            hidden_size: 5_120,
            intermediate_size: 17_408,
            num_hidden_layers: 64,
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
            rope_theta: 10_000_000.0,
            rms_norm_epsilon: 1e-6,
            max_position_embeddings: 262_144,
            full_attention_interval: 4,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            mtp_num_hidden_layers: 1,
            vision_is_separate: true,
        }
    }
}

impl Qwen38Config {
    pub fn layer_kind(&self, layer: usize) -> Option<LayerKind> {
        if layer >= self.num_hidden_layers {
            return None;
        }
        Some(
            if (layer + 1).is_multiple_of(self.full_attention_interval) {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            },
        )
    }

    pub fn full_attention_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&layer| self.layer_kind(layer) == Some(LayerKind::FullAttention))
            .count()
    }

    pub fn linear_attention_layers(&self) -> usize {
        self.num_hidden_layers - self.full_attention_layers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_topology_has_16_full_attention_layers() {
        let config = Qwen38Config::default();
        assert_eq!(config.full_attention_layers(), 16);
        assert_eq!(config.linear_attention_layers(), 48);
        assert_eq!(config.layer_kind(0), Some(LayerKind::LinearAttention));
        assert_eq!(config.layer_kind(3), Some(LayerKind::FullAttention));
        assert_eq!(config.layer_kind(63), Some(LayerKind::FullAttention));
        assert_eq!(config.rotary_dim, config.head_dim / 4);
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert_eq!(config.rms_norm_epsilon, 1e-6);
    }
}
