use serde::{Deserialize, Serialize};

use crate::config::LayerKind;
use crate::Qwen38Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    RmsNorm,
    QuantizedProjection,
    FullAttention,
    LinearAttention,
    SwiGlu,
    Mtp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapdragonOwner {
    Htp,
    Adreno,
    CpuControl,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphStep {
    pub layer: Option<usize>,
    pub operation: OperationKind,
    pub snapdragon_owner: SnapdragonOwner,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphPlan {
    pub steps: Vec<GraphStep>,
    pub mtp_resident: bool,
    pub vision_resident: bool,
}

impl GraphPlan {
    pub fn qwen38(config: &Qwen38Config) -> Self {
        let mut steps = Vec::with_capacity(config.num_hidden_layers * 5 + 1);
        for layer in 0..config.num_hidden_layers {
            steps.push(GraphStep {
                layer: Some(layer),
                operation: OperationKind::RmsNorm,
                snapdragon_owner: SnapdragonOwner::Adreno,
            });
            steps.push(GraphStep {
                layer: Some(layer),
                operation: OperationKind::QuantizedProjection,
                snapdragon_owner: SnapdragonOwner::Htp,
            });
            steps.push(GraphStep {
                layer: Some(layer),
                operation: match config.layer_kind(layer).expect("layer is in range") {
                    LayerKind::LinearAttention => OperationKind::LinearAttention,
                    LayerKind::FullAttention => OperationKind::FullAttention,
                },
                snapdragon_owner: SnapdragonOwner::Adreno,
            });
            steps.push(GraphStep {
                layer: Some(layer),
                operation: OperationKind::QuantizedProjection,
                snapdragon_owner: SnapdragonOwner::Htp,
            });
            steps.push(GraphStep {
                layer: Some(layer),
                operation: OperationKind::SwiGlu,
                snapdragon_owner: SnapdragonOwner::Htp,
            });
        }
        steps.push(GraphStep {
            layer: None,
            operation: OperationKind::Mtp,
            snapdragon_owner: SnapdragonOwner::Htp,
        });
        Self {
            steps,
            mtp_resident: true,
            vision_resident: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_plan_keeps_heavy_ops_off_cpu() {
        let plan = GraphPlan::qwen38(&Qwen38Config::default());
        assert!(plan.mtp_resident);
        assert!(!plan.vision_resident);
        assert!(plan
            .steps
            .iter()
            .all(|step| step.snapdragon_owner != SnapdragonOwner::CpuControl));
        assert_eq!(
            plan.steps
                .iter()
                .filter(|step| step.operation == OperationKind::FullAttention)
                .count(),
            16
        );
    }
}
