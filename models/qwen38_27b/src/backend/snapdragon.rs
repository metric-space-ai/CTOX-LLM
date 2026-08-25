use crate::backend::{Backend, BackendKind, FusedMatVec, PromotionState};
use crate::{EngineError, Result};

/// Qualcomm contract. Proprietary QNN/Hexagon SDK objects are developer inputs
/// and never vendored into the public repository.
pub struct SnapdragonBackend;

impl Backend for SnapdragonBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Snapdragon
    }

    fn promotion_state(&self) -> PromotionState {
        PromotionState::Contract
    }

    fn profile(&self) -> &'static str {
        "qnn-htp-vulkan-contract"
    }

    fn fused_matvec(&self, _operation: &FusedMatVec<'_>) -> Result<Vec<f32>> {
        Err(EngineError::UnsupportedOperation {
            backend: "snapdragon",
            operation: "A8W2/A8W4 fused matvec",
            reason: "target SoC, QNN SDK, op package, and device verifier are not available".into(),
        })
    }
}
