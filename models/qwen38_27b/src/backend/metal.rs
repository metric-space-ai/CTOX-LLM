use crate::backend::{Backend, BackendKind, FusedMatVec, PromotionState};
use crate::{EngineError, Result};

/// Metal contract. No MLX/MPSGraph inference fallback is permitted.
pub struct MetalBackend;

impl Backend for MetalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn promotion_state(&self) -> PromotionState {
        PromotionState::Contract
    }

    fn profile(&self) -> &'static str {
        "metal-contract"
    }

    fn fused_matvec(&self, _operation: &FusedMatVec<'_>) -> Result<Vec<f32>> {
        Err(EngineError::UnsupportedOperation {
            backend: "metal",
            operation: "q2/q4 fused matvec",
            reason: "MSL candidate has not passed verifier and same-device benchmark gates".into(),
        })
    }
}
