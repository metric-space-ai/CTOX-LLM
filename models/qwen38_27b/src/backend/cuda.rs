use crate::backend::{Backend, BackendKind, FusedMatVec, PromotionState};
use crate::{EngineError, Result};

/// CUDA contract. Kernel promotion remains fail-closed until vendored kernels,
/// immutable pins, per-op verifiers, and SM86 benchmark evidence land together.
pub struct CudaBackend;

impl Backend for CudaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn promotion_state(&self) -> PromotionState {
        PromotionState::Contract
    }

    fn profile(&self) -> &'static str {
        "cuda-contract"
    }

    fn fused_matvec(&self, _operation: &FusedMatVec<'_>) -> Result<Vec<f32>> {
        Err(EngineError::UnsupportedOperation {
            backend: "cuda",
            operation: "q2/q4 fused matvec",
            reason: "kernel has not passed the SM86 verifier and benchmark gates".into(),
        })
    }
}
