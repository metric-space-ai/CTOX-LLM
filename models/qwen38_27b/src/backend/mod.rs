pub mod cpu;
pub mod cuda;
pub mod metal;
pub mod snapdragon;

use serde::{Deserialize, Serialize};

use crate::format::TensorDType;
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
    Snapdragon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionState {
    Unavailable,
    Contract,
    Verifier,
    Experimental,
    Optimized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPolicy {
    /// Permits scalar correctness kernels. Tests and offline verification only.
    Verifier,
    /// Refuses scalar or unverified fallback paths.
    Production,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Identity,
    Silu,
}

impl Activation {
    #[inline]
    pub fn apply(self, value: f32) -> f32 {
        match self {
            Self::Identity => value,
            Self::Silu => value / (1.0 + (-value).exp()),
        }
    }
}

pub struct FusedMatVec<'a> {
    pub dtype: TensorDType,
    pub weights: &'a [u8],
    pub rows: usize,
    pub columns: usize,
    pub input: &'a [f32],
    pub s_in: Option<&'a [f32]>,
    pub s_out: Option<&'a [f32]>,
    pub bias: Option<&'a [f32]>,
    pub activation: Activation,
}

pub trait Backend {
    fn kind(&self) -> BackendKind;
    fn promotion_state(&self) -> PromotionState;
    fn profile(&self) -> &'static str;
    fn fused_matvec(&self, operation: &FusedMatVec<'_>) -> Result<Vec<f32>>;
}
