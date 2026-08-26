pub mod cpu;
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_executor;
#[cfg(feature = "cuda")]
pub mod cuda_graph;
#[cfg(feature = "cuda")]
pub mod cuda_runtime;
pub mod cuda_schedule;
pub mod metal;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal_runtime;
pub mod snapdragon;

use serde::{Deserialize, Serialize};

use crate::format::{QuantSegment, TensorDType};
use crate::EngineError;
use crate::Result;
use half::f16;

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

/// Recovery scales can be supplied either by tests/host code as native f32 or
/// directly from the little-endian FP16 CTOXQ payload. Production loaders use
/// the latter so no full scale tensor is expanded or duplicated before a
/// fused kernel launch.
#[derive(Debug, Clone, Copy)]
pub enum ScaleSlice<'a> {
    F32(&'a [f32]),
    F16Le(&'a [u8]),
}

impl ScaleSlice<'_> {
    pub fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16Le(bytes) => bytes.len() / 2,
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn value(self, index: usize) -> Result<f32> {
        if index >= self.len() {
            return Err(EngineError::Shape(format!(
                "recovery scale index {index} exceeds {} values",
                self.len()
            )));
        }
        let value = match self {
            Self::F32(values) => values[index],
            Self::F16Le(bytes) => {
                let offset = index * 2;
                f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
            }
        };
        if !value.is_finite() {
            return Err(EngineError::InvalidArtifact(format!(
                "recovery scale {index} is non-finite"
            )));
        }
        Ok(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FusedMatVec<'a> {
    pub dtype: TensorDType,
    pub weights: &'a [u8],
    /// Exact row-segment layout from the CTOXQ manifest for
    /// `MixedQ2Q4B64`. Pure Q2/Q4 operations must leave this empty.
    pub segments: &'a [QuantSegment],
    pub rows: usize,
    pub columns: usize,
    pub input: &'a [f32],
    pub s_in: Option<ScaleSlice<'a>>,
    pub s_out: Option<ScaleSlice<'a>>,
    pub bias: Option<&'a [f32]>,
    pub activation: Activation,
}

#[derive(Debug, Clone, Copy)]
pub struct RecoveredRow<'a> {
    pub dtype: TensorDType,
    /// Exactly one packed matrix row, already resolved from a pure tensor or
    /// the containing mixed Q2/Q4 row segment.
    pub weights: &'a [u8],
    pub columns: usize,
    pub s_in: ScaleSlice<'a>,
    /// One widened row scale. Only this scalar is touched for an embedding
    /// lookup; the complete s_out tensor stays packed in the artifact.
    pub s_out: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct RecoveredRowMatVec<'a> {
    pub dtype: TensorDType,
    /// Exactly one packed row resolved from the canonical CTOXQ matrix.
    pub weights: &'a [u8],
    /// Input after applying the matrix-wide packed `s_in` exactly once.
    pub corrected_input: &'a [f32],
    pub s_out: f32,
}

pub trait Backend {
    fn kind(&self) -> BackendKind;
    fn promotion_state(&self) -> PromotionState;
    fn profile(&self) -> &'static str;
    fn fused_matvec(&self, operation: &FusedMatVec<'_>) -> Result<Vec<f32>>;
    /// Execute projections that consume one logical activation. Backends may
    /// share correction/quantization work when the inputs and `s_in` values
    /// are identical; the default preserves exact independent semantics.
    fn fused_matvec_fanout(&self, operations: &[FusedMatVec<'_>]) -> Result<Vec<Vec<f32>>> {
        operations
            .iter()
            .map(|operation| self.fused_matvec(operation))
            .collect()
    }
    fn recovered_row(&self, operation: &RecoveredRow<'_>) -> Result<Vec<f32>>;
    fn recovered_row_matvec(&self, operation: &RecoveredRowMatVec<'_>) -> Result<f32>;
}
