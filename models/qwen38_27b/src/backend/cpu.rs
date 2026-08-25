use crate::backend::{Backend, BackendKind, ExecutionPolicy, FusedMatVec, PromotionState};
use crate::format::TensorDType;
use crate::quant::{Q2Block64, Q4Block64, BLOCK_LEN, Q2_BLOCK_BYTES, Q4_BLOCK_BYTES};
use crate::{EngineError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuProfile {
    ScalarVerifier,
    Avx2,
    Neon,
}

pub struct CpuBackend {
    profile: CpuProfile,
}

impl CpuBackend {
    pub fn detect(policy: ExecutionPolicy) -> Result<Self> {
        detect_profile(policy).map(|profile| Self { profile })
    }

    pub fn scalar_verifier() -> Self {
        Self {
            profile: CpuProfile::ScalarVerifier,
        }
    }

    pub fn profile_kind(&self) -> CpuProfile {
        self.profile
    }

    fn validate(operation: &FusedMatVec<'_>) -> Result<(usize, usize)> {
        if operation.columns == 0
            || operation.rows == 0
            || !operation.columns.is_multiple_of(BLOCK_LEN)
        {
            return Err(EngineError::Shape(
                "Q2/Q4 fused matvec dimensions must be non-zero and columns divisible by 64".into(),
            ));
        }
        if operation.input.len() != operation.columns {
            return Err(EngineError::Shape(format!(
                "input has {}, expected {} values",
                operation.input.len(),
                operation.columns
            )));
        }
        if let Some(scales) = operation.s_in {
            if scales.len() != operation.columns {
                return Err(EngineError::Shape(
                    "s_in length differs from columns".into(),
                ));
            }
        }
        for (label, values) in [("s_out", operation.s_out), ("bias", operation.bias)] {
            if let Some(values) = values {
                if values.len() != operation.rows {
                    return Err(EngineError::Shape(format!(
                        "{label} length differs from rows"
                    )));
                }
            }
        }
        let block_bytes = match operation.dtype {
            TensorDType::Q2B64 => Q2_BLOCK_BYTES,
            TensorDType::Q4B64 => Q4_BLOCK_BYTES,
            other => {
                return Err(EngineError::UnsupportedDType(format!("{other:?}")));
            }
        };
        let blocks_per_row = operation.columns / BLOCK_LEN;
        let expected = operation.rows * blocks_per_row * block_bytes;
        if operation.weights.len() != expected {
            return Err(EngineError::Shape(format!(
                "weight buffer has {} bytes, expected {expected}",
                operation.weights.len()
            )));
        }
        Ok((blocks_per_row, block_bytes))
    }

    fn input_block(operation: &FusedMatVec<'_>, start: usize) -> [f32; BLOCK_LEN] {
        std::array::from_fn(|index| {
            let position = start + index;
            operation.input[position] * operation.s_in.map(|scales| scales[position]).unwrap_or(1.0)
        })
    }

    fn block_dot(&self, weights: &[f32; BLOCK_LEN], input: &[f32; BLOCK_LEN]) -> f32 {
        match self.profile {
            CpuProfile::ScalarVerifier => scalar_dot(weights, input),
            #[cfg(target_arch = "x86_64")]
            CpuProfile::Avx2 => {
                // SAFETY: this profile is constructed only after AVX2 runtime
                // detection.
                unsafe { avx2_dot(weights, input) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            CpuProfile::Avx2 => unreachable!("AVX2 profile on non-x86 target"),
            #[cfg(target_arch = "aarch64")]
            CpuProfile::Neon => {
                // SAFETY: AArch64 requires Advanced SIMD.
                unsafe { neon_dot(weights, input) }
            }
            #[cfg(not(target_arch = "aarch64"))]
            CpuProfile::Neon => unreachable!("NEON profile on non-AArch64 target"),
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_profile(policy: ExecutionPolicy) -> Result<CpuProfile> {
    if std::arch::is_x86_feature_detected!("avx2") {
        return Ok(CpuProfile::Avx2);
    }
    verifier_or_error(policy)
}

#[cfg(target_arch = "aarch64")]
fn detect_profile(_policy: ExecutionPolicy) -> Result<CpuProfile> {
    Ok(CpuProfile::Neon)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_profile(policy: ExecutionPolicy) -> Result<CpuProfile> {
    verifier_or_error(policy)
}

#[cfg(not(target_arch = "aarch64"))]
fn verifier_or_error(policy: ExecutionPolicy) -> Result<CpuProfile> {
    if policy == ExecutionPolicy::Verifier {
        return Ok(CpuProfile::ScalarVerifier);
    }
    Err(EngineError::UnsupportedOperation {
        backend: "cpu",
        operation: "backend initialization",
        reason: "no verified SIMD profile and scalar fallback is forbidden".into(),
    })
}

impl Backend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn promotion_state(&self) -> PromotionState {
        match self.profile {
            CpuProfile::ScalarVerifier => PromotionState::Verifier,
            CpuProfile::Avx2 | CpuProfile::Neon => PromotionState::Experimental,
        }
    }

    fn profile(&self) -> &'static str {
        match self.profile {
            CpuProfile::ScalarVerifier => "scalar-verifier",
            CpuProfile::Avx2 => "x86_64-avx2",
            CpuProfile::Neon => "aarch64-neon",
        }
    }

    fn fused_matvec(&self, operation: &FusedMatVec<'_>) -> Result<Vec<f32>> {
        let (blocks_per_row, block_bytes) = Self::validate(operation)?;
        let mut output = vec![0.0_f32; operation.rows];
        for (row, output_value) in output.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for block_index in 0..blocks_per_row {
                let matrix_block = row * blocks_per_row + block_index;
                let byte_start = matrix_block * block_bytes;
                let input_start = block_index * BLOCK_LEN;
                let input = Self::input_block(operation, input_start);
                let weights = match operation.dtype {
                    TensorDType::Q2B64 => {
                        Q2Block64::decode(&operation.weights[byte_start..byte_start + block_bytes])?
                            .dequantize()
                    }
                    TensorDType::Q4B64 => {
                        Q4Block64::decode(&operation.weights[byte_start..byte_start + block_bytes])?
                            .dequantize()
                    }
                    _ => unreachable!("dtype validated"),
                };
                sum += self.block_dot(&weights, &input);
            }
            sum += operation.bias.map(|bias| bias[row]).unwrap_or(0.0);
            sum *= operation.s_out.map(|scales| scales[row]).unwrap_or(1.0);
            *output_value = operation.activation.apply(sum);
        }
        Ok(output)
    }
}

#[inline]
fn scalar_dot(left: &[f32; BLOCK_LEN], right: &[f32; BLOCK_LEN]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dot(left: &[f32; BLOCK_LEN], right: &[f32; BLOCK_LEN]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    for offset in (0..BLOCK_LEN).step_by(8) {
        let lhs = _mm256_loadu_ps(left.as_ptr().add(offset));
        let rhs = _mm256_loadu_ps(right.as_ptr().add(offset));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(lhs, rhs));
    }
    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    lanes.iter().sum()
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_dot(left: &[f32; BLOCK_LEN], right: &[f32; BLOCK_LEN]) -> f32 {
    use std::arch::aarch64::*;
    let mut sum = vdupq_n_f32(0.0);
    for offset in (0..BLOCK_LEN).step_by(4) {
        let lhs = vld1q_f32(left.as_ptr().add(offset));
        let rhs = vld1q_f32(right.as_ptr().add(offset));
        sum = vfmaq_f32(sum, lhs, rhs);
    }
    vaddvq_f32(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Activation;
    use crate::quant::{Q2Block64, Q4Block64};

    fn run(dtype: TensorDType) {
        let source: [f32; BLOCK_LEN] = std::array::from_fn(|index| (index as f32 - 31.5) / 31.5);
        let weights = match dtype {
            TensorDType::Q2B64 => Q2Block64::quantize(&source).unwrap().encode().to_vec(),
            TensorDType::Q4B64 => Q4Block64::quantize(&source).unwrap().encode().to_vec(),
            _ => unreachable!(),
        };
        let input = [1.0_f32; BLOCK_LEN];
        let operation = FusedMatVec {
            dtype,
            weights: &weights,
            rows: 1,
            columns: BLOCK_LEN,
            input: &input,
            s_in: None,
            s_out: Some(&[2.0]),
            bias: Some(&[1.0]),
            activation: Activation::Identity,
        };
        let scalar = CpuBackend::scalar_verifier()
            .fused_matvec(&operation)
            .unwrap();
        let detected = CpuBackend::detect(ExecutionPolicy::Verifier)
            .unwrap()
            .fused_matvec(&operation)
            .unwrap();
        assert!((scalar[0] - detected[0]).abs() < 1e-5);
    }

    #[test]
    fn detected_q2_matches_scalar() {
        run(TensorDType::Q2B64);
    }

    #[test]
    fn detected_q4_matches_scalar() {
        run(TensorDType::Q4B64);
    }
}
