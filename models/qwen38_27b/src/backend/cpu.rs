use crate::backend::{
    Backend, BackendKind, ExecutionPolicy, FusedMatVec, PromotionState, RecoveredRow,
};
use crate::format::TensorDType;
use crate::quant::{BLOCK_LEN, Q2_BLOCK_BYTES, Q2_CODEBOOK, Q4_BLOCK_BYTES};
use crate::{EngineError, Result};
use half::f16;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{uint32x4_t, uint8x16_t};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__m256;

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

    fn validate(operation: &FusedMatVec<'_>) -> Result<usize> {
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
        if operation
            .s_out
            .is_some_and(|values| values.len() != operation.rows)
        {
            return Err(EngineError::Shape("s_out length differs from rows".into()));
        }
        if operation
            .bias
            .is_some_and(|values| values.len() != operation.rows)
        {
            return Err(EngineError::Shape("bias length differs from rows".into()));
        }
        let blocks_per_row = operation.columns / BLOCK_LEN;
        let block_bytes = match operation.dtype {
            TensorDType::Q2B64 => Some(Q2_BLOCK_BYTES),
            TensorDType::Q4B64 => Some(Q4_BLOCK_BYTES),
            TensorDType::MixedQ2Q4B64 => None,
            other => {
                return Err(EngineError::UnsupportedDType(format!("{other:?}")));
            }
        };
        if let Some(block_bytes) = block_bytes {
            if !operation.segments.is_empty() {
                return Err(EngineError::InvalidArtifact(
                    "pure Q2/Q4 operation declares mixed row segments".into(),
                ));
            }
            let expected = operation
                .rows
                .checked_mul(blocks_per_row)
                .and_then(|blocks| blocks.checked_mul(block_bytes))
                .ok_or_else(|| EngineError::Shape("weight buffer size overflows usize".into()))?;
            if operation.weights.len() != expected {
                return Err(EngineError::Shape(format!(
                    "weight buffer has {} bytes, expected {expected}",
                    operation.weights.len()
                )));
            }
            return Ok(blocks_per_row);
        }

        if operation.segments.is_empty() {
            return Err(EngineError::InvalidArtifact(
                "mixed Q2/Q4 operation has no row segments".into(),
            ));
        }
        let mut expected_row = 0_usize;
        let mut expected_offset = 0_usize;
        for (expected_group, segment) in operation.segments.iter().enumerate() {
            let group_index = usize::try_from(segment.group_index).map_err(|_| {
                EngineError::InvalidArtifact("mixed segment group index overflows usize".into())
            })?;
            let row_start = usize::try_from(segment.row_start).map_err(|_| {
                EngineError::InvalidArtifact("mixed segment row start overflows usize".into())
            })?;
            let row_end = usize::try_from(segment.row_end).map_err(|_| {
                EngineError::InvalidArtifact("mixed segment row end overflows usize".into())
            })?;
            let offset = usize::try_from(segment.offset).map_err(|_| {
                EngineError::InvalidArtifact("mixed segment offset overflows usize".into())
            })?;
            let length = usize::try_from(segment.length).map_err(|_| {
                EngineError::InvalidArtifact("mixed segment length overflows usize".into())
            })?;
            if group_index != expected_group
                || row_start != expected_row
                || row_end <= row_start
                || row_end > operation.rows
                || offset != expected_offset
            {
                return Err(EngineError::InvalidArtifact(format!(
                    "mixed Q2/Q4 operation has non-contiguous segment {}",
                    segment.group_index
                )));
            }
            let segment_block_bytes = match segment.dtype {
                TensorDType::Q2B64 => Q2_BLOCK_BYTES,
                TensorDType::Q4B64 => Q4_BLOCK_BYTES,
                other => {
                    return Err(EngineError::InvalidArtifact(format!(
                        "mixed Q2/Q4 segment {} has invalid dtype {other:?}",
                        segment.group_index
                    )));
                }
            };
            let expected_length = row_end
                .checked_sub(row_start)
                .and_then(|rows| rows.checked_mul(blocks_per_row))
                .and_then(|blocks| blocks.checked_mul(segment_block_bytes))
                .ok_or_else(|| EngineError::Shape("mixed segment size overflows usize".into()))?;
            if length != expected_length {
                return Err(EngineError::InvalidArtifact(format!(
                    "mixed Q2/Q4 segment {} has {length} bytes, expected {expected_length}",
                    segment.group_index
                )));
            }
            expected_row = row_end;
            expected_offset = expected_offset.checked_add(length).ok_or_else(|| {
                EngineError::Shape("mixed weight buffer size overflows usize".into())
            })?;
        }
        if expected_row != operation.rows || expected_offset != operation.weights.len() {
            return Err(EngineError::Shape(format!(
                "mixed segments cover {expected_row}/{} rows and {expected_offset}/{} bytes",
                operation.rows,
                operation.weights.len(),
            )));
        }
        Ok(blocks_per_row)
    }

    /// Dot product of one packed Q2 row against the corrected input.
    ///
    /// No heap allocation happens inside the block loop: block scales and
    /// code bytes are read straight from the packed weight slice and decoded
    /// into SIMD registers.
    fn row_sum_q2(&self, row: &[u8], input: &[f32]) -> Result<f32> {
        debug_assert_eq!(row.len() % Q2_BLOCK_BYTES, 0);
        debug_assert_eq!(input.len(), row.len() / Q2_BLOCK_BYTES * BLOCK_LEN);
        let mut sum = 0.0_f32;
        for (block_index, block) in row.chunks_exact(Q2_BLOCK_BYTES).enumerate() {
            let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            if !scale.is_finite() {
                return Err(EngineError::InvalidArtifact(
                    "Q2 block scale is non-finite".into(),
                ));
            }
            let codes: &[u8; 16] = block[2..]
                .try_into()
                .expect("Q2 block chunk carries 16 code bytes");
            let input_block: &[f32; BLOCK_LEN] = input
                [block_index * BLOCK_LEN..(block_index + 1) * BLOCK_LEN]
                .try_into()
                .expect("validated input length");
            sum += packed_q2_dot(self.profile, scale, codes, input_block);
        }
        Ok(sum)
    }

    /// Dot product of one packed Q4 row against the corrected input. Same
    /// allocation-free contract as [`Self::row_sum_q2`].
    fn row_sum_q4(&self, row: &[u8], input: &[f32]) -> Result<f32> {
        debug_assert_eq!(row.len() % Q4_BLOCK_BYTES, 0);
        debug_assert_eq!(input.len(), row.len() / Q4_BLOCK_BYTES * BLOCK_LEN);
        let mut sum = 0.0_f32;
        for (block_index, block) in row.chunks_exact(Q4_BLOCK_BYTES).enumerate() {
            let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            if !scale.is_finite() {
                return Err(EngineError::InvalidArtifact(
                    "Q4 block scale is non-finite".into(),
                ));
            }
            let codes: &[u8; 32] = block[2..]
                .try_into()
                .expect("Q4 block chunk carries 32 code bytes");
            let input_block: &[f32; BLOCK_LEN] = input
                [block_index * BLOCK_LEN..(block_index + 1) * BLOCK_LEN]
                .try_into()
                .expect("validated input length");
            sum += packed_q4_dot(self.profile, scale, codes, input_block);
        }
        Ok(sum)
    }

    fn decode_recovered_row(operation: &RecoveredRow<'_>) -> Result<Vec<f32>> {
        if operation.columns == 0 || !operation.columns.is_multiple_of(BLOCK_LEN) {
            return Err(EngineError::Shape(
                "recovered row columns must be non-zero and divisible by 64".into(),
            ));
        }
        if operation.s_in.len() != operation.columns || !operation.s_out.is_finite() {
            return Err(EngineError::Shape(
                "recovered row scale contract differs".into(),
            ));
        }
        let block_bytes = match operation.dtype {
            TensorDType::Q2B64 => Q2_BLOCK_BYTES,
            TensorDType::Q4B64 => Q4_BLOCK_BYTES,
            other => return Err(EngineError::UnsupportedDType(format!("{other:?}"))),
        };
        let expected = operation
            .columns
            .checked_div(BLOCK_LEN)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or_else(|| EngineError::Shape("recovered row size overflows usize".into()))?;
        if operation.weights.len() != expected {
            return Err(EngineError::Shape(format!(
                "recovered row has {} packed bytes, expected {expected}",
                operation.weights.len()
            )));
        }

        let mut output = Vec::with_capacity(operation.columns);
        for (block_index, block) in operation.weights.chunks_exact(block_bytes).enumerate() {
            let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            if !scale.is_finite() {
                return Err(EngineError::InvalidArtifact(
                    "recovered row block scale is non-finite".into(),
                ));
            }
            for local_column in 0..BLOCK_LEN {
                let normalized = match operation.dtype {
                    TensorDType::Q2B64 => {
                        let packed = block[2 + local_column / 4];
                        Q2_CODEBOOK[((packed >> ((local_column % 4) * 2)) & 0x3) as usize]
                    }
                    TensorDType::Q4B64 => {
                        let packed = block[2 + local_column / 2];
                        let code = if local_column.is_multiple_of(2) {
                            packed & 0x0f
                        } else {
                            packed >> 4
                        };
                        (f32::from(code) - 7.5) / 7.5
                    }
                    _ => unreachable!("dtype validated"),
                };
                let column = block_index * BLOCK_LEN + local_column;
                output.push(scale * normalized * operation.s_in.value(column)? * operation.s_out);
            }
        }
        Ok(output)
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
        let blocks_per_row = Self::validate(operation)?;
        // Apply s_in exactly once per operation. The previous implementation
        // rebuilt a scaled [f32; 64] input copy for every block of every row.
        let corrected;
        let input: &[f32] = match operation.s_in {
            Some(scales) => {
                corrected = operation
                    .input
                    .iter()
                    .enumerate()
                    .map(|(index, value)| Ok(value * scales.value(index)?))
                    .collect::<Result<Vec<f32>>>()?;
                &corrected
            }
            None => operation.input,
        };
        let mut output = vec![0.0_f32; operation.rows];
        let finish_row = |row: usize, mut sum: f32| -> Result<f32> {
            sum += operation.bias.map(|bias| bias[row]).unwrap_or(0.0);
            sum *= operation
                .s_out
                .map(|scales| scales.value(row))
                .transpose()?
                .unwrap_or(1.0);
            Ok(operation.activation.apply(sum))
        };
        match operation.dtype {
            dtype @ (TensorDType::Q2B64 | TensorDType::Q4B64) => {
                let block_bytes = match dtype {
                    TensorDType::Q2B64 => Q2_BLOCK_BYTES,
                    TensorDType::Q4B64 => Q4_BLOCK_BYTES,
                    _ => unreachable!(),
                };
                let row_bytes = blocks_per_row * block_bytes;
                for (row, output_value) in output.iter_mut().enumerate() {
                    let row_weights = &operation.weights[row * row_bytes..(row + 1) * row_bytes];
                    let sum = match dtype {
                        TensorDType::Q2B64 => self.row_sum_q2(row_weights, input)?,
                        TensorDType::Q4B64 => self.row_sum_q4(row_weights, input)?,
                        _ => unreachable!(),
                    };
                    *output_value = finish_row(row, sum)?;
                }
            }
            TensorDType::MixedQ2Q4B64 => {
                for segment in operation.segments {
                    let row_start = usize::try_from(segment.row_start)
                        .expect("mixed segment row start validated");
                    let row_end =
                        usize::try_from(segment.row_end).expect("mixed segment row end validated");
                    let offset =
                        usize::try_from(segment.offset).expect("mixed segment offset validated");
                    let length =
                        usize::try_from(segment.length).expect("mixed segment length validated");
                    let row_bytes = match segment.dtype {
                        TensorDType::Q2B64 => blocks_per_row * Q2_BLOCK_BYTES,
                        TensorDType::Q4B64 => blocks_per_row * Q4_BLOCK_BYTES,
                        _ => unreachable!("mixed segment dtype validated"),
                    };
                    let payload = &operation.weights[offset..offset + length];
                    for (local_row, output_value) in
                        output[row_start..row_end].iter_mut().enumerate()
                    {
                        let row = row_start + local_row;
                        let row_weights =
                            &payload[local_row * row_bytes..(local_row + 1) * row_bytes];
                        let sum = match segment.dtype {
                            TensorDType::Q2B64 => self.row_sum_q2(row_weights, input)?,
                            TensorDType::Q4B64 => self.row_sum_q4(row_weights, input)?,
                            _ => unreachable!("mixed segment dtype validated"),
                        };
                        *output_value = finish_row(row, sum)?;
                    }
                }
            }
            _ => unreachable!("dtype validated"),
        }
        Ok(output)
    }

    fn recovered_row(&self, operation: &RecoveredRow<'_>) -> Result<Vec<f32>> {
        Self::decode_recovered_row(operation)
    }
}

/// Direct packed Q2 dot: `sum(scale * codebook[code] * input)` without
/// materializing a dequantized `[f32; 64]` block. The `ScalarVerifier` arm
/// reproduces the oracle arithmetic exactly (sequential f32 accumulation of
/// `scale * Q2_CODEBOOK[code] * input`).
fn packed_q2_dot(
    profile: CpuProfile,
    scale: f32,
    codes: &[u8; 16],
    input: &[f32; BLOCK_LEN],
) -> f32 {
    match profile {
        CpuProfile::ScalarVerifier => scalar_packed_q2_dot(scale, codes, input),
        #[cfg(target_arch = "x86_64")]
        CpuProfile::Avx2 => {
            // SAFETY: this profile is constructed only after AVX2 runtime
            // detection.
            unsafe { avx2_packed_q2_dot(scale, codes, input) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        CpuProfile::Avx2 => unreachable!("AVX2 profile on non-x86 target"),
        #[cfg(target_arch = "aarch64")]
        CpuProfile::Neon => {
            // SAFETY: AArch64 requires Advanced SIMD.
            unsafe { neon_packed_q2_dot(scale, codes, input) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        CpuProfile::Neon => unreachable!("NEON profile on non-AArch64 target"),
    }
}

/// Direct packed Q4 dot: `sum(scale * (code - 7.5) / 7.5 * input)` without
/// materializing a dequantized `[f32; 64]` block.
fn packed_q4_dot(
    profile: CpuProfile,
    scale: f32,
    codes: &[u8; 32],
    input: &[f32; BLOCK_LEN],
) -> f32 {
    match profile {
        CpuProfile::ScalarVerifier => scalar_packed_q4_dot(scale, codes, input),
        #[cfg(target_arch = "x86_64")]
        CpuProfile::Avx2 => {
            // SAFETY: this profile is constructed only after AVX2 runtime
            // detection.
            unsafe { avx2_packed_q4_dot(scale, codes, input) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        CpuProfile::Avx2 => unreachable!("AVX2 profile on non-x86 target"),
        #[cfg(target_arch = "aarch64")]
        CpuProfile::Neon => {
            // SAFETY: AArch64 requires Advanced SIMD.
            unsafe { neon_packed_q4_dot(scale, codes, input) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        CpuProfile::Neon => unreachable!("NEON profile on non-AArch64 target"),
    }
}

fn scalar_packed_q2_dot(scale: f32, codes: &[u8; 16], input: &[f32; BLOCK_LEN]) -> f32 {
    let mut sum = 0.0_f32;
    for (index, x) in input.iter().enumerate() {
        let code = (codes[index / 4] >> ((index % 4) * 2)) & 0x03;
        sum += scale * Q2_CODEBOOK[code as usize] * x;
    }
    sum
}

fn scalar_packed_q4_dot(scale: f32, codes: &[u8; 32], input: &[f32; BLOCK_LEN]) -> f32 {
    let mut sum = 0.0_f32;
    for (index, x) in input.iter().enumerate() {
        let code = (codes[index / 2] >> ((index % 2) * 4)) & 0x0f;
        let normalized = (code as f32 - 7.5) / 7.5;
        sum += scale * normalized * x;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_horizontal_sum(vector: __m256) -> f32 {
    use std::arch::x86_64::*;
    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), vector);
    lanes.iter().sum()
}

/// AVX2 packed Q2 block dot.
///
/// Each little-endian u32 word of the 16 code bytes holds 16 two-bit codes:
/// code `l` sits at bits `2l`. Variable per-lane right shifts broadcast the
/// word and extract eight codes per vector. The dequantized value is
/// `scale * (2 * code - 3) / 3`, which is bit-identical to
/// `scale * Q2_CODEBOOK[code]` because `(2c - 3) / 3` and the codebook
/// constant are the correctly rounded results of the same real quotient.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_packed_q2_dot(scale: f32, codes: &[u8; 16], input: &[f32; BLOCK_LEN]) -> f32 {
    use std::arch::x86_64::*;
    let vscale = _mm256_set1_ps(scale);
    let divisor = _mm256_set1_ps(3.0);
    let mask = _mm256_set1_epi32(0x03);
    let offset = _mm256_set1_epi32(3);
    let shifts_lo = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    let shifts_hi = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
    let mut acc = _mm256_setzero_ps();
    for (word_index, word_bytes) in codes.chunks_exact(4).enumerate() {
        let word = i32::from_le_bytes([word_bytes[0], word_bytes[1], word_bytes[2], word_bytes[3]]);
        let broadcast = _mm256_set1_epi32(word);
        for (shifts, input_offset) in [
            (shifts_lo, word_index * 16),
            (shifts_hi, word_index * 16 + 8),
        ] {
            let code = _mm256_and_si256(_mm256_srlv_epi32(broadcast, shifts), mask);
            let centered = _mm256_sub_epi32(_mm256_slli_epi32::<1>(code), offset);
            let normalized = _mm256_div_ps(_mm256_cvtepi32_ps(centered), divisor);
            let value = _mm256_mul_ps(normalized, vscale);
            let x = _mm256_loadu_ps(input.as_ptr().add(input_offset));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(value, x));
        }
    }
    avx2_horizontal_sum(acc)
}

/// AVX2 packed Q4 block dot.
///
/// Code byte `j` packs the low nibble (code `2j`) and the high nibble
/// (code `2j + 1`), so even-indexed codes pair with even-indexed inputs.
/// Two contiguous input vectors are deinterleaved with `shuffle_ps` plus a
/// lane-fixing `permutevar8x32_ps`; codes are widened with `cvtepu8_epi32`.
/// The value `scale * (2 * code - 15) / 15` is bit-identical to the scalar
/// `scale * (code - 7.5) / 7.5` (same real quotient, correctly rounded).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_packed_q4_dot(scale: f32, codes: &[u8; 32], input: &[f32; BLOCK_LEN]) -> f32 {
    use std::arch::x86_64::*;
    let vscale = _mm256_set1_ps(scale);
    let divisor = _mm256_set1_ps(15.0);
    let nibble_mask = _mm_set1_epi8(0x0f);
    let offset = _mm256_set1_epi32(15);
    let lane_fix = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    let mut acc = _mm256_setzero_ps();
    for (group, byte_group) in codes.chunks_exact(8).enumerate() {
        let bytes = _mm_loadl_epi64(byte_group.as_ptr().cast());
        let even_codes = _mm256_cvtepu8_epi32(_mm_and_si128(bytes, nibble_mask));
        let odd_codes =
            _mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16::<4>(bytes), nibble_mask));
        let lower = _mm256_loadu_ps(input.as_ptr().add(group * 16));
        let upper = _mm256_loadu_ps(input.as_ptr().add(group * 16 + 8));
        let even_inputs =
            _mm256_permutevar8x32_ps(_mm256_shuffle_ps::<0b1000_1000>(lower, upper), lane_fix);
        let odd_inputs =
            _mm256_permutevar8x32_ps(_mm256_shuffle_ps::<0b1101_1101>(lower, upper), lane_fix);
        for (code_vector, x) in [(even_codes, even_inputs), (odd_codes, odd_inputs)] {
            let centered = _mm256_sub_epi32(_mm256_slli_epi32::<1>(code_vector), offset);
            let normalized = _mm256_div_ps(_mm256_cvtepi32_ps(centered), divisor);
            let value = _mm256_mul_ps(normalized, vscale);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(value, x));
        }
    }
    avx2_horizontal_sum(acc)
}

/// Widen sixteen u8 lanes to four u32x4 vectors, preserving lane order.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn widen_bytes(nibbles: uint8x16_t) -> [uint32x4_t; 4] {
    use std::arch::aarch64::*;
    let lo = vmovl_u8(vget_low_u8(nibbles));
    let hi = vmovl_u8(vget_high_u8(nibbles));
    [
        vmovl_u16(vget_low_u16(lo)),
        vmovl_u16(vget_high_u16(lo)),
        vmovl_u16(vget_low_u16(hi)),
        vmovl_u16(vget_high_u16(hi)),
    ]
}

/// NEON packed Q2 block dot. Mirrors the AVX2 scheme: broadcast each u32
/// code word and extract four codes per vector with per-lane negative
/// (right) shifts, then dequantize as `scale * (2 * code - 3) / 3`.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_packed_q2_dot(scale: f32, codes: &[u8; 16], input: &[f32; BLOCK_LEN]) -> f32 {
    use std::arch::aarch64::*;
    const SHIFT_LANES: [[i32; 4]; 4] = [
        [0, -2, -4, -6],
        [-8, -10, -12, -14],
        [-16, -18, -20, -22],
        [-24, -26, -28, -30],
    ];
    let mask = vdupq_n_u32(0x03);
    let vscale = vdupq_n_f32(scale);
    let divisor = vdupq_n_f32(3.0);
    let offset = vdupq_n_s32(3);
    let mut acc = vdupq_n_f32(0.0);
    for (word_index, word_bytes) in codes.chunks_exact(4).enumerate() {
        let word = u32::from_le_bytes([word_bytes[0], word_bytes[1], word_bytes[2], word_bytes[3]]);
        let broadcast = vdupq_n_u32(word);
        for (group, shift_lanes) in SHIFT_LANES.iter().enumerate() {
            let code = vandq_u32(vshlq_u32(broadcast, vld1q_s32(shift_lanes.as_ptr())), mask);
            let centered = vsubq_s32(vshlq_n_s32::<1>(vreinterpretq_s32_u32(code)), offset);
            let normalized = vdivq_f32(vcvtq_f32_s32(centered), divisor);
            let value = vmulq_f32(normalized, vscale);
            let x = vld1q_f32(input.as_ptr().add(word_index * 16 + group * 4));
            acc = vfmaq_f32(acc, value, x);
        }
    }
    vaddvq_f32(acc)
}

/// NEON packed Q4 block dot. Low nibbles are the even-indexed codes and pair
/// with even-indexed inputs; `vld2q_f32` deinterleaves the input pairs for
/// free. Values are `scale * (2 * code - 15) / 15` as in the AVX2 kernel.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_packed_q4_dot(scale: f32, codes: &[u8; 32], input: &[f32; BLOCK_LEN]) -> f32 {
    use std::arch::aarch64::*;
    let nibble_mask = vdupq_n_u8(0x0f);
    let vscale = vdupq_n_f32(scale);
    let divisor = vdupq_n_f32(15.0);
    let offset = vdupq_n_s32(15);
    let mut acc = vdupq_n_f32(0.0);
    for (half, byte_half) in codes.chunks_exact(16).enumerate() {
        let bytes = vld1q_u8(byte_half.as_ptr());
        let even_codes = widen_bytes(vandq_u8(bytes, nibble_mask));
        let odd_codes = widen_bytes(vshrq_n_u8::<4>(bytes));
        for (quad, (even, odd)) in even_codes.iter().zip(odd_codes.iter()).enumerate() {
            let pair = vld2q_f32(input.as_ptr().add(half * 32 + quad * 8));
            for (code_vector, x) in [(*even, pair.0), (*odd, pair.1)] {
                let centered =
                    vsubq_s32(vshlq_n_s32::<1>(vreinterpretq_s32_u32(code_vector)), offset);
                let normalized = vdivq_f32(vcvtq_f32_s32(centered), divisor);
                let value = vmulq_f32(normalized, vscale);
                acc = vfmaq_f32(acc, value, x);
            }
        }
    }
    vaddvq_f32(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Activation, ScaleSlice};
    use crate::format::QuantSegment;
    use crate::quant::{Q2Block64, Q4Block64};

    /// SIMD lanes accumulate in a different order than the sequential scalar
    /// oracle and the NEON kernels use fused multiply-add, so bitwise
    /// equality is not expected. The difference is pure reassociation/FMA
    /// rounding of O(1) terms over at most a few hundred products per row,
    /// which stays far below this absolute tolerance.
    /// See docs/CPU_KERNEL_NOTES.md.
    const SIMD_TOLERANCE: f32 = 2e-4;

    fn detected() -> CpuBackend {
        CpuBackend::detect(ExecutionPolicy::Verifier).unwrap()
    }

    fn encode_fixture(dtype: TensorDType, rows: usize, columns: usize) -> Vec<u8> {
        let mut weights = Vec::new();
        for matrix_block in 0..rows * (columns / BLOCK_LEN) {
            let values: [f32; BLOCK_LEN] = std::array::from_fn(|index| {
                let phase = (matrix_block * BLOCK_LEN + index) as f32;
                (phase * 0.013_579).sin() * 0.75
            });
            match dtype {
                TensorDType::Q2B64 => {
                    weights.extend_from_slice(&Q2Block64::quantize(&values).unwrap().encode())
                }
                TensorDType::Q4B64 => {
                    weights.extend_from_slice(&Q4Block64::quantize(&values).unwrap().encode())
                }
                _ => unreachable!(),
            }
        }
        weights
    }

    /// Deterministic fused-operation comparison of the detected SIMD profile
    /// against the scalar oracle: multiple rows and blocks, non-identity
    /// s_in/s_out, bias, and a configurable activation.
    fn fused_case(dtype: TensorDType, rows: usize, columns: usize, activation: Activation) {
        let weights = encode_fixture(dtype, rows, columns);
        let input: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.031_25).cos())
            .collect();
        let s_in: Vec<f32> = (0..columns)
            .map(|index| 0.75 + 0.005 * (index % 7) as f32)
            .collect();
        let s_out: Vec<f32> = (0..rows)
            .map(|row| 1.25 - 0.05 * (row % 5) as f32)
            .collect();
        let bias: Vec<f32> = (0..rows)
            .map(|row| (row as f32 * 1.7).sin() * 0.1)
            .collect();
        let operation = FusedMatVec {
            dtype,
            weights: &weights,
            segments: &[],
            rows,
            columns,
            input: &input,
            s_in: Some(ScaleSlice::F32(&s_in)),
            s_out: Some(ScaleSlice::F32(&s_out)),
            bias: Some(&bias),
            activation,
        };
        let scalar = CpuBackend::scalar_verifier()
            .fused_matvec(&operation)
            .unwrap();
        let simd = detected().fused_matvec(&operation).unwrap();
        for row in 0..rows {
            assert!(
                (scalar[row] - simd[row]).abs() <= SIMD_TOLERANCE,
                "{dtype:?} row {row}: scalar {} vs simd {}",
                scalar[row],
                simd[row]
            );
        }
    }

    #[test]
    fn q2_multirow_silu_matches_scalar() {
        fused_case(TensorDType::Q2B64, 7, 3 * BLOCK_LEN, Activation::Silu);
    }

    #[test]
    fn q4_multirow_silu_matches_scalar() {
        fused_case(TensorDType::Q4B64, 7, 3 * BLOCK_LEN, Activation::Silu);
    }

    #[test]
    fn q2_single_block_identity_matches_scalar() {
        fused_case(TensorDType::Q2B64, 3, BLOCK_LEN, Activation::Identity);
    }

    #[test]
    fn q4_single_block_identity_matches_scalar() {
        fused_case(TensorDType::Q4B64, 3, BLOCK_LEN, Activation::Identity);
    }

    #[test]
    fn packed_f16_recovery_scales_match_host_f32_scales() {
        let weights = encode_fixture(TensorDType::Q2B64, 2, BLOCK_LEN);
        let input: Vec<f32> = (0..BLOCK_LEN)
            .map(|index| (index as f32 * 0.0625).sin())
            .collect();
        let s_in: Vec<f32> = (0..BLOCK_LEN)
            .map(|index| f16::from_f32(0.75 + (index % 9) as f32 * 0.03125).to_f32())
            .collect();
        let s_out = [f16::from_f32(1.25).to_f32(), f16::from_f32(0.875).to_f32()];
        let s_in_f16: Vec<u8> = s_in
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        let s_out_f16: Vec<u8> = s_out
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        let f32_operation = FusedMatVec {
            dtype: TensorDType::Q2B64,
            weights: &weights,
            segments: &[],
            rows: 2,
            columns: BLOCK_LEN,
            input: &input,
            s_in: Some(ScaleSlice::F32(&s_in)),
            s_out: Some(ScaleSlice::F32(&s_out)),
            bias: None,
            activation: Activation::Identity,
        };
        let f16_operation = FusedMatVec {
            s_in: Some(ScaleSlice::F16Le(&s_in_f16)),
            s_out: Some(ScaleSlice::F16Le(&s_out_f16)),
            ..f32_operation
        };
        for backend in [CpuBackend::scalar_verifier(), detected()] {
            assert_eq!(
                backend.fused_matvec(&f16_operation).unwrap(),
                backend.fused_matvec(&f32_operation).unwrap()
            );
        }
    }

    #[test]
    fn mixed_q2_q4_rows_match_their_pure_kernels() {
        let columns = 2 * BLOCK_LEN;
        let rows_per_segment = 2;
        let q2 = encode_fixture(TensorDType::Q2B64, rows_per_segment, columns);
        let q4 = encode_fixture(TensorDType::Q4B64, rows_per_segment, columns);
        let q2_length = u64::try_from(q2.len()).unwrap();
        let q4_length = u64::try_from(q4.len()).unwrap();
        let mut weights = q2.clone();
        weights.extend_from_slice(&q4);
        let segments = [
            QuantSegment {
                group_index: 0,
                row_start: 0,
                row_end: 2,
                dtype: TensorDType::Q2B64,
                offset: 0,
                length: q2_length,
            },
            QuantSegment {
                group_index: 1,
                row_start: 2,
                row_end: 4,
                dtype: TensorDType::Q4B64,
                offset: q2_length,
                length: q4_length,
            },
        ];
        let input: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.023_437_5).cos())
            .collect();
        let mixed = FusedMatVec {
            dtype: TensorDType::MixedQ2Q4B64,
            weights: &weights,
            segments: &segments,
            rows: 4,
            columns,
            input: &input,
            s_in: None,
            s_out: None,
            bias: None,
            activation: Activation::Identity,
        };
        let q2_operation = FusedMatVec {
            dtype: TensorDType::Q2B64,
            weights: &q2,
            segments: &[],
            rows: rows_per_segment,
            columns,
            input: &input,
            s_in: None,
            s_out: None,
            bias: None,
            activation: Activation::Identity,
        };
        let q4_operation = FusedMatVec {
            dtype: TensorDType::Q4B64,
            weights: &q4,
            segments: &[],
            rows: rows_per_segment,
            columns,
            input: &input,
            s_in: None,
            s_out: None,
            bias: None,
            activation: Activation::Identity,
        };

        for backend in [CpuBackend::scalar_verifier(), detected()] {
            let output = backend.fused_matvec(&mixed).unwrap();
            let q2_output = backend.fused_matvec(&q2_operation).unwrap();
            let q4_output = backend.fused_matvec(&q4_operation).unwrap();
            assert_eq!(&output[..2], q2_output.as_slice());
            assert_eq!(&output[2..], q4_output.as_slice());
        }
    }

    #[test]
    fn mixed_q2_q4_rejects_non_contiguous_manifest_segments() {
        let columns = BLOCK_LEN;
        let weights = encode_fixture(TensorDType::Q2B64, 2, columns);
        let mut segments = [QuantSegment {
            group_index: 0,
            row_start: 0,
            row_end: 2,
            dtype: TensorDType::Q2B64,
            offset: 0,
            length: u64::try_from(weights.len()).unwrap(),
        }];
        segments[0].row_start = 1;
        let input = [1.0_f32; BLOCK_LEN];
        let operation = FusedMatVec {
            dtype: TensorDType::MixedQ2Q4B64,
            weights: &weights,
            segments: &segments,
            rows: 2,
            columns,
            input: &input,
            s_in: None,
            s_out: None,
            bias: None,
            activation: Activation::Identity,
        };
        assert!(matches!(
            detected().fused_matvec(&operation),
            Err(EngineError::InvalidArtifact(_))
        ));
    }

    /// Exercise the packed decoders with arbitrary code bytes (not only the
    /// distributions `quantize` produces) against block-wise dequantization.
    #[test]
    fn packed_q2_decodes_arbitrary_codes() {
        let profile = detected().profile_kind();
        let codes: [u8; 16] =
            std::array::from_fn(|index| (index as u8).wrapping_mul(0x45).wrapping_add(0x1b));
        let scale = f16::from_f32(0.625);
        let input: [f32; BLOCK_LEN] =
            std::array::from_fn(|index| (index as f32 * 0.11).sin() + 0.25);
        let block = Q2Block64 { scale, codes };
        let reference = block
            .dequantize()
            .iter()
            .zip(&input)
            .map(|(weight, x)| weight * x)
            .sum::<f32>();
        let packed = packed_q2_dot(profile, scale.to_f32(), &codes, &input);
        assert!(
            (reference - packed).abs() <= 1e-5,
            "scalar {reference} vs packed {packed}"
        );
    }

    #[test]
    fn packed_q4_decodes_arbitrary_codes() {
        let profile = detected().profile_kind();
        let codes: [u8; 32] =
            std::array::from_fn(|index| (index as u8).wrapping_mul(0x9e).wrapping_add(0x3d));
        let scale = f16::from_f32(0.375);
        let input: [f32; BLOCK_LEN] =
            std::array::from_fn(|index| (index as f32 * 0.07).cos() - 0.125);
        let block = Q4Block64 { scale, codes };
        let reference = block
            .dequantize()
            .iter()
            .zip(&input)
            .map(|(weight, x)| weight * x)
            .sum::<f32>();
        let packed = packed_q4_dot(profile, scale.to_f32(), &codes, &input);
        assert!(
            (reference - packed).abs() <= 1e-5,
            "scalar {reference} vs packed {packed}"
        );
    }

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
            segments: &[],
            rows: 1,
            columns: BLOCK_LEN,
            input: &input,
            s_in: None,
            s_out: Some(ScaleSlice::F32(&[2.0])),
            bias: Some(&[1.0]),
            activation: Activation::Identity,
        };
        let scalar = CpuBackend::scalar_verifier()
            .fused_matvec(&operation)
            .unwrap();
        let detected = detected().fused_matvec(&operation).unwrap();
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
