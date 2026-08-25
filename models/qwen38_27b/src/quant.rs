use half::f16;

use crate::{EngineError, Result};

pub const BLOCK_LEN: usize = 64;
pub const Q2_BLOCK_BYTES: usize = 2 + 16;
pub const Q4_BLOCK_BYTES: usize = 2 + 32;
pub const Q2_CODEBOOK: [f32; 4] = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Q2Block64 {
    pub scale: f16,
    pub codes: [u8; 16],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Q4Block64 {
    pub scale: f16,
    pub codes: [u8; 32],
}

fn require_block(values: &[f32]) -> Result<()> {
    if values.len() != BLOCK_LEN {
        return Err(EngineError::Shape(format!(
            "quant block must contain {BLOCK_LEN} values, got {}",
            values.len()
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::InvalidArtifact(
            "quantization input contains non-finite values".into(),
        ));
    }
    Ok(())
}

impl Q2Block64 {
    pub fn quantize(values: &[f32]) -> Result<Self> {
        require_block(values)?;
        let scale = values
            .iter()
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        let mut codes = [0_u8; 16];
        if scale == 0.0 {
            return Ok(Self {
                scale: f16::ZERO,
                codes,
            });
        }

        for (index, value) in values.iter().enumerate() {
            let normalized = *value / scale;
            let code = Q2_CODEBOOK
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    (normalized - **left)
                        .abs()
                        .total_cmp(&(normalized - **right).abs())
                })
                .map(|(code, _)| code as u8)
                .expect("Q2 codebook is not empty");
            codes[index / 4] |= code << ((index % 4) * 2);
        }
        Ok(Self {
            scale: f16::from_f32(scale),
            codes,
        })
    }

    #[inline]
    pub fn value(&self, index: usize) -> f32 {
        let code = (self.codes[index / 4] >> ((index % 4) * 2)) & 0x03;
        self.scale.to_f32() * Q2_CODEBOOK[code as usize]
    }

    pub fn dequantize(&self) -> [f32; BLOCK_LEN] {
        std::array::from_fn(|index| self.value(index))
    }

    pub fn encode(&self) -> [u8; Q2_BLOCK_BYTES] {
        let mut encoded = [0_u8; Q2_BLOCK_BYTES];
        encoded[..2].copy_from_slice(&self.scale.to_bits().to_le_bytes());
        encoded[2..].copy_from_slice(&self.codes);
        encoded
    }

    pub fn decode(encoded: &[u8]) -> Result<Self> {
        if encoded.len() != Q2_BLOCK_BYTES {
            return Err(EngineError::InvalidArtifact(format!(
                "Q2 block is {} bytes, expected {Q2_BLOCK_BYTES}",
                encoded.len()
            )));
        }
        let scale = f16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]]));
        if !scale.to_f32().is_finite() {
            return Err(EngineError::InvalidArtifact(
                "Q2 block scale is non-finite".into(),
            ));
        }
        let mut codes = [0_u8; 16];
        codes.copy_from_slice(&encoded[2..]);
        Ok(Self { scale, codes })
    }
}

impl Q4Block64 {
    pub fn quantize(values: &[f32]) -> Result<Self> {
        require_block(values)?;
        let scale = values
            .iter()
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        let mut codes = [0_u8; 32];
        if scale == 0.0 {
            return Ok(Self {
                scale: f16::ZERO,
                codes,
            });
        }

        for (index, value) in values.iter().enumerate() {
            // Sixteen uniformly spaced levels in [-1, 1]. The midpoint lies
            // between codes 7 and 8, matching the no-zero Q2 codebook.
            let normalized = (*value / scale).clamp(-1.0, 1.0);
            let code = ((normalized * 7.5) + 7.5).round().clamp(0.0, 15.0) as u8;
            codes[index / 2] |= code << ((index % 2) * 4);
        }
        Ok(Self {
            scale: f16::from_f32(scale),
            codes,
        })
    }

    #[inline]
    pub fn value(&self, index: usize) -> f32 {
        let code = (self.codes[index / 2] >> ((index % 2) * 4)) & 0x0f;
        let normalized = (code as f32 - 7.5) / 7.5;
        self.scale.to_f32() * normalized
    }

    pub fn dequantize(&self) -> [f32; BLOCK_LEN] {
        std::array::from_fn(|index| self.value(index))
    }

    pub fn encode(&self) -> [u8; Q4_BLOCK_BYTES] {
        let mut encoded = [0_u8; Q4_BLOCK_BYTES];
        encoded[..2].copy_from_slice(&self.scale.to_bits().to_le_bytes());
        encoded[2..].copy_from_slice(&self.codes);
        encoded
    }

    pub fn decode(encoded: &[u8]) -> Result<Self> {
        if encoded.len() != Q4_BLOCK_BYTES {
            return Err(EngineError::InvalidArtifact(format!(
                "Q4 block is {} bytes, expected {Q4_BLOCK_BYTES}",
                encoded.len()
            )));
        }
        let scale = f16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]]));
        if !scale.to_f32().is_finite() {
            return Err(EngineError::InvalidArtifact(
                "Q4 block scale is non-finite".into(),
            ));
        }
        let mut codes = [0_u8; 32];
        codes.copy_from_slice(&encoded[2..]);
        Ok(Self { scale, codes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> [f32; BLOCK_LEN] {
        std::array::from_fn(|index| ((index as f32 - 31.5) / 31.5).sin())
    }

    #[test]
    fn q2_encoding_is_deterministic() {
        let block = Q2Block64::quantize(&fixture()).unwrap();
        let decoded = Q2Block64::decode(&block.encode()).unwrap();
        assert_eq!(block, decoded);
        assert!(block
            .dequantize()
            .iter()
            .zip(fixture())
            .all(|(actual, expected)| (actual - expected).abs() <= 0.34));
    }

    #[test]
    fn q4_encoding_is_deterministic() {
        let block = Q4Block64::quantize(&fixture()).unwrap();
        let decoded = Q4Block64::decode(&block.encode()).unwrap();
        assert_eq!(block, decoded);
        assert!(block
            .dequantize()
            .iter()
            .zip(fixture())
            .all(|(actual, expected)| (actual - expected).abs() <= 0.07));
    }

    #[test]
    fn rejects_non_finite_input() {
        let mut values = fixture();
        values[3] = f32::NAN;
        assert!(Q2Block64::quantize(&values).is_err());
    }
}
