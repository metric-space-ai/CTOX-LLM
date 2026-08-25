use crate::quant::{Q2Block64, Q4Block64, BLOCK_LEN, Q2_BLOCK_BYTES, Q4_BLOCK_BYTES};
use crate::{EngineError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    Q2,
    Q4,
}

#[derive(Debug, Clone)]
pub struct QuantizedKvToken {
    precision: KvPrecision,
    values: usize,
    bytes: Vec<u8>,
}

impl QuantizedKvToken {
    pub fn quantize(values: &[f32], precision: KvPrecision) -> Result<Self> {
        if values.is_empty() || !values.len().is_multiple_of(BLOCK_LEN) {
            return Err(EngineError::Shape(format!(
                "KV token has {} values; length must be a non-zero multiple of {BLOCK_LEN}",
                values.len()
            )));
        }
        let block_bytes = match precision {
            KvPrecision::Q2 => Q2_BLOCK_BYTES,
            KvPrecision::Q4 => Q4_BLOCK_BYTES,
        };
        let mut bytes = Vec::with_capacity(values.len() / BLOCK_LEN * block_bytes);
        for block in values.chunks_exact(BLOCK_LEN) {
            match precision {
                KvPrecision::Q2 => bytes.extend_from_slice(&Q2Block64::quantize(block)?.encode()),
                KvPrecision::Q4 => bytes.extend_from_slice(&Q4Block64::quantize(block)?.encode()),
            }
        }
        Ok(Self {
            precision,
            values: values.len(),
            bytes,
        })
    }

    pub fn precision(&self) -> KvPrecision {
        self.precision
    }

    pub fn packed_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn dequantize(&self) -> Result<Vec<f32>> {
        let block_bytes = match self.precision {
            KvPrecision::Q2 => Q2_BLOCK_BYTES,
            KvPrecision::Q4 => Q4_BLOCK_BYTES,
        };
        let mut output = Vec::with_capacity(self.values);
        for block in self.bytes.chunks_exact(block_bytes) {
            match self.precision {
                KvPrecision::Q2 => {
                    output.extend_from_slice(&Q2Block64::decode(block)?.dequantize())
                }
                KvPrecision::Q4 => {
                    output.extend_from_slice(&Q4Block64::decode(block)?.dequantize())
                }
            }
        }
        Ok(output)
    }

    fn requantize(&mut self, precision: KvPrecision) -> Result<()> {
        if self.precision == precision {
            return Ok(());
        }
        *self = Self::quantize(&self.dequantize()?, precision)?;
        Ok(())
    }
}

/// Reference policy for the accelerator-resident mixed Q2/Q4 KV cache.
/// Snapdragon production code applies the same transitions in a Vulkan kernel.
pub struct MixedKvCache {
    max_tokens: usize,
    sink_tokens: usize,
    recent_tokens: usize,
    values_per_token: usize,
    tokens: Vec<QuantizedKvToken>,
}

impl MixedKvCache {
    pub fn new(
        max_tokens: usize,
        values_per_token: usize,
        sink_tokens: usize,
        recent_tokens: usize,
    ) -> Result<Self> {
        if max_tokens == 0 || sink_tokens + recent_tokens > max_tokens {
            return Err(EngineError::Shape(
                "invalid KV cache capacity/sink/recent policy".into(),
            ));
        }
        if values_per_token == 0 || !values_per_token.is_multiple_of(BLOCK_LEN) {
            return Err(EngineError::Shape(format!(
                "values_per_token must be a non-zero multiple of {BLOCK_LEN}"
            )));
        }
        Ok(Self {
            max_tokens,
            sink_tokens,
            recent_tokens,
            values_per_token,
            tokens: Vec::new(),
        })
    }

    pub fn push(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.values_per_token {
            return Err(EngineError::Shape(format!(
                "KV token has {} values, expected {}",
                values.len(),
                self.values_per_token
            )));
        }
        if self.tokens.len() == self.max_tokens {
            return Err(EngineError::MemoryBudget(format!(
                "KV cache reached {} tokens",
                self.max_tokens
            )));
        }
        self.tokens
            .push(QuantizedKvToken::quantize(values, KvPrecision::Q4)?);
        let length = self.tokens.len();
        if length > self.recent_tokens {
            let candidate = length - self.recent_tokens - 1;
            if candidate >= self.sink_tokens {
                self.tokens[candidate].requantize(KvPrecision::Q2)?;
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn token(&self, index: usize) -> Option<&QuantizedKvToken> {
        self.tokens.get(index)
    }

    pub fn packed_bytes(&self) -> usize {
        self.tokens.iter().map(QuantizedKvToken::packed_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_sink_and_recent_tokens_at_q4() {
        let mut cache = MixedKvCache::new(16, 64, 2, 3).unwrap();
        for token in 0..10 {
            let values = [token as f32 / 10.0; 64];
            cache.push(&values).unwrap();
        }
        assert_eq!(cache.token(0).unwrap().precision(), KvPrecision::Q4);
        assert_eq!(cache.token(1).unwrap().precision(), KvPrecision::Q4);
        assert_eq!(cache.token(2).unwrap().precision(), KvPrecision::Q2);
        assert_eq!(cache.token(6).unwrap().precision(), KvPrecision::Q2);
        assert_eq!(cache.token(7).unwrap().precision(), KvPrecision::Q4);
        assert_eq!(cache.token(9).unwrap().precision(), KvPrecision::Q4);
        assert_eq!(
            cache.packed_bytes(),
            5 * Q4_BLOCK_BYTES + 5 * Q2_BLOCK_BYTES
        );
    }

    #[test]
    fn refuses_to_exceed_capacity() {
        let mut cache = MixedKvCache::new(2, 64, 1, 1).unwrap();
        cache.push(&[0.0; 64]).unwrap();
        cache.push(&[0.0; 64]).unwrap();
        assert!(cache.push(&[0.0; 64]).is_err());
    }
}
