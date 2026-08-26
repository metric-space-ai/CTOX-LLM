use crate::quant::{Q2Block64, Q4Block64, BLOCK_LEN, Q2_BLOCK_BYTES, Q4_BLOCK_BYTES};
use crate::{EngineError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    Q2,
    Q4,
}

pub const DEFAULT_KV_PAGE_TOKENS: usize = 128;
pub const DEFAULT_KV_SINK_TOKENS: usize = 128;
pub const DEFAULT_KV_RECENT_TOKENS: usize = 256;

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

/// One contiguous accelerator-transferable KV page. A page is always wholly
/// Q2 or Q4, so production kernels can dispatch without a precision branch per
/// token. Pages leave the Q4 recent window only when every token in the page is
/// old enough; the resulting boundary overhead is bounded by one page.
#[derive(Debug, Clone)]
struct QuantizedKvPage {
    precision: KvPrecision,
    maximum_tokens: usize,
    values_per_token: usize,
    bytes: Vec<u8>,
}

impl QuantizedKvPage {
    fn new(precision: KvPrecision, maximum_tokens: usize, values_per_token: usize) -> Result<Self> {
        let token_bytes = packed_bytes_for_values(values_per_token, precision)?;
        let capacity = maximum_tokens
            .checked_mul(token_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("KV page capacity overflows".into()))?;
        Ok(Self {
            precision,
            maximum_tokens,
            values_per_token,
            bytes: Vec::with_capacity(capacity),
        })
    }

    fn token_bytes(&self) -> usize {
        packed_bytes_for_values(self.values_per_token, self.precision)
            .expect("validated KV page geometry")
    }

    fn tokens(&self) -> usize {
        self.bytes.len() / self.token_bytes()
    }

    fn is_full(&self) -> bool {
        self.tokens() == self.maximum_tokens
    }

    fn push(&mut self, values: &[f32]) -> Result<()> {
        if self.is_full() || values.len() != self.values_per_token {
            return Err(EngineError::Shape(
                "KV page capacity or token geometry differs".into(),
            ));
        }
        append_quantized(&mut self.bytes, values, self.precision)
    }

    fn dequantize_token(&self, token: usize) -> Result<Vec<f32>> {
        if token >= self.tokens() {
            return Err(EngineError::Shape("KV page token is outside page".into()));
        }
        let token_bytes = self.token_bytes();
        dequantize_bytes(
            &self.bytes[token * token_bytes..(token + 1) * token_bytes],
            self.precision,
            self.values_per_token,
        )
    }

    fn requantize(&mut self, precision: KvPrecision) -> Result<()> {
        if self.precision == precision {
            return Ok(());
        }
        let tokens = self.tokens();
        let capacity = tokens
            .checked_mul(packed_bytes_for_values(self.values_per_token, precision)?)
            .ok_or_else(|| EngineError::MemoryBudget("requantized KV page overflows".into()))?;
        let mut bytes = Vec::with_capacity(capacity);
        for token in 0..tokens {
            append_quantized(&mut bytes, &self.dequantize_token(token)?, precision)?;
        }
        self.precision = precision;
        self.bytes = bytes;
        Ok(())
    }
}

fn packed_bytes_for_values(values: usize, precision: KvPrecision) -> Result<usize> {
    if values == 0 || !values.is_multiple_of(BLOCK_LEN) {
        return Err(EngineError::Shape(format!(
            "packed KV values must be a non-zero multiple of {BLOCK_LEN}"
        )));
    }
    let block_bytes = match precision {
        KvPrecision::Q2 => Q2_BLOCK_BYTES,
        KvPrecision::Q4 => Q4_BLOCK_BYTES,
    };
    values
        .checked_div(BLOCK_LEN)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| EngineError::MemoryBudget("packed KV byte count overflows".into()))
}

fn append_quantized(output: &mut Vec<u8>, values: &[f32], precision: KvPrecision) -> Result<()> {
    packed_bytes_for_values(values.len(), precision)?;
    for block in values.chunks_exact(BLOCK_LEN) {
        match precision {
            KvPrecision::Q2 => output.extend_from_slice(&Q2Block64::quantize(block)?.encode()),
            KvPrecision::Q4 => output.extend_from_slice(&Q4Block64::quantize(block)?.encode()),
        }
    }
    Ok(())
}

fn dequantize_bytes(bytes: &[u8], precision: KvPrecision, values: usize) -> Result<Vec<f32>> {
    if bytes.len() != packed_bytes_for_values(values, precision)? {
        return Err(EngineError::InvalidArtifact(
            "packed KV token byte count differs".into(),
        ));
    }
    let block_bytes = match precision {
        KvPrecision::Q2 => Q2_BLOCK_BYTES,
        KvPrecision::Q4 => Q4_BLOCK_BYTES,
    };
    let mut output = Vec::with_capacity(values);
    for block in bytes.chunks_exact(block_bytes) {
        match precision {
            KvPrecision::Q2 => output.extend_from_slice(&Q2Block64::decode(block)?.dequantize()),
            KvPrecision::Q4 => output.extend_from_slice(&Q4Block64::decode(block)?.dequantize()),
        }
    }
    Ok(output)
}

/// Reference implementation of the production paged KV layout. Each token is
/// stored as `[K heads, V heads]`; flattening transposes it into the head-major
/// layout consumed by the scalar grouped-query-attention oracle.
#[derive(Debug, Clone)]
pub struct PagedKvCache {
    maximum_tokens: usize,
    component_values: usize,
    page_tokens: usize,
    sink_tokens: usize,
    recent_tokens: usize,
    tokens: usize,
    pages: Vec<QuantizedKvPage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheUpdate {
    pub page_index: usize,
    pub token_in_page: usize,
    pub demoted_pages: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct KvPageView<'a> {
    pub page_index: usize,
    pub first_token: usize,
    pub tokens: usize,
    pub precision: KvPrecision,
    pub values_per_token: usize,
    /// Canonical little-endian Q2_B64 or Q4_B64 blocks, token-major as
    /// `[K heads, V heads]`. Backends may upload this slice directly.
    pub bytes: &'a [u8],
}

impl PagedKvCache {
    pub fn new(
        maximum_tokens: usize,
        component_values: usize,
        page_tokens: usize,
        sink_tokens: usize,
        recent_tokens: usize,
    ) -> Result<Self> {
        if maximum_tokens == 0
            || page_tokens == 0
            || sink_tokens > maximum_tokens
            || recent_tokens > maximum_tokens
            || !sink_tokens.is_multiple_of(page_tokens)
            || component_values == 0
            || !component_values.is_multiple_of(BLOCK_LEN)
        {
            return Err(EngineError::Shape(
                "invalid paged KV capacity, geometry, or retention policy".into(),
            ));
        }
        component_values
            .checked_mul(2)
            .ok_or_else(|| EngineError::MemoryBudget("combined KV width overflows".into()))?;
        Ok(Self {
            maximum_tokens,
            component_values,
            page_tokens,
            sink_tokens,
            recent_tokens,
            tokens: 0,
            pages: Vec::new(),
        })
    }

    pub fn qwen_default(maximum_tokens: usize, component_values: usize) -> Result<Self> {
        Self::new(
            maximum_tokens,
            component_values,
            DEFAULT_KV_PAGE_TOKENS,
            if maximum_tokens >= DEFAULT_KV_SINK_TOKENS {
                DEFAULT_KV_SINK_TOKENS
            } else {
                0
            },
            DEFAULT_KV_RECENT_TOKENS.min(maximum_tokens),
        )
    }

    pub fn push(&mut self, key: &[f32], value: &[f32]) -> Result<KvCacheUpdate> {
        if self.tokens >= self.maximum_tokens
            || key.len() != self.component_values
            || value.len() != self.component_values
        {
            return Err(EngineError::Shape(
                "paged KV append capacity or geometry differs".into(),
            ));
        }
        if self.pages.last().is_none_or(QuantizedKvPage::is_full) {
            self.pages.push(QuantizedKvPage::new(
                KvPrecision::Q4,
                self.page_tokens.min(self.maximum_tokens - self.tokens),
                self.component_values * 2,
            )?);
        }
        let mut combined = Vec::with_capacity(self.component_values * 2);
        combined.extend_from_slice(key);
        combined.extend_from_slice(value);
        self.pages
            .last_mut()
            .expect("a KV page was created")
            .push(&combined)?;
        self.tokens += 1;
        let page_index = self.pages.len() - 1;
        let token_in_page = self.pages[page_index].tokens() - 1;
        Ok(KvCacheUpdate {
            page_index,
            token_in_page,
            demoted_pages: self.demote_old_pages()?,
        })
    }

    fn demote_old_pages(&mut self) -> Result<Vec<usize>> {
        let recent_start = self.tokens.saturating_sub(self.recent_tokens);
        let mut demoted = Vec::new();
        for (page_index, page) in self.pages.iter_mut().enumerate() {
            let page_start = page_index * self.page_tokens;
            let page_end = page_start + page.tokens();
            if page.precision == KvPrecision::Q4
                && page_start >= self.sink_tokens
                && page_end <= recent_start
            {
                page.requantize(KvPrecision::Q2)?;
                demoted.push(page_index);
            }
        }
        Ok(demoted)
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn reset(&mut self) {
        self.pages = Vec::new();
        self.tokens = 0;
    }

    pub fn packed_bytes(&self) -> usize {
        self.pages.iter().map(|page| page.bytes.len()).sum()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.pages
            .iter()
            .map(|page| page.bytes.capacity())
            .sum::<usize>()
            + self.pages.capacity() * std::mem::size_of::<QuantizedKvPage>()
    }

    pub fn projected_packed_bytes(&self, tokens: usize) -> Result<usize> {
        if tokens > self.maximum_tokens {
            return Err(EngineError::MemoryBudget(
                "projected KV tokens exceed cache capacity".into(),
            ));
        }
        let recent_start = tokens.saturating_sub(self.recent_tokens);
        let combined_values = self
            .component_values
            .checked_mul(2)
            .ok_or_else(|| EngineError::MemoryBudget("projected KV width overflows".into()))?;
        let q2_token_bytes = packed_bytes_for_values(combined_values, KvPrecision::Q2)?;
        let q4_token_bytes = packed_bytes_for_values(combined_values, KvPrecision::Q4)?;
        let mut bytes = 0_usize;
        for page_start in (0..tokens).step_by(self.page_tokens) {
            let page_end = (page_start + self.page_tokens).min(tokens);
            let precision = if page_start < self.sink_tokens || page_end > recent_start {
                q4_token_bytes
            } else {
                q2_token_bytes
            };
            bytes =
                bytes
                    .checked_add((page_end - page_start).checked_mul(precision).ok_or_else(
                        || EngineError::MemoryBudget("projected KV page bytes overflow".into()),
                    )?)
                    .ok_or_else(|| {
                        EngineError::MemoryBudget("projected KV bytes overflow".into())
                    })?;
        }
        Ok(bytes)
    }

    pub fn q4_tokens(&self) -> usize {
        self.pages
            .iter()
            .filter(|page| page.precision == KvPrecision::Q4)
            .map(QuantizedKvPage::tokens)
            .sum()
    }

    pub fn page_views(&self) -> impl ExactSizeIterator<Item = KvPageView<'_>> {
        self.pages
            .iter()
            .enumerate()
            .map(|(page_index, page)| KvPageView {
                page_index,
                first_token: page_index * self.page_tokens,
                tokens: page.tokens(),
                precision: page.precision,
                values_per_token: page.values_per_token,
                bytes: &page.bytes,
            })
    }

    pub fn maximum_boundary_q4_tokens(&self) -> usize {
        self.sink_tokens
            .saturating_add(self.recent_tokens)
            .saturating_add(self.page_tokens.saturating_sub(1))
            .min(self.maximum_tokens)
    }

    pub fn flattened_key(&self, heads: usize, head_dim: usize) -> Result<Vec<f32>> {
        self.flattened_component(0, heads, head_dim)
    }

    pub fn flattened_value(&self, heads: usize, head_dim: usize) -> Result<Vec<f32>> {
        self.flattened_component(self.component_values, heads, head_dim)
    }

    fn flattened_component(
        &self,
        component_offset: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        if heads.checked_mul(head_dim) != Some(self.component_values) {
            return Err(EngineError::Shape(
                "paged KV flatten geometry differs".into(),
            ));
        }
        let mut output = vec![0.0; self.tokens * self.component_values];
        let mut global_token = 0;
        for page in &self.pages {
            for token in 0..page.tokens() {
                let values = page.dequantize_token(token)?;
                for head in 0..heads {
                    let source = component_offset + head * head_dim;
                    let target = (head * self.tokens + global_token) * head_dim;
                    output[target..target + head_dim]
                        .copy_from_slice(&values[source..source + head_dim]);
                }
                global_token += 1;
            }
        }
        Ok(output)
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

    #[test]
    fn paged_cache_demotes_only_pages_outside_sink_and_recent_windows() {
        let mut cache = PagedKvCache::new(16, 64, 4, 4, 4).unwrap();
        for token in 0..16 {
            let update = cache
                .push(&[token as f32; 64], &[-(token as f32); 64])
                .unwrap();
            if token == 11 {
                assert_eq!(update.demoted_pages, vec![1]);
            }
        }
        assert_eq!(cache.tokens(), 16);
        assert_eq!(cache.q4_tokens(), 8);
        assert_eq!(cache.maximum_boundary_q4_tokens(), 11);
        assert_eq!(
            cache.packed_bytes(),
            8 * 2 * Q4_BLOCK_BYTES + 8 * 2 * Q2_BLOCK_BYTES
        );
        assert_eq!(
            cache.projected_packed_bytes(16).unwrap(),
            cache.packed_bytes()
        );
        let views = cache.page_views().collect::<Vec<_>>();
        assert_eq!(views.len(), 4);
        assert_eq!(views[0].first_token, 0);
        assert_eq!(views[1].first_token, 4);
        assert_eq!(views[1].precision, KvPrecision::Q2);
        assert_eq!(views[3].tokens, 4);
        assert_eq!(views[3].bytes.len(), views[3].tokens * 2 * Q4_BLOCK_BYTES);
        let key = cache.flattened_key(1, 64).unwrap();
        let value = cache.flattened_value(1, 64).unwrap();
        assert_eq!(key.len(), 16 * 64);
        assert_eq!(value.len(), 16 * 64);
        assert!(key.iter().all(|value| value.is_finite()));
        assert!(value.iter().all(|value| value.is_finite()));
        cache.reset();
        assert_eq!(cache.tokens(), 0);
        assert_eq!(cache.packed_bytes(), 0);
        assert_eq!(cache.allocated_bytes(), 0);
    }

    #[test]
    fn paged_cache_flattens_token_major_pages_into_head_major_attention() {
        let mut cache = PagedKvCache::new(4, 128, 2, 0, 4).unwrap();
        for token in 0..4 {
            let mut key = vec![0.0; 128];
            let mut value = vec![0.0; 128];
            key[..64].fill((token + 1) as f32);
            key[64..].fill((token + 11) as f32);
            value[..64].fill(-((token + 1) as f32));
            value[64..].fill(-((token + 11) as f32));
            cache.push(&key, &value).unwrap();
        }
        let key = cache.flattened_key(2, 64).unwrap();
        let value = cache.flattened_value(2, 64).unwrap();
        for head in 0..2 {
            for token in 0..4 {
                let expected = (token + 1 + head * 10) as f32;
                let start = (head * 4 + token) * 64;
                assert!(key[start..start + 64]
                    .iter()
                    .all(|actual| (*actual - expected).abs() < 1e-3));
                assert!(value[start..start + 64]
                    .iter()
                    .all(|actual| (*actual + expected).abs() < 1e-3));
            }
        }
    }

    #[test]
    fn qwen_128k_packed_bytes_match_the_release_formula() {
        let config = crate::Qwen38Config::default();
        let component_values = config.num_key_value_heads * config.head_dim;
        let cache = PagedKvCache::qwen_default(131_072, component_values).unwrap();
        let per_layer_q2 = 131_072 * component_values * 2 / 64 * Q2_BLOCK_BYTES;
        let q4_delta = 384 * component_values * 2 / 64 * (Q4_BLOCK_BYTES - Q2_BLOCK_BYTES);
        assert_eq!(
            cache.projected_packed_bytes(131_072).unwrap(),
            per_layer_q2 + q4_delta
        );
    }
}
