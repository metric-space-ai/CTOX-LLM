use crate::{EngineError, Result};

#[derive(Debug, Clone, Copy)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: 0,
        }
    }
}

pub struct Sampler {
    config: SamplerConfig,
    rng: Pcg32,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Result<Self> {
        if !config.temperature.is_finite() || config.temperature < 0.0 {
            return Err(EngineError::InvalidArtifact(
                "temperature must be finite and non-negative".into(),
            ));
        }
        if !(config.top_p.is_finite() && 0.0 < config.top_p && config.top_p <= 1.0) {
            return Err(EngineError::InvalidArtifact(
                "top_p must be in (0, 1]".into(),
            ));
        }
        Ok(Self {
            config,
            rng: Pcg32::new(config.seed),
        })
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<usize> {
        if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidArtifact(
                "sampler logits are empty or non-finite".into(),
            ));
        }
        if self.config.temperature == 0.0 {
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .expect("logits are not empty")
                .0);
        }

        let mut candidates: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(token, logit)| (token, *logit / self.config.temperature))
            .collect();
        candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        if self.config.top_k > 0 {
            candidates.truncate(self.config.top_k.min(candidates.len()));
        }
        let maximum = candidates[0].1;
        let normalization: f32 = candidates
            .iter_mut()
            .map(|(_, value)| {
                *value = (*value - maximum).exp();
                *value
            })
            .sum();
        for (_, probability) in &mut candidates {
            *probability /= normalization;
        }

        let mut cumulative = 0.0_f32;
        let mut nucleus_len = candidates.len();
        for (index, (_, probability)) in candidates.iter().enumerate() {
            cumulative += *probability;
            if cumulative >= self.config.top_p {
                nucleus_len = index + 1;
                break;
            }
        }
        candidates.truncate(nucleus_len);
        let nucleus_total: f32 = candidates.iter().map(|(_, probability)| probability).sum();
        let draw = self.rng.next_f32() * nucleus_total;
        let mut cumulative = 0.0_f32;
        for (token, probability) in &candidates {
            cumulative += *probability;
            if draw <= cumulative {
                return Ok(*token);
            }
        }
        Ok(candidates.last().expect("nucleus is not empty").0)
    }
}

/// Small deterministic PCG-XSH-RR generator so sampler parity does not depend
/// on a platform RNG implementation.
struct Pcg32 {
    state: u64,
}

impl Pcg32 {
    fn new(seed: u64) -> Self {
        let mut generator = Self {
            state: seed.wrapping_add(0x853c_49e6_748f_ea9b),
        };
        let _ = generator.next_u32();
        generator
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let shifted = (((old >> 18) ^ old) >> 27) as u32;
        shifted.rotate_right((old >> 59) as u32)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64 + 1.0)) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_argmax() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        })
        .unwrap();
        assert_eq!(sampler.sample(&[-1.0, 3.0, 2.0]).unwrap(), 1);
    }

    #[test]
    fn same_seed_produces_same_sequence() {
        let config = SamplerConfig {
            seed: 42,
            ..SamplerConfig::default()
        };
        let mut left = Sampler::new(config).unwrap();
        let mut right = Sampler::new(config).unwrap();
        let logits = [0.1, 0.2, 0.3, 0.4];
        let left_sequence: Vec<_> = (0..32).map(|_| left.sample(&logits).unwrap()).collect();
        let right_sequence: Vec<_> = (0..32).map(|_| right.sample(&logits).unwrap()).collect();
        assert_eq!(left_sequence, right_sequence);
    }
}
