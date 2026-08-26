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
        let draw = self.next_draw();
        self.sample_with_draw(logits, draw)
    }

    pub(crate) fn config(&self) -> SamplerConfig {
        self.config
    }

    pub(crate) fn next_draw(&mut self) -> f32 {
        if self.config.temperature == 0.0 {
            0.0
        } else {
            self.rng.next_f32()
        }
    }

    /// Deterministic sampling entry point used by accelerator parity
    /// verifiers. `draw` is a canonical unit-interval RNG value and does not
    /// advance this sampler's PCG state.
    pub fn sample_with_draw(&self, logits: &[f32], draw: f32) -> Result<usize> {
        if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidArtifact(
                "sampler logits are empty or non-finite".into(),
            ));
        }
        if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
            return Err(EngineError::InvalidArtifact(
                "sampler draw must be finite and in [0, 1)".into(),
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

        let inverse_temperature = self.config.temperature.recip();
        if !inverse_temperature.is_finite() {
            return Err(EngineError::InvalidArtifact(
                "sampler inverse temperature must be finite".into(),
            ));
        }
        let mut candidates: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(token, logit)| (token, *logit * inverse_temperature))
            .collect();
        if candidates.iter().any(|(_, value)| !value.is_finite()) {
            return Err(EngineError::InvalidArtifact(
                "sampler temperature produced non-finite scaled logits".into(),
            ));
        }
        candidates.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
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
        if !normalization.is_finite() || normalization <= 0.0 {
            return Err(EngineError::InvalidArtifact(
                "sampler probability normalization is invalid".into(),
            ));
        }

        let mut cumulative = 0.0_f32;
        let mut nucleus_len = candidates.len();
        for (index, (_, probability)) in candidates.iter().enumerate() {
            cumulative += *probability;
            if cumulative >= self.config.top_p * normalization {
                nucleus_len = index + 1;
                break;
            }
        }
        candidates.truncate(nucleus_len);
        let nucleus_total: f32 = candidates.iter().map(|(_, probability)| probability).sum();
        let threshold = draw * nucleus_total;
        let mut cumulative = 0.0_f32;
        for (token, probability) in &candidates {
            cumulative += *probability;
            if threshold <= cumulative {
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

    #[test]
    fn explicit_draw_is_replayable_without_advancing_rng() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.8,
            top_k: 4,
            top_p: 0.9,
            seed: 17,
        })
        .unwrap();
        let logits = [-0.5, 0.25, 1.5, 0.75];
        let left = sampler.sample_with_draw(&logits, 0.625).unwrap();
        let right = sampler.sample_with_draw(&logits, 0.625).unwrap();
        assert_eq!(left, right);

        let first_seeded = sampler.sample(&logits).unwrap();
        let mut fresh = Sampler::new(SamplerConfig {
            temperature: 0.8,
            top_k: 4,
            top_p: 0.9,
            seed: 17,
        })
        .unwrap();
        assert_eq!(first_seeded, fresh.sample(&logits).unwrap());
    }

    #[test]
    fn tied_candidates_use_the_later_token_like_greedy_argmax() {
        let sampler = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: 0,
        })
        .unwrap();
        assert_eq!(sampler.sample_with_draw(&[2.0, 2.0], 0.0).unwrap(), 1);
    }

    #[test]
    fn explicit_draw_and_scaled_logits_fail_closed() {
        let sampler = Sampler::new(SamplerConfig {
            temperature: f32::from_bits(1),
            top_k: 2,
            top_p: 1.0,
            seed: 0,
        })
        .unwrap();
        assert!(sampler.sample_with_draw(&[1.0, 0.0], 0.5).is_err());
        assert!(sampler.sample_with_draw(&[1.0], 1.0).is_err());
    }
}
