//! Hardware-roofline accounting for model and kernel promotion evidence.

use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};

pub const ROOFLINE_FORMAT: &str = "ctox.qwen38.roofline-measurement.v1";
pub const MIN_OPTIMIZED_PRACTICAL_EFFICIENCY: f64 = 0.85;
const MAX_ACCOUNTING_OVERSHOOT: f64 = 1.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferencePhase {
    Prefill,
    Decode,
    BatchedDecode,
    SpeculativeVerify,
    KvAttention,
    RecurrentState,
    FusedKernel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrafficSource {
    HardwareCounter,
    ExactTensorSchedule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DominantCeiling {
    MemoryBandwidth,
    Compute,
    Dispatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RooflineMeasurement {
    pub format: String,
    pub hardware_profile: String,
    pub phase: InferencePhase,
    pub traffic_source: TrafficSource,
    pub elapsed_seconds: f64,
    pub accepted_tokens: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub floating_point_operations: f64,
    pub sustainable_bandwidth_bytes_per_second: f64,
    pub sustainable_compute_flops_per_second: f64,
    pub dispatch_floor_seconds: f64,
    pub theoretical_bandwidth_bytes_per_second: Option<f64>,
    pub theoretical_compute_flops_per_second: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RooflineReport {
    pub format: &'static str,
    pub hardware_profile: String,
    pub phase: InferencePhase,
    pub traffic_source: TrafficSource,
    pub accepted_tokens: u64,
    pub measured_tokens_per_second: f64,
    pub practical_roofline_tokens_per_second: f64,
    pub practical_efficiency: f64,
    pub theoretical_efficiency: Option<f64>,
    pub operational_intensity_flops_per_byte: f64,
    pub achieved_bandwidth_bytes_per_second: f64,
    pub achieved_compute_flops_per_second: f64,
    pub memory_floor_seconds: f64,
    pub compute_floor_seconds: f64,
    pub dispatch_floor_seconds: f64,
    pub practical_floor_seconds: f64,
    pub dominant_ceiling: DominantCeiling,
    pub accounting_overshoot: bool,
    pub optimized_gate_passed: bool,
    pub minimum_practical_efficiency: f64,
}

impl RooflineMeasurement {
    pub fn evaluate(&self) -> Result<RooflineReport> {
        self.validate()?;
        let total_bytes = self
            .bytes_read
            .checked_add(self.bytes_written)
            .ok_or_else(|| EngineError::InvalidArtifact("roofline byte count overflows".into()))?;
        let memory_floor_seconds = total_bytes as f64 / self.sustainable_bandwidth_bytes_per_second;
        let compute_floor_seconds =
            self.floating_point_operations / self.sustainable_compute_flops_per_second;
        let practical_floor_seconds = memory_floor_seconds
            .max(compute_floor_seconds)
            .max(self.dispatch_floor_seconds);
        let dominant_ceiling = if practical_floor_seconds == memory_floor_seconds {
            DominantCeiling::MemoryBandwidth
        } else if practical_floor_seconds == compute_floor_seconds {
            DominantCeiling::Compute
        } else {
            DominantCeiling::Dispatch
        };
        let measured_tokens_per_second = self.accepted_tokens as f64 / self.elapsed_seconds;
        let practical_roofline_tokens_per_second =
            self.accepted_tokens as f64 / practical_floor_seconds;
        let practical_efficiency = practical_floor_seconds / self.elapsed_seconds;
        let theoretical_floor = match (
            self.theoretical_bandwidth_bytes_per_second,
            self.theoretical_compute_flops_per_second,
        ) {
            (Some(bandwidth), Some(compute)) => Some(
                (total_bytes as f64 / bandwidth)
                    .max(self.floating_point_operations / compute)
                    .max(self.dispatch_floor_seconds),
            ),
            _ => None,
        };
        let accounting_overshoot = practical_efficiency > MAX_ACCOUNTING_OVERSHOOT;
        Ok(RooflineReport {
            format: "ctox.qwen38.roofline-report.v1",
            hardware_profile: self.hardware_profile.clone(),
            phase: self.phase,
            traffic_source: self.traffic_source,
            accepted_tokens: self.accepted_tokens,
            measured_tokens_per_second,
            practical_roofline_tokens_per_second,
            practical_efficiency,
            theoretical_efficiency: theoretical_floor.map(|floor| floor / self.elapsed_seconds),
            operational_intensity_flops_per_byte: self.floating_point_operations
                / total_bytes as f64,
            achieved_bandwidth_bytes_per_second: total_bytes as f64 / self.elapsed_seconds,
            achieved_compute_flops_per_second: self.floating_point_operations
                / self.elapsed_seconds,
            memory_floor_seconds,
            compute_floor_seconds,
            dispatch_floor_seconds: self.dispatch_floor_seconds,
            practical_floor_seconds,
            dominant_ceiling,
            accounting_overshoot,
            optimized_gate_passed: !accounting_overshoot
                && practical_efficiency >= MIN_OPTIMIZED_PRACTICAL_EFFICIENCY,
            minimum_practical_efficiency: MIN_OPTIMIZED_PRACTICAL_EFFICIENCY,
        })
    }

    fn validate(&self) -> Result<()> {
        if self.format != ROOFLINE_FORMAT {
            return Err(EngineError::InvalidArtifact(
                "unsupported roofline measurement format".into(),
            ));
        }
        if self.hardware_profile.trim().is_empty() {
            return Err(EngineError::InvalidArtifact(
                "roofline hardware profile is empty".into(),
            ));
        }
        if self.accepted_tokens == 0 || self.bytes_read == 0 {
            return Err(EngineError::InvalidArtifact(
                "roofline measurement requires accepted tokens and byte traffic".into(),
            ));
        }
        for (name, value, allow_zero) in [
            ("elapsed_seconds", self.elapsed_seconds, false),
            (
                "floating_point_operations",
                self.floating_point_operations,
                false,
            ),
            (
                "sustainable_bandwidth_bytes_per_second",
                self.sustainable_bandwidth_bytes_per_second,
                false,
            ),
            (
                "sustainable_compute_flops_per_second",
                self.sustainable_compute_flops_per_second,
                false,
            ),
            ("dispatch_floor_seconds", self.dispatch_floor_seconds, true),
        ] {
            if !value.is_finite() || value < 0.0 || (!allow_zero && value == 0.0) {
                return Err(EngineError::InvalidArtifact(format!(
                    "roofline {name} is not finite and positive"
                )));
            }
        }
        for (name, value) in [
            (
                "theoretical_bandwidth_bytes_per_second",
                self.theoretical_bandwidth_bytes_per_second,
            ),
            (
                "theoretical_compute_flops_per_second",
                self.theoretical_compute_flops_per_second,
            ),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value <= 0.0 {
                    return Err(EngineError::InvalidArtifact(format!(
                        "roofline {name} is not finite and positive"
                    )));
                }
            }
        }
        if self.theoretical_bandwidth_bytes_per_second.is_some()
            != self.theoretical_compute_flops_per_second.is_some()
        {
            return Err(EngineError::InvalidArtifact(
                "roofline theoretical bandwidth and compute ceilings must be paired".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measurement(elapsed_seconds: f64) -> RooflineMeasurement {
        RooflineMeasurement {
            format: ROOFLINE_FORMAT.into(),
            hardware_profile: "sm86-test".into(),
            phase: InferencePhase::Decode,
            traffic_source: TrafficSource::HardwareCounter,
            elapsed_seconds,
            accepted_tokens: 10,
            bytes_read: 900_000_000,
            bytes_written: 100_000_000,
            floating_point_operations: 100_000_000_000.0,
            sustainable_bandwidth_bytes_per_second: 100_000_000_000.0,
            sustainable_compute_flops_per_second: 100_000_000_000_000.0,
            dispatch_floor_seconds: 0.001,
            theoretical_bandwidth_bytes_per_second: Some(125_000_000_000.0),
            theoretical_compute_flops_per_second: Some(125_000_000_000_000.0),
        }
    }

    #[test]
    fn memory_bound_measurement_passes_at_eighty_five_percent() {
        let report = measurement(0.01 / 0.85).evaluate().unwrap();
        assert_eq!(report.dominant_ceiling, DominantCeiling::MemoryBandwidth);
        assert!((report.practical_efficiency - 0.85).abs() < 1e-12);
        assert!(report.optimized_gate_passed);
        assert!(!report.accounting_overshoot);
    }

    #[test]
    fn unexplained_gap_cannot_be_promoted() {
        let report = measurement(0.02).evaluate().unwrap();
        assert!((report.practical_efficiency - 0.5).abs() < 1e-12);
        assert!(!report.optimized_gate_passed);
    }

    #[test]
    fn impossible_result_exposes_incomplete_accounting() {
        let report = measurement(0.005).evaluate().unwrap();
        assert!(report.accounting_overshoot);
        assert!(!report.optimized_gate_passed);
    }

    #[test]
    fn theoretical_ceilings_must_be_a_pair() {
        let mut input = measurement(0.01);
        input.theoretical_compute_flops_per_second = None;
        assert!(input.evaluate().is_err());
    }
}
