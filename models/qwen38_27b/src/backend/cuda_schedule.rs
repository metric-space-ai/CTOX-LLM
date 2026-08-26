//! Frozen CUDA decode schedule for Qwen3.8-27B.
//!
//! This is the model-specific assembly contract between prepared CUDA
//! operators and the future production `ModelExecutor`. It deliberately
//! contains no generic graph optimizer: every edge, state mutation, fused
//! residual/norm binding, and token barrier is explicit.

use std::collections::HashSet;

use crate::config::LayerKind;
use crate::{EngineError, Qwen38Config, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaBufferSlot {
    HiddenA,
    HiddenB,
    Normalized,
    Qkv,
    QueryGate,
    Query,
    Key,
    Value,
    AttentionGate,
    MixerOutput,
    LinearZ,
    LinearA,
    LinearB,
    LogDecay,
    Beta,
    FfnGate,
    FfnUp,
    FfnDown,
    TargetLogits,
    MtpDraft,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaNormBinding {
    LayerInput(usize),
    LayerPostAttention(usize),
    Final,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaDecodeOperation {
    Embedding,
    RmsNorm,
    FullAttentionFanout,
    QueryGateNormRope,
    KeyRope,
    PagedKvAppend,
    PagedGqa,
    AttentionGateA8OutputProjection,
    LinearFanout,
    CausalConvolution,
    GatedDeltaPrepare,
    GatedDeltaRecurrent,
    GatedRmsNorm,
    LinearOutputProjection,
    ResidualRmsNorm,
    FfnGateUpFanout,
    SwiGluA8DownProjection,
    LmHead,
    MtpDraftAndTargetVerify,
    TokenBarrier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDecodeStep {
    pub layer: Option<usize>,
    pub operation: CudaDecodeOperation,
    pub reads: Vec<CudaBufferSlot>,
    pub writes: Vec<CudaBufferSlot>,
    pub norm: Option<CudaNormBinding>,
    pub mutates_session_state: bool,
    pub host_barrier_after: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDecodeSchedule {
    pub steps: Vec<CudaDecodeStep>,
    pub full_attention_layers: usize,
    pub linear_attention_layers: usize,
    pub token_barriers: usize,
}

impl CudaDecodeSchedule {
    pub fn qwen38(config: &Qwen38Config) -> Result<Self> {
        validate_frozen_geometry(config)?;
        let mut steps = Vec::with_capacity(config.num_hidden_layers * 11 + 5);
        push(
            &mut steps,
            None,
            CudaDecodeOperation::Embedding,
            &[],
            &[CudaBufferSlot::HiddenA],
            None,
            false,
            false,
        );
        push(
            &mut steps,
            Some(0),
            CudaDecodeOperation::RmsNorm,
            &[CudaBufferSlot::HiddenA],
            &[CudaBufferSlot::Normalized],
            Some(CudaNormBinding::LayerInput(0)),
            false,
            false,
        );

        for layer in 0..config.num_hidden_layers {
            match config.layer_kind(layer).expect("validated layer") {
                LayerKind::FullAttention => full_attention(&mut steps, layer),
                LayerKind::LinearAttention => linear_attention(&mut steps, layer),
            }
            push(
                &mut steps,
                Some(layer),
                CudaDecodeOperation::ResidualRmsNorm,
                &[CudaBufferSlot::HiddenA, CudaBufferSlot::MixerOutput],
                &[CudaBufferSlot::HiddenB, CudaBufferSlot::Normalized],
                Some(CudaNormBinding::LayerPostAttention(layer)),
                false,
                false,
            );
            push(
                &mut steps,
                Some(layer),
                CudaDecodeOperation::FfnGateUpFanout,
                &[CudaBufferSlot::Normalized],
                &[CudaBufferSlot::FfnGate, CudaBufferSlot::FfnUp],
                None,
                false,
                false,
            );
            push(
                &mut steps,
                Some(layer),
                CudaDecodeOperation::SwiGluA8DownProjection,
                &[CudaBufferSlot::FfnGate, CudaBufferSlot::FfnUp],
                &[CudaBufferSlot::FfnDown],
                None,
                false,
                false,
            );
            let binding = if layer + 1 == config.num_hidden_layers {
                CudaNormBinding::Final
            } else {
                CudaNormBinding::LayerInput(layer + 1)
            };
            push(
                &mut steps,
                Some(layer),
                CudaDecodeOperation::ResidualRmsNorm,
                &[CudaBufferSlot::HiddenB, CudaBufferSlot::FfnDown],
                &[CudaBufferSlot::HiddenA, CudaBufferSlot::Normalized],
                Some(binding),
                false,
                false,
            );
        }
        push(
            &mut steps,
            None,
            CudaDecodeOperation::LmHead,
            &[CudaBufferSlot::Normalized],
            &[CudaBufferSlot::TargetLogits],
            None,
            false,
            false,
        );
        push(
            &mut steps,
            None,
            CudaDecodeOperation::MtpDraftAndTargetVerify,
            &[CudaBufferSlot::Normalized, CudaBufferSlot::TargetLogits],
            &[CudaBufferSlot::MtpDraft],
            None,
            true,
            false,
        );
        push(
            &mut steps,
            None,
            CudaDecodeOperation::TokenBarrier,
            &[CudaBufferSlot::TargetLogits, CudaBufferSlot::MtpDraft],
            &[],
            None,
            false,
            true,
        );
        let token_barriers = steps.iter().filter(|step| step.host_barrier_after).count();
        Ok(Self {
            steps,
            full_attention_layers: config.full_attention_layers(),
            linear_attention_layers: config.linear_attention_layers(),
            token_barriers,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.token_barriers != 1
            || self.steps.last().is_none_or(|step| {
                step.operation != CudaDecodeOperation::TokenBarrier || !step.host_barrier_after
            })
            || self
                .steps
                .iter()
                .take(self.steps.len().saturating_sub(1))
                .any(|step| step.host_barrier_after)
        {
            return Err(EngineError::InvalidState(
                "CUDA decode schedule must have exactly one final token barrier".into(),
            ));
        }
        let mut available = HashSet::new();
        for (index, step) in self.steps.iter().enumerate() {
            for slot in &step.reads {
                if !available.contains(slot) {
                    return Err(EngineError::InvalidState(format!(
                        "CUDA decode step {index} reads unavailable slot {slot:?}"
                    )));
                }
            }
            available.extend(step.writes.iter().copied());
        }
        if !available.contains(&CudaBufferSlot::TargetLogits)
            || !available.contains(&CudaBufferSlot::MtpDraft)
        {
            return Err(EngineError::InvalidState(
                "CUDA decode schedule does not produce target and MTP outputs".into(),
            ));
        }
        for layer in 0..self.full_attention_layers + self.linear_attention_layers {
            let layer_steps: Vec<_> = self
                .steps
                .iter()
                .filter(|step| step.layer == Some(layer))
                .collect();
            let residual_norms = layer_steps
                .iter()
                .filter(|step| step.operation == CudaDecodeOperation::ResidualRmsNorm)
                .count();
            if residual_norms != 2 {
                return Err(EngineError::InvalidState(format!(
                    "CUDA layer {layer} has {residual_norms} residual norms, expected two"
                )));
            }
        }
        Ok(())
    }
}

fn full_attention(steps: &mut Vec<CudaDecodeStep>, layer: usize) {
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::FullAttentionFanout,
        &[CudaBufferSlot::Normalized],
        &[
            CudaBufferSlot::QueryGate,
            CudaBufferSlot::Key,
            CudaBufferSlot::Value,
        ],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::QueryGateNormRope,
        &[CudaBufferSlot::QueryGate],
        &[CudaBufferSlot::Query, CudaBufferSlot::AttentionGate],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::KeyRope,
        &[CudaBufferSlot::Key],
        &[CudaBufferSlot::Key],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::PagedKvAppend,
        &[CudaBufferSlot::Key, CudaBufferSlot::Value],
        &[],
        None,
        true,
        false,
    );
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::PagedGqa,
        &[CudaBufferSlot::Query],
        &[CudaBufferSlot::MixerOutput],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::AttentionGateA8OutputProjection,
        &[CudaBufferSlot::MixerOutput, CudaBufferSlot::AttentionGate],
        &[CudaBufferSlot::MixerOutput],
        None,
        false,
        false,
    );
}

fn linear_attention(steps: &mut Vec<CudaDecodeStep>, layer: usize) {
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::LinearFanout,
        &[CudaBufferSlot::Normalized],
        &[
            CudaBufferSlot::Qkv,
            CudaBufferSlot::LinearZ,
            CudaBufferSlot::LinearA,
            CudaBufferSlot::LinearB,
        ],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::CausalConvolution,
        &[CudaBufferSlot::Qkv],
        &[CudaBufferSlot::Qkv],
        None,
        true,
        false,
    );
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::GatedDeltaPrepare,
        &[
            CudaBufferSlot::Qkv,
            CudaBufferSlot::LinearA,
            CudaBufferSlot::LinearB,
        ],
        &[
            CudaBufferSlot::Query,
            CudaBufferSlot::Key,
            CudaBufferSlot::LogDecay,
            CudaBufferSlot::Beta,
        ],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::GatedDeltaRecurrent,
        &[
            CudaBufferSlot::Query,
            CudaBufferSlot::Key,
            CudaBufferSlot::Qkv,
            CudaBufferSlot::LogDecay,
            CudaBufferSlot::Beta,
        ],
        &[CudaBufferSlot::MixerOutput],
        None,
        true,
        false,
    );
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::GatedRmsNorm,
        &[CudaBufferSlot::MixerOutput, CudaBufferSlot::LinearZ],
        &[CudaBufferSlot::MixerOutput],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        CudaDecodeOperation::LinearOutputProjection,
        &[CudaBufferSlot::MixerOutput],
        &[CudaBufferSlot::MixerOutput],
        None,
        false,
        false,
    );
}

#[allow(clippy::too_many_arguments)]
fn push(
    steps: &mut Vec<CudaDecodeStep>,
    layer: Option<usize>,
    operation: CudaDecodeOperation,
    reads: &[CudaBufferSlot],
    writes: &[CudaBufferSlot],
    norm: Option<CudaNormBinding>,
    mutates_session_state: bool,
    host_barrier_after: bool,
) {
    steps.push(CudaDecodeStep {
        layer,
        operation,
        reads: reads.to_vec(),
        writes: writes.to_vec(),
        norm,
        mutates_session_state,
        host_barrier_after,
    });
}

fn validate_frozen_geometry(config: &Qwen38Config) -> Result<()> {
    let expected = Qwen38Config::default();
    if config != &expected {
        return Err(EngineError::Shape(
            "CUDA production schedule currently accepts only the frozen Qwen3.8-27B topology"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_schedule_covers_every_layer_and_has_one_token_barrier() {
        let schedule = CudaDecodeSchedule::qwen38(&Qwen38Config::default()).unwrap();
        schedule.validate().unwrap();
        assert_eq!(schedule.full_attention_layers, 16);
        assert_eq!(schedule.linear_attention_layers, 48);
        assert_eq!(schedule.token_barriers, 1);
        assert_eq!(schedule.steps.len(), 645);
        assert_eq!(
            schedule
                .steps
                .iter()
                .filter(|step| step.operation == CudaDecodeOperation::ResidualRmsNorm)
                .count(),
            128
        );
        assert_eq!(
            schedule
                .steps
                .iter()
                .filter(|step| step.operation == CudaDecodeOperation::GatedDeltaRecurrent)
                .count(),
            48
        );
        assert_eq!(
            schedule
                .steps
                .iter()
                .filter(|step| step.operation == CudaDecodeOperation::PagedGqa)
                .count(),
            16
        );
    }

    #[test]
    fn every_layer_uses_cross_layer_norm_fusion() {
        let schedule = CudaDecodeSchedule::qwen38(&Qwen38Config::default()).unwrap();
        for layer in 0..63 {
            assert!(schedule.steps.iter().any(|step| {
                step.layer == Some(layer)
                    && step.norm == Some(CudaNormBinding::LayerInput(layer + 1))
            }));
        }
        assert!(schedule
            .steps
            .iter()
            .any(|step| { step.layer == Some(63) && step.norm == Some(CudaNormBinding::Final) }));
    }

    #[test]
    fn mtp_consumes_the_final_normalized_target_hidden_state() {
        let schedule = CudaDecodeSchedule::qwen38(&Qwen38Config::default()).unwrap();
        let mtp = schedule
            .steps
            .iter()
            .find(|step| step.operation == CudaDecodeOperation::MtpDraftAndTargetVerify)
            .unwrap();
        assert!(mtp.reads.contains(&CudaBufferSlot::Normalized));
        assert!(!mtp.reads.contains(&CudaBufferSlot::HiddenA));
    }

    #[test]
    fn non_frozen_topology_is_rejected() {
        let config = Qwen38Config {
            num_hidden_layers: 63,
            ..Qwen38Config::default()
        };
        assert!(CudaDecodeSchedule::qwen38(&config).is_err());
    }
}
