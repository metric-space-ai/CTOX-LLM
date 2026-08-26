//! Frozen Metal decode schedule for Qwen3.8-27B.
//!
//! This is the model-specific ordering contract for the future complete Metal
//! executor. It deliberately names every state mutation and the sole host wait;
//! isolated kernels cannot be promoted by bypassing this graph.

use std::collections::HashSet;

use crate::config::LayerKind;
use crate::{EngineError, Qwen38Config, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetalBufferSlot {
    HiddenA,
    HiddenB,
    Normalized,
    QueryGate,
    Query,
    Key,
    Value,
    AttentionGate,
    AttentionOutput,
    MixerOutput,
    LinearQkv,
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
pub enum MetalNormBinding {
    LayerInput(usize),
    LayerPostAttention(usize),
    Final,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalDecodeOperation {
    Embedding,
    RmsNorm,
    FullAttentionFanout,
    QueryGateNormRope,
    KeyRope,
    PagedKvAppend,
    PagedGqa,
    AttentionGateOutputProjection,
    LinearFanout,
    CausalConvolution,
    GatedDeltaPrepare,
    GatedDeltaRecurrent,
    GatedRmsNorm,
    LinearOutputProjection,
    ResidualRmsNorm,
    FfnGateUpFanout,
    SwiGluDownProjection,
    LmHead,
    MtpDraftAndTargetVerify,
    TokenCommandBufferCommit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDecodeStep {
    pub layer: Option<usize>,
    pub operation: MetalDecodeOperation,
    pub reads: Vec<MetalBufferSlot>,
    pub writes: Vec<MetalBufferSlot>,
    pub norm: Option<MetalNormBinding>,
    pub mutates_session_state: bool,
    pub host_wait_after: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDecodeSchedule {
    pub steps: Vec<MetalDecodeStep>,
    pub full_attention_layers: usize,
    pub linear_attention_layers: usize,
    pub command_buffer_commits: usize,
    pub host_waits: usize,
}

impl MetalDecodeSchedule {
    pub fn qwen38(config: &Qwen38Config) -> Result<Self> {
        if config != &Qwen38Config::default() {
            return Err(EngineError::Shape(
                "Metal decode schedule requires the frozen Qwen3.8-27B topology".into(),
            ));
        }
        let mut steps = Vec::with_capacity(645);
        push(
            &mut steps,
            None,
            MetalDecodeOperation::Embedding,
            &[],
            &[MetalBufferSlot::HiddenA],
            None,
            false,
            false,
        );
        push(
            &mut steps,
            Some(0),
            MetalDecodeOperation::RmsNorm,
            &[MetalBufferSlot::HiddenA],
            &[MetalBufferSlot::Normalized],
            Some(MetalNormBinding::LayerInput(0)),
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
                MetalDecodeOperation::ResidualRmsNorm,
                &[MetalBufferSlot::HiddenA, MetalBufferSlot::MixerOutput],
                &[MetalBufferSlot::HiddenB, MetalBufferSlot::Normalized],
                Some(MetalNormBinding::LayerPostAttention(layer)),
                false,
                false,
            );
            push(
                &mut steps,
                Some(layer),
                MetalDecodeOperation::FfnGateUpFanout,
                &[MetalBufferSlot::Normalized],
                &[MetalBufferSlot::FfnGate, MetalBufferSlot::FfnUp],
                None,
                false,
                false,
            );
            push(
                &mut steps,
                Some(layer),
                MetalDecodeOperation::SwiGluDownProjection,
                &[MetalBufferSlot::FfnGate, MetalBufferSlot::FfnUp],
                &[MetalBufferSlot::FfnDown],
                None,
                false,
                false,
            );
            let norm = if layer + 1 == config.num_hidden_layers {
                MetalNormBinding::Final
            } else {
                MetalNormBinding::LayerInput(layer + 1)
            };
            push(
                &mut steps,
                Some(layer),
                MetalDecodeOperation::ResidualRmsNorm,
                &[MetalBufferSlot::HiddenB, MetalBufferSlot::FfnDown],
                &[MetalBufferSlot::HiddenA, MetalBufferSlot::Normalized],
                Some(norm),
                false,
                false,
            );
        }
        push(
            &mut steps,
            None,
            MetalDecodeOperation::LmHead,
            &[MetalBufferSlot::Normalized],
            &[MetalBufferSlot::TargetLogits],
            None,
            false,
            false,
        );
        push(
            &mut steps,
            None,
            MetalDecodeOperation::MtpDraftAndTargetVerify,
            &[MetalBufferSlot::Normalized, MetalBufferSlot::TargetLogits],
            &[MetalBufferSlot::MtpDraft],
            None,
            true,
            false,
        );
        push(
            &mut steps,
            None,
            MetalDecodeOperation::TokenCommandBufferCommit,
            &[MetalBufferSlot::TargetLogits, MetalBufferSlot::MtpDraft],
            &[],
            None,
            false,
            true,
        );
        let schedule = Self {
            command_buffer_commits: steps
                .iter()
                .filter(|step| step.operation == MetalDecodeOperation::TokenCommandBufferCommit)
                .count(),
            host_waits: steps.iter().filter(|step| step.host_wait_after).count(),
            steps,
            full_attention_layers: config.full_attention_layers(),
            linear_attention_layers: config.linear_attention_layers(),
        };
        schedule.validate()?;
        Ok(schedule)
    }

    pub fn validate(&self) -> Result<()> {
        if self.steps.len() != 645
            || self.command_buffer_commits != 1
            || self.host_waits != 1
            || self.steps.last().is_none_or(|step| {
                step.operation != MetalDecodeOperation::TokenCommandBufferCommit
                    || !step.host_wait_after
            })
            || self
                .steps
                .iter()
                .take(self.steps.len().saturating_sub(1))
                .any(|step| step.host_wait_after)
        {
            return Err(EngineError::InvalidState(
                "Metal decode requires exactly 645 ordered steps and one final command-buffer wait"
                    .into(),
            ));
        }
        if self.full_attention_layers != 16 || self.linear_attention_layers != 48 {
            return Err(EngineError::InvalidState(
                "Metal decode layer counts differ from the frozen topology".into(),
            ));
        }

        let mut available = HashSet::new();
        for (index, step) in self.steps.iter().enumerate() {
            for slot in &step.reads {
                if !available.contains(slot) {
                    return Err(EngineError::InvalidState(format!(
                        "Metal decode step {index} reads unavailable slot {slot:?}"
                    )));
                }
            }
            available.extend(step.writes.iter().copied());
        }
        if !available.contains(&MetalBufferSlot::TargetLogits)
            || !available.contains(&MetalBufferSlot::MtpDraft)
        {
            return Err(EngineError::InvalidState(
                "Metal decode schedule does not produce target and MTP outputs".into(),
            ));
        }
        for layer in 0..self.full_attention_layers + self.linear_attention_layers {
            let layer_steps: Vec<_> = self
                .steps
                .iter()
                .filter(|step| step.layer == Some(layer))
                .collect();
            if layer_steps
                .iter()
                .filter(|step| step.operation == MetalDecodeOperation::ResidualRmsNorm)
                .count()
                != 2
            {
                return Err(EngineError::InvalidState(format!(
                    "Metal decode layer {layer} does not bind two residual norms"
                )));
            }
            let state_mutations = layer_steps
                .iter()
                .filter(|step| step.mutates_session_state)
                .count();
            let expected_state_mutations = if layer_steps
                .iter()
                .any(|step| step.operation == MetalDecodeOperation::PagedKvAppend)
            {
                1
            } else {
                2
            };
            if state_mutations != expected_state_mutations {
                return Err(EngineError::InvalidState(format!(
                    "Metal decode layer {layer} has {state_mutations} state mutations, expected {expected_state_mutations}"
                )));
            }
        }
        Ok(())
    }
}

fn full_attention(steps: &mut Vec<MetalDecodeStep>, layer: usize) {
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::FullAttentionFanout,
        &[MetalBufferSlot::Normalized],
        &[
            MetalBufferSlot::QueryGate,
            MetalBufferSlot::Key,
            MetalBufferSlot::Value,
        ],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::QueryGateNormRope,
        &[MetalBufferSlot::QueryGate],
        &[MetalBufferSlot::Query, MetalBufferSlot::AttentionGate],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::KeyRope,
        &[MetalBufferSlot::Key],
        &[MetalBufferSlot::Key],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::PagedKvAppend,
        &[MetalBufferSlot::Key, MetalBufferSlot::Value],
        &[],
        None,
        true,
        false,
    );
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::PagedGqa,
        &[MetalBufferSlot::Query],
        &[MetalBufferSlot::AttentionOutput],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::AttentionGateOutputProjection,
        &[
            MetalBufferSlot::AttentionOutput,
            MetalBufferSlot::AttentionGate,
        ],
        &[MetalBufferSlot::MixerOutput],
        None,
        false,
        false,
    );
}

fn linear_attention(steps: &mut Vec<MetalDecodeStep>, layer: usize) {
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::LinearFanout,
        &[MetalBufferSlot::Normalized],
        &[
            MetalBufferSlot::LinearQkv,
            MetalBufferSlot::LinearZ,
            MetalBufferSlot::LinearA,
            MetalBufferSlot::LinearB,
        ],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::CausalConvolution,
        &[MetalBufferSlot::LinearQkv],
        &[MetalBufferSlot::LinearQkv],
        None,
        true,
        false,
    );
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::GatedDeltaPrepare,
        &[
            MetalBufferSlot::LinearQkv,
            MetalBufferSlot::LinearA,
            MetalBufferSlot::LinearB,
        ],
        &[
            MetalBufferSlot::Query,
            MetalBufferSlot::Key,
            MetalBufferSlot::Value,
            MetalBufferSlot::LogDecay,
            MetalBufferSlot::Beta,
        ],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::GatedDeltaRecurrent,
        &[
            MetalBufferSlot::Query,
            MetalBufferSlot::Key,
            MetalBufferSlot::Value,
            MetalBufferSlot::LogDecay,
            MetalBufferSlot::Beta,
        ],
        &[MetalBufferSlot::AttentionOutput],
        None,
        true,
        false,
    );
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::GatedRmsNorm,
        &[MetalBufferSlot::AttentionOutput, MetalBufferSlot::LinearZ],
        &[MetalBufferSlot::AttentionOutput],
        None,
        false,
        false,
    );
    push(
        steps,
        Some(layer),
        MetalDecodeOperation::LinearOutputProjection,
        &[MetalBufferSlot::AttentionOutput],
        &[MetalBufferSlot::MixerOutput],
        None,
        false,
        false,
    );
}

#[allow(clippy::too_many_arguments)]
fn push(
    steps: &mut Vec<MetalDecodeStep>,
    layer: Option<usize>,
    operation: MetalDecodeOperation,
    reads: &[MetalBufferSlot],
    writes: &[MetalBufferSlot],
    norm: Option<MetalNormBinding>,
    mutates_session_state: bool,
    host_wait_after: bool,
) {
    steps.push(MetalDecodeStep {
        layer,
        operation,
        reads: reads.to_vec(),
        writes: writes.to_vec(),
        norm,
        mutates_session_state,
        host_wait_after,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_metal_decode_is_one_645_step_command_buffer() {
        let schedule = MetalDecodeSchedule::qwen38(&Qwen38Config::default()).unwrap();
        assert_eq!(schedule.steps.len(), 645);
        assert_eq!(schedule.full_attention_layers, 16);
        assert_eq!(schedule.linear_attention_layers, 48);
        assert_eq!(schedule.command_buffer_commits, 1);
        assert_eq!(schedule.host_waits, 1);
        assert_eq!(
            schedule.steps.last().unwrap().operation,
            MetalDecodeOperation::TokenCommandBufferCommit
        );
    }

    #[test]
    fn metal_layers_bind_every_causal_state_mutation() {
        let schedule = MetalDecodeSchedule::qwen38(&Qwen38Config::default()).unwrap();
        for layer in 0..64 {
            let expected =
                if Qwen38Config::default().layer_kind(layer).unwrap() == LayerKind::FullAttention {
                    1
                } else {
                    2
                };
            assert_eq!(
                schedule
                    .steps
                    .iter()
                    .filter(|step| step.layer == Some(layer) && step.mutates_session_state)
                    .count(),
                expected
            );
        }
    }

    #[test]
    fn altered_topology_cannot_reuse_the_metal_schedule() {
        let mut config = Qwen38Config::default();
        config.num_hidden_layers -= 1;
        assert!(MetalDecodeSchedule::qwen38(&config).is_err());
    }
}
