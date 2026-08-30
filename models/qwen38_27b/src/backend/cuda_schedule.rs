//! Frozen CUDA correctness/admission schedule for Qwen3.8-27B.
//!
//! The target topology is byte-for-byte compatible with the established CTOX
//! Qwen3.5-27B target path. Production prefill/decode therefore executes the
//! direct layer-major Qwen3.5 path in `cuda_graph.rs`; it must never interpret
//! these 645 records in the hot loop. This expanded schedule is retained only
//! to prove complete resource ownership, edge order, state mutations, and the
//! sole token/chunk commit barrier before a direct submission is admitted.

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

/// Layer-major operations for one bounded prompt chunk. Matrix operations are
/// explicitly batched; order-dependent state updates remain causal device
/// scans. No operation denotes a host-side token loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaPrefillOperation {
    EmbeddingBatch,
    RmsNormBatch,
    FullAttentionFanoutBatch,
    QueryGateNormRopeBatch,
    KeyRopeBatch,
    PagedKvAppendBatch,
    PagedGqaCausalScan,
    AttentionGateOutputProjectionBatch,
    LinearFanoutBatch,
    CausalConvolutionScan,
    GatedDeltaPrepareBatch,
    GatedDeltaCausalScan,
    GatedRmsNormBatch,
    LinearOutputProjectionBatch,
    ResidualRmsNormBatch,
    FfnGateUpFanoutBatch,
    SwiGluDownProjectionBatch,
    LastTokenLmHead,
    MtpPrefillCausalScan,
    ChunkBarrier,
}

impl CudaPrefillOperation {
    pub fn is_batched_projection(self) -> bool {
        matches!(
            self,
            Self::FullAttentionFanoutBatch
                | Self::AttentionGateOutputProjectionBatch
                | Self::LinearFanoutBatch
                | Self::LinearOutputProjectionBatch
                | Self::FfnGateUpFanoutBatch
                | Self::SwiGluDownProjectionBatch
                | Self::LastTokenLmHead
        )
    }

    pub fn is_causal_device_scan(self) -> bool {
        matches!(
            self,
            Self::PagedGqaCausalScan
                | Self::CausalConvolutionScan
                | Self::GatedDeltaCausalScan
                | Self::MtpPrefillCausalScan
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPrefillStep {
    pub layer: Option<usize>,
    pub operation: CudaPrefillOperation,
    pub host_barrier_after: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPrefillChunk {
    pub start_position: usize,
    pub token_count: usize,
}

/// Exact causal alignment for the one-layer MTP program attached to a target
/// prefill chunk. MTP consumes the embedding at absolute position `i` together
/// with the final target hidden state at `i - 1`, so its cache index is always
/// one behind its RoPE position. The first prompt token has no MTP row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMtpPrefillAlignment {
    pub chunk: CudaPrefillChunk,
    pub input_token_offset: usize,
    pub rows: usize,
    pub previous_chunk_hidden_rows: usize,
    pub current_chunk_hidden_rows: usize,
    pub cache_start_token: usize,
    pub rope_start_position: usize,
    pub committed_mtp_tokens: usize,
}

impl CudaMtpPrefillAlignment {
    pub fn qwen38(
        chunk: CudaPrefillChunk,
        committed_target_tokens: usize,
        committed_mtp_tokens: usize,
    ) -> Result<Self> {
        if chunk.token_count == 0 || chunk.start_position != committed_target_tokens {
            return Err(EngineError::InvalidState(format!(
                "CUDA MTP prefill chunk {:?} does not start at committed target position {committed_target_tokens}",
                chunk
            )));
        }
        if committed_target_tokens == 0 {
            if committed_mtp_tokens != 0 {
                return Err(EngineError::InvalidState(format!(
                    "CUDA initial MTP prefill has {committed_mtp_tokens} committed rows"
                )));
            }
        } else {
            let expected_target_tokens = committed_mtp_tokens.checked_add(1).ok_or_else(|| {
                EngineError::MemoryBudget("CUDA MTP token count overflows".into())
            })?;
            if committed_target_tokens != expected_target_tokens {
                return Err(EngineError::InvalidState(format!(
                    "CUDA MTP prefill requires target exactly one token ahead, observed {committed_target_tokens}/{committed_mtp_tokens}"
                )));
            }
        }

        let first_chunk = committed_target_tokens == 0;
        let input_token_offset = usize::from(first_chunk);
        let previous_chunk_hidden_rows = usize::from(!first_chunk);
        let current_chunk_hidden_rows = chunk.token_count.saturating_sub(1);
        let rows = previous_chunk_hidden_rows
            .checked_add(current_chunk_hidden_rows)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA MTP prefill rows overflow".into()))?;
        let cache_start_token = committed_mtp_tokens;
        let rope_start_position = chunk
            .start_position
            .checked_add(input_token_offset)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA MTP RoPE position overflows".into()))?;
        let committed_mtp_tokens = committed_mtp_tokens
            .checked_add(rows)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA MTP token count overflows".into()))?;
        let committed_target_after = chunk
            .start_position
            .checked_add(chunk.token_count)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA target token count overflows".into()))?;
        let expected_target_after = committed_mtp_tokens
            .checked_add(1)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA MTP token count overflows".into()))?;
        if committed_target_after != expected_target_after {
            return Err(EngineError::InvalidState(format!(
                "CUDA MTP prefill alignment would commit target/MTP {committed_target_after}/{committed_mtp_tokens}"
            )));
        }
        Ok(Self {
            chunk,
            input_token_offset,
            rows,
            previous_chunk_hidden_rows,
            current_chunk_hidden_rows,
            cache_start_token,
            rope_start_position,
            committed_mtp_tokens,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaPrefillSchedule {
    pub steps: Vec<CudaPrefillStep>,
    pub max_chunk_tokens: usize,
    pub full_attention_layers: usize,
    pub linear_attention_layers: usize,
}

impl CudaPrefillSchedule {
    pub fn qwen38(config: &Qwen38Config, max_chunk_tokens: usize) -> Result<Self> {
        validate_frozen_geometry(config)?;
        if max_chunk_tokens == 0 || max_chunk_tokens > 65_535 {
            return Err(EngineError::Shape(
                "CUDA prefill chunk capacity must be in 1..=65535".into(),
            ));
        }
        let mut steps = Vec::with_capacity(645);
        prefill_push(
            &mut steps,
            None,
            CudaPrefillOperation::EmbeddingBatch,
            false,
        );
        prefill_push(
            &mut steps,
            Some(0),
            CudaPrefillOperation::RmsNormBatch,
            false,
        );
        for layer in 0..config.num_hidden_layers {
            match config.layer_kind(layer).expect("validated layer") {
                LayerKind::FullAttention => {
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::FullAttentionFanoutBatch,
                        false,
                    );
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::QueryGateNormRopeBatch,
                        false,
                    );
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::KeyRopeBatch,
                        false,
                    );
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::PagedKvAppendBatch,
                        false,
                    );
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::PagedGqaCausalScan,
                        false,
                    );
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::AttentionGateOutputProjectionBatch,
                        false,
                    );
                }
                LayerKind::LinearAttention => {
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::LinearFanoutBatch,
                        false,
                    );
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::CausalConvolutionScan,
                        false,
                    );
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::GatedDeltaPrepareBatch,
                        false,
                    );
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::GatedDeltaCausalScan,
                        false,
                    );
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::GatedRmsNormBatch,
                        false,
                    );
                    prefill_push(
                        &mut steps,
                        Some(layer),
                        CudaPrefillOperation::LinearOutputProjectionBatch,
                        false,
                    );
                }
            }
            for operation in [
                CudaPrefillOperation::ResidualRmsNormBatch,
                CudaPrefillOperation::FfnGateUpFanoutBatch,
                CudaPrefillOperation::SwiGluDownProjectionBatch,
                CudaPrefillOperation::ResidualRmsNormBatch,
            ] {
                prefill_push(&mut steps, Some(layer), operation, false);
            }
        }
        prefill_push(
            &mut steps,
            None,
            CudaPrefillOperation::LastTokenLmHead,
            false,
        );
        prefill_push(
            &mut steps,
            None,
            CudaPrefillOperation::MtpPrefillCausalScan,
            false,
        );
        prefill_push(&mut steps, None, CudaPrefillOperation::ChunkBarrier, true);
        let schedule = Self {
            steps,
            max_chunk_tokens,
            full_attention_layers: config.full_attention_layers(),
            linear_attention_layers: config.linear_attention_layers(),
        };
        schedule.validate()?;
        Ok(schedule)
    }

    pub fn chunks(
        &self,
        prompt_tokens: usize,
        admitted_context: usize,
    ) -> Result<Vec<CudaPrefillChunk>> {
        if prompt_tokens == 0 || prompt_tokens > admitted_context {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA prefill prompt has {prompt_tokens} tokens, admitted context is {admitted_context}"
            )));
        }
        Ok((0..prompt_tokens)
            .step_by(self.max_chunk_tokens)
            .map(|start_position| CudaPrefillChunk {
                start_position,
                token_count: self.max_chunk_tokens.min(prompt_tokens - start_position),
            })
            .collect())
    }

    pub fn validate(&self) -> Result<()> {
        if self.steps.last().is_none_or(|step| {
            step.operation != CudaPrefillOperation::ChunkBarrier || !step.host_barrier_after
        }) || self
            .steps
            .iter()
            .take(self.steps.len().saturating_sub(1))
            .any(|step| step.host_barrier_after)
        {
            return Err(EngineError::InvalidState(
                "CUDA prefill schedule requires one final chunk barrier".into(),
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
                .filter(|step| step.operation == CudaPrefillOperation::ResidualRmsNormBatch)
                .count()
                != 2
            {
                return Err(EngineError::InvalidState(format!(
                    "CUDA prefill layer {layer} does not have two residual/norm batches"
                )));
            }
            let causal_scans = layer_steps
                .iter()
                .filter(|step| step.operation.is_causal_device_scan())
                .count();
            let expected = if layer_steps
                .iter()
                .any(|step| step.operation == CudaPrefillOperation::FullAttentionFanoutBatch)
            {
                1
            } else {
                2
            };
            if causal_scans != expected {
                return Err(EngineError::InvalidState(format!(
                    "CUDA prefill layer {layer} has {causal_scans} causal scans, expected {expected}"
                )));
            }
            if layer_steps
                .iter()
                .any(|step| step.operation == CudaPrefillOperation::FullAttentionFanoutBatch)
            {
                let operations: Vec<_> = layer_steps.iter().map(|step| step.operation).collect();
                let required = [
                    CudaPrefillOperation::FullAttentionFanoutBatch,
                    CudaPrefillOperation::QueryGateNormRopeBatch,
                    CudaPrefillOperation::KeyRopeBatch,
                    CudaPrefillOperation::PagedKvAppendBatch,
                    CudaPrefillOperation::PagedGqaCausalScan,
                    CudaPrefillOperation::AttentionGateOutputProjectionBatch,
                ];
                if !operations.starts_with(&required) {
                    return Err(EngineError::InvalidState(format!(
                        "CUDA prefill full-attention layer {layer} omits or reorders RoPE/KV/causal attention"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn prefill_push(
    steps: &mut Vec<CudaPrefillStep>,
    layer: Option<usize>,
    operation: CudaPrefillOperation,
    host_barrier_after: bool,
) {
    steps.push(CudaPrefillStep {
        layer,
        operation,
        host_barrier_after,
    });
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

    #[test]
    fn prefill_schedule_is_layer_major_and_has_no_host_token_loop() {
        let schedule = CudaPrefillSchedule::qwen38(&Qwen38Config::default(), 512).unwrap();
        assert_eq!(schedule.steps.len(), 645);
        assert_eq!(schedule.full_attention_layers, 16);
        assert_eq!(schedule.linear_attention_layers, 48);
        assert_eq!(
            schedule
                .steps
                .iter()
                .filter(|step| step.operation.is_causal_device_scan())
                .count(),
            113
        );
        assert_eq!(
            schedule
                .steps
                .iter()
                .filter(|step| step.operation == CudaPrefillOperation::LastTokenLmHead)
                .count(),
            1
        );
        assert_eq!(
            schedule
                .steps
                .iter()
                .filter(|step| step.host_barrier_after)
                .count(),
            1
        );
        assert!(schedule.steps.iter().any(|step| {
            step.operation == CudaPrefillOperation::FullAttentionFanoutBatch
                && step.operation.is_batched_projection()
        }));
        assert_eq!(
            schedule
                .steps
                .iter()
                .filter(|step| step.operation == CudaPrefillOperation::PagedKvAppendBatch)
                .count(),
            16
        );
    }

    #[test]
    fn prefill_chunks_cover_prompt_once_without_padding() {
        let schedule = CudaPrefillSchedule::qwen38(&Qwen38Config::default(), 512).unwrap();
        let chunks = schedule.chunks(1_025, 131_072).unwrap();
        assert_eq!(
            chunks,
            vec![
                CudaPrefillChunk {
                    start_position: 0,
                    token_count: 512,
                },
                CudaPrefillChunk {
                    start_position: 512,
                    token_count: 512,
                },
                CudaPrefillChunk {
                    start_position: 1_024,
                    token_count: 1,
                },
            ]
        );
        assert!(schedule.chunks(0, 131_072).is_err());
        assert!(schedule.chunks(131_073, 131_072).is_err());
        assert!(CudaPrefillSchedule::qwen38(&Qwen38Config::default(), 65_536).is_err());
    }

    #[test]
    fn mtp_prefill_alignment_skips_only_the_first_prompt_token() {
        let first = CudaMtpPrefillAlignment::qwen38(
            CudaPrefillChunk {
                start_position: 0,
                token_count: 512,
            },
            0,
            0,
        )
        .unwrap();
        assert_eq!(first.input_token_offset, 1);
        assert_eq!(first.rows, 511);
        assert_eq!(first.previous_chunk_hidden_rows, 0);
        assert_eq!(first.current_chunk_hidden_rows, 511);
        assert_eq!(first.cache_start_token, 0);
        assert_eq!(first.rope_start_position, 1);
        assert_eq!(first.committed_mtp_tokens, 511);

        let second = CudaMtpPrefillAlignment::qwen38(
            CudaPrefillChunk {
                start_position: 512,
                token_count: 512,
            },
            512,
            511,
        )
        .unwrap();
        assert_eq!(second.input_token_offset, 0);
        assert_eq!(second.rows, 512);
        assert_eq!(second.previous_chunk_hidden_rows, 1);
        assert_eq!(second.current_chunk_hidden_rows, 511);
        assert_eq!(second.cache_start_token, 511);
        assert_eq!(second.rope_start_position, 512);
        assert_eq!(second.committed_mtp_tokens, 1_023);
    }

    #[test]
    fn one_token_prompt_preserves_target_one_ahead_state() {
        let alignment = CudaMtpPrefillAlignment::qwen38(
            CudaPrefillChunk {
                start_position: 0,
                token_count: 1,
            },
            0,
            0,
        )
        .unwrap();
        assert_eq!(alignment.rows, 0);
        assert_eq!(alignment.committed_mtp_tokens, 0);
    }

    #[test]
    fn mtp_prefill_alignment_rejects_divergent_counters() {
        assert!(CudaMtpPrefillAlignment::qwen38(
            CudaPrefillChunk {
                start_position: 512,
                token_count: 16,
            },
            512,
            512,
        )
        .is_err());
    }
}
