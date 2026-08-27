//! Embeddable Apple-Silicon executor over the model-specific Metal graph.
//!
//! The executor owns one CTOXQ mmap, one shared decode arena, target state,
//! MTP state, and the Metal runtime. It never materializes a second weight
//! copy or falls back to CPU model operators. Until chunked Metal prefill and
//! stochastic resident sampling are promoted, prefill is deliberately serial
//! and target selection is greedy-only.

use crate::backend::metal_graph::{
    MetalDecodeBindingPlan, MetalDecodeWorkspacePlan, MetalMtpWorkspacePlan, MetalProjectionPlan,
    METAL_MTP4_RECORDS,
};
use crate::backend::metal_runtime::{
    MappedMetalArtifact, MetalCandidateRuntime, MetalGreedyMtpVerification, MetalPagedGqaConfig,
    PreparedMappedMetalMtpCore, PreparedMappedMetalTargetCore, PreparedMetalDecodeWorkspace,
};
use crate::backend::metal_schedule::MetalDecodeSchedule;
use crate::backend::{BackendKind, PromotionState};
use crate::engine::{
    AllocationSnapshot, CancellationToken, ExecutorCapabilities, ExecutorStep,
    GreedyMtpVerification, ModelExecutor,
};
use crate::loader::ModelArtifact;
use crate::memory::{LinearStateDType, SpeculativeStateStrategy};
use crate::release::{KvMemoryFormula, MemoryProfile};
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::TOKENIZER_VOCAB_SIZE;
use crate::{EngineError, Qwen38Config, Result};

pub const METAL_EXECUTOR_PROFILE: &str = "metal-apple-silicon-qwen38-verifier";
const METAL_KV_PAGE_TOKENS: usize = 64;
const METAL_KV_SINK_TOKENS: usize = 128;

#[derive(Debug, Clone, Copy)]
struct PendingMetalSpeculativeAck {
    accepted_drafts: u32,
}

pub struct MetalModelExecutor {
    hardware_profile: String,
    config: Qwen38Config,
    runtime: Option<MetalCandidateRuntime>,
    mapping: Option<MappedMetalArtifact>,
    binding_plan: Option<MetalDecodeBindingPlan>,
    workspace: Option<PreparedMetalDecodeWorkspace>,
    target: Option<PreparedMappedMetalTargetCore>,
    mtp: Option<PreparedMappedMetalMtpCore>,
    admitted_context: usize,
    admitted_draft_tokens: usize,
    pending_ack: Option<PendingMetalSpeculativeAck>,
    last_target_token: Option<u32>,
    warmed: bool,
    allocations: AllocationSnapshot,
}

impl MetalModelExecutor {
    pub fn new() -> Result<Self> {
        Self::new_for_profile(METAL_EXECUTOR_PROFILE)
    }

    /// Compile an executor bound to the exact hardware profile carried by a
    /// signed release pack. The engine rechecks this string before opening the
    /// artifact, so a pack cannot silently run through a differently tuned
    /// Metal profile.
    pub fn new_for_profile(hardware_profile: impl Into<String>) -> Result<Self> {
        let hardware_profile = hardware_profile.into();
        if hardware_profile.trim().is_empty() {
            return Err(EngineError::InvalidArtifact(
                "Metal executor hardware profile must not be empty".into(),
            ));
        }
        Ok(Self {
            hardware_profile,
            config: Qwen38Config::default(),
            runtime: Some(MetalCandidateRuntime::new()?),
            mapping: None,
            binding_plan: None,
            workspace: None,
            target: None,
            mtp: None,
            admitted_context: 0,
            admitted_draft_tokens: 0,
            pending_ack: None,
            last_target_token: None,
            warmed: false,
            allocations: AllocationSnapshot::default(),
        })
    }

    fn is_loaded(&self) -> bool {
        self.mapping.is_some()
            && self.binding_plan.is_some()
            && self.workspace.is_some()
            && self.target.is_some()
            && self.mtp.is_some()
    }

    fn validate_ready(&self, cancellation: &CancellationToken) -> Result<()> {
        if cancellation.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        if !self.warmed || !self.is_loaded() || self.runtime.is_none() {
            return Err(EngineError::InvalidState(
                "Metal executor is not warm and completely loaded".into(),
            ));
        }
        if self.pending_ack.is_some() {
            return Err(EngineError::InvalidState(
                "Metal executor still awaits speculative acknowledgement".into(),
            ));
        }
        Ok(())
    }

    fn canonical_draft_vocabulary(&self, token_ids: &[u32]) -> bool {
        !token_ids.is_empty()
            && token_ids.len() <= TOKENIZER_VOCAB_SIZE
            && token_ids
                .iter()
                .all(|token| (*token as usize) < TOKENIZER_VOCAB_SIZE)
            && token_ids.windows(2).all(|pair| pair[0] < pair[1])
    }

    fn cache_config(
        &self,
        context_tokens: usize,
        formula: &KvMemoryFormula,
    ) -> Result<MetalPagedGqaConfig> {
        let retained = usize::try_from(
            formula
                .retained_q4_tokens_per_session
                .min(context_tokens as u64),
        )
        .map_err(|_| EngineError::MemoryBudget("Metal retained Q4 window exceeds usize".into()))?;
        let sink_tokens = retained.min(METAL_KV_SINK_TOKENS);
        let recent_tokens = retained.saturating_sub(sink_tokens);
        Ok(MetalPagedGqaConfig {
            query_heads: self.config.num_attention_heads,
            key_value_heads: self.config.num_key_value_heads,
            head_dim: self.config.head_dim,
            maximum_tokens: context_tokens,
            page_tokens: METAL_KV_PAGE_TOKENS.min(context_tokens),
            sink_tokens,
            recent_tokens,
        })
    }

    fn reset_loaded_state(&mut self) -> Result<()> {
        self.pending_ack = None;
        self.last_target_token = None;
        if let Some(target) = self.target.as_mut() {
            target.reset_session()?;
        }
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.reset_session()?;
        }
        if let Some(workspace) = self.workspace.as_mut() {
            workspace.reset();
        }
        Ok(())
    }

    fn compact_step(records: &[MetalGreedyMtpVerification], bonus_token: u32) -> ExecutorStep {
        ExecutorStep {
            target_logits: Vec::new(),
            draft_logits: Vec::new(),
            target_verification_logits: Vec::new(),
            bonus_logits: None,
            compact_greedy_mtp: Some(GreedyMtpVerification {
                draft_tokens: records.iter().map(|record| record.draft_token).collect(),
                target_tokens: records.iter().map(|record| record.target_token).collect(),
                bonus_token,
            }),
        }
    }

    fn resident_target_step() -> ExecutorStep {
        ExecutorStep {
            target_logits: Vec::new(),
            draft_logits: Vec::new(),
            target_verification_logits: Vec::new(),
            bonus_logits: None,
            compact_greedy_mtp: None,
        }
    }

    fn accepted_prefix(records: &[MetalGreedyMtpVerification]) -> usize {
        records.iter().take_while(|record| record.accepted).count()
    }
}

impl ModelExecutor for MetalModelExecutor {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn hardware_profile(&self) -> &str {
        &self.hardware_profile
    }

    fn promotion_state(&self) -> PromotionState {
        PromotionState::Verifier
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            vocab_size: TOKENIZER_VOCAB_SIZE,
            maximum_context_tokens: self.config.max_position_embeddings as u64,
            mtp: self.config.mtp_num_hidden_layers == 1,
            maximum_draft_tokens: METAL_MTP4_RECORDS as u32,
            compact_greedy_mtp_verification: true,
            resident_target_selection: true,
            cancellation: true,
            session_reset: true,
            no_hidden_fallbacks: true,
        }
    }

    fn load(
        &mut self,
        artifact: &ModelArtifact,
        profile: &MemoryProfile,
        mtp_draft_token_ids: &[u32],
    ) -> Result<()> {
        if self.is_loaded() || self.warmed || !self.allocations.is_zero() {
            return Err(EngineError::InvalidState(
                "Metal executor is already loaded".into(),
            ));
        }
        if profile.sessions != 1
            || profile.linear_state_dtype != LinearStateDType::F16
            || profile.mtp_draft_tokens == 0
            || profile.mtp_draft_tokens as usize > METAL_MTP4_RECORDS
            || profile.speculative_state_strategy != SpeculativeStateStrategy::ReplayOnReject
            || profile.speculative_linear_state_bytes_per_session
                != profile.linear_state_bytes_per_session
        {
            return Err(EngineError::UnsupportedOperation {
                backend: "metal",
                operation: "executor memory profile",
                reason: "Metal requires one session, FP16 linear state, and one replay-on-reject checkpoint"
                    .into(),
            });
        }
        let admitted_context = usize::try_from(profile.context_tokens)
            .map_err(|_| EngineError::MemoryBudget("Metal context exceeds usize".into()))?;
        if admitted_context == 0 || admitted_context > self.config.max_position_embeddings {
            return Err(EngineError::MemoryBudget(
                "Metal memory profile context exceeds model capacity".into(),
            ));
        }
        if !self.canonical_draft_vocabulary(mtp_draft_token_ids) {
            return Err(EngineError::InvalidArtifact(
                "Metal executor received a noncanonical MTP draft vocabulary".into(),
            ));
        }

        let runtime = self.runtime.as_ref().ok_or_else(|| {
            EngineError::InvalidState("Metal executor runtime was already unloaded".into())
        })?;
        let schedule = MetalDecodeSchedule::qwen38(&self.config)?;
        let projections = MetalProjectionPlan::qwen38(&self.config)?;
        let binding_plan = MetalDecodeBindingPlan::qwen38(&schedule, &projections, &self.config)?;
        let workspace_plan =
            MetalDecodeWorkspacePlan::qwen38(&schedule, &self.config, mtp_draft_token_ids.len())?;
        let _mtp_workspace_plan = MetalMtpWorkspacePlan::qwen38(&self.config)?;
        let mapping = runtime.map_artifact_no_copy(artifact)?;
        let target_cache = self.cache_config(admitted_context, &profile.kv)?;
        let mtp_cache = self.cache_config(admitted_context, &profile.mtp_kv)?;
        let workspace = runtime.prepare_decode_workspace(&workspace_plan)?;
        let target = runtime.prepare_mapped_target_core(&mapping, 0, target_cache)?;
        let mtp = runtime.prepare_mapped_mtp_core(&mapping, 0, mtp_cache, mtp_draft_token_ids)?;
        if target.copied_model_bytes() != 0 || mtp.copied_model_bytes() != 0 {
            return Err(EngineError::InvalidState(
                "Metal executor prepared a duplicate model allocation".into(),
            ));
        }

        let target_graph_bytes = target.transient_bytes()?;
        let mtp_graph_bytes = mtp.transient_bytes()?;
        let graph_bytes = u64::try_from(
            workspace
                .total_bytes()
                .checked_add(target_graph_bytes)
                .and_then(|bytes| bytes.checked_add(mtp_graph_bytes))
                .ok_or_else(|| EngineError::MemoryBudget("Metal graph bytes overflow".into()))?,
        )
        .map_err(|_| EngineError::MemoryBudget("Metal graph bytes exceed u64".into()))?;
        let session_bytes = u64::try_from(
            target
                .resident_state_bytes()?
                .checked_add(mtp.resident_state_bytes())
                .ok_or_else(|| EngineError::MemoryBudget("Metal session bytes overflow".into()))?,
        )
        .map_err(|_| EngineError::MemoryBudget("Metal session bytes exceed u64".into()))?
        .checked_add(profile.speculative_linear_state_bytes_per_session)
        .ok_or_else(|| EngineError::MemoryBudget("Metal checkpoint bytes overflow".into()))?;
        let allocations = AllocationSnapshot {
            model_bytes: mapping.mapped_file_bytes(),
            graph_bytes,
            session_bytes,
            ..AllocationSnapshot::default()
        };
        if allocations.total_bytes()? > profile.hard_limit_bytes {
            return Err(EngineError::MemoryBudget(format!(
                "Metal prepared residency {} exceeds hard limit {}",
                allocations.total_bytes()?,
                profile.hard_limit_bytes
            )));
        }

        self.mapping = Some(mapping);
        self.binding_plan = Some(binding_plan);
        self.workspace = Some(workspace);
        self.target = Some(target);
        self.mtp = Some(mtp);
        self.admitted_context = admitted_context;
        self.admitted_draft_tokens = profile.mtp_draft_tokens as usize;
        self.allocations = allocations;
        Ok(())
    }

    fn warmup(&mut self) -> Result<()> {
        if !self.is_loaded() || self.runtime.is_none() {
            return Err(EngineError::InvalidState(
                "Metal executor is not loaded".into(),
            ));
        }
        self.warmed = true;
        Ok(())
    }

    fn prefill(
        &mut self,
        tokens: &[u32],
        mtp_enabled: bool,
        cancellation: &CancellationToken,
    ) -> Result<ExecutorStep> {
        self.validate_ready(cancellation)?;
        if tokens.is_empty() || tokens.len() > self.admitted_context {
            return Err(EngineError::MemoryBudget(
                "Metal prefill requires an admitted non-empty token sequence".into(),
            ));
        }
        if tokens
            .iter()
            .any(|token| (*token as usize) >= TOKENIZER_VOCAB_SIZE)
        {
            return Err(EngineError::Shape(
                "Metal prefill token exceeds tokenizer vocabulary".into(),
            ));
        }
        self.reset_loaded_state()?;

        let result = (|| {
            let runtime = self.runtime.as_ref().expect("validated Metal runtime");
            let plan = self
                .binding_plan
                .as_ref()
                .expect("validated Metal binding plan");
            let workspace = self.workspace.as_ref().expect("validated Metal workspace");
            let program = workspace.bind_decode_program(plan)?;
            let target = self.target.as_mut().expect("validated Metal target");
            let mtp = self.mtp.as_mut().expect("validated Metal MTP");

            let mut committed = runtime.dispatch_prepared_mapped_complete_target_core(
                &program,
                target,
                tokens[0] as usize,
                0,
                0,
                self.admitted_context,
            )?;
            for token in &tokens[1..] {
                if cancellation.is_cancelled() {
                    return Err(EngineError::Cancelled);
                }
                committed = if mtp_enabled {
                    runtime
                        .dispatch_prepared_mapped_complete_token_verifier(
                            &program,
                            target,
                            mtp,
                            *token,
                            committed,
                            committed,
                            self.admitted_context,
                        )?
                        .committed_tokens
                } else {
                    runtime.dispatch_prepared_mapped_complete_target_core(
                        &program,
                        target,
                        *token as usize,
                        committed,
                        committed,
                        self.admitted_context,
                    )?
                };
            }
            if committed != tokens.len() {
                return Err(EngineError::InvalidState(format!(
                    "Metal prefill committed {committed} tokens, expected {}",
                    tokens.len()
                )));
            }
            if mtp_enabled && target.cached_tokens()? != mtp.cached_tokens().saturating_add(1) {
                return Err(EngineError::InvalidState(
                    "Metal prefill broke target-one-token-ahead MTP state".into(),
                ));
            }
            runtime.dispatch_prepared_mapped_target_argmax(&program, target, mtp)
        })();

        match result {
            Ok(token) => {
                self.last_target_token = Some(token);
                Ok(Self::resident_target_step())
            }
            Err(error) => {
                let _ = self.reset_loaded_state();
                Err(error)
            }
        }
    }

    fn decode(
        &mut self,
        token: u32,
        mtp_enabled: bool,
        cancellation: &CancellationToken,
    ) -> Result<ExecutorStep> {
        self.validate_ready(cancellation)?;
        if token as usize >= TOKENIZER_VOCAB_SIZE {
            return Err(EngineError::Shape(
                "Metal decode token exceeds tokenizer vocabulary".into(),
            ));
        }
        let runtime = self.runtime.as_ref().expect("validated Metal runtime");
        let plan = self
            .binding_plan
            .as_ref()
            .expect("validated Metal binding plan");
        let workspace = self.workspace.as_ref().expect("validated Metal workspace");
        let program = workspace.bind_decode_program(plan)?;
        let target = self.target.as_mut().expect("validated Metal target");
        let mtp = self.mtp.as_mut().expect("validated Metal MTP");
        let start = target.cached_tokens()?;

        if !mtp_enabled {
            let committed = runtime.dispatch_prepared_mapped_complete_target_core(
                &program,
                target,
                token as usize,
                start,
                start,
                self.admitted_context,
            )?;
            if committed != start + 1 {
                return Err(EngineError::InvalidState(
                    "Metal target-only decode published the wrong position".into(),
                ));
            }
            self.last_target_token =
                Some(runtime.dispatch_prepared_mapped_target_argmax(&program, target, mtp)?);
            return Ok(Self::resident_target_step());
        }

        let depth = self.admitted_draft_tokens;
        if depth == 0 {
            return Err(EngineError::InvalidState(
                "Metal MTP decode has no admitted draft depth".into(),
            ));
        }
        let mut records = Vec::with_capacity(depth);
        let (bonus_token, committed) = if depth == METAL_MTP4_RECORDS {
            let completed = runtime
                .dispatch_prepared_mapped_complete_greedy_mtp4_from_token_verifier(
                    &program,
                    target,
                    mtp,
                    token,
                    start,
                    start,
                    self.admitted_context,
                )?;
            records.extend(completed.outcome.records);
            (
                completed.outcome.prefix.next_token,
                completed.committed_tokens,
            )
        } else {
            let first = runtime.dispatch_prepared_mapped_complete_token_verifier(
                &program,
                target,
                mtp,
                token,
                start,
                start,
                self.admitted_context,
            )?;
            let mut committed = first.committed_tokens;
            records.push(first.verification);
            while records.len() < depth && records.last().is_some_and(|record| record.accepted) {
                let next = runtime.dispatch_prepared_mapped_complete_mtp_target_verifier(
                    &program,
                    target,
                    mtp,
                    committed,
                    committed,
                    self.admitted_context,
                )?;
                committed = next.committed_tokens;
                records.push(next.verification);
            }
            let bonus = if records.len() == depth && records.iter().all(|record| record.accepted) {
                let next = runtime.dispatch_prepared_mapped_complete_mtp_target_verifier(
                    &program,
                    target,
                    mtp,
                    committed,
                    committed,
                    self.admitted_context,
                )?;
                committed = next.committed_tokens;
                next.verification.target_token
            } else {
                records
                    .last()
                    .expect("Metal MTP sequence contains its initial record")
                    .target_token
            };
            (bonus, committed)
        };
        let accepted = Self::accepted_prefix(&records);
        let expected_committed = start
            .checked_add(1 + accepted)
            .ok_or_else(|| EngineError::MemoryBudget("Metal decode position overflows".into()))?;
        if committed != expected_committed {
            return Err(EngineError::InvalidState(format!(
                "Metal MTP state committed {committed} tokens, expected {expected_committed}"
            )));
        }
        self.pending_ack = Some(PendingMetalSpeculativeAck {
            accepted_drafts: accepted as u32,
        });
        Ok(Self::compact_step(&records, bonus_token))
    }

    fn select_target_token(&mut self, sampling: SamplerConfig, _draw: f32) -> Result<Option<u32>> {
        let _ = Sampler::new(sampling)?;
        if sampling.temperature != 0.0 {
            return Err(EngineError::UnsupportedOperation {
                backend: "metal",
                operation: "resident target sampling",
                reason: "Metal top-k/top-p sampling is not yet promoted; greedy selection remains device-resident"
                    .into(),
            });
        }
        self.last_target_token.take().map(Some).ok_or_else(|| {
            EngineError::InvalidState("Metal executor has no resident target selection".into())
        })
    }

    fn commit_speculative(
        &mut self,
        accepted_drafts: u32,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        let pending = self.pending_ack.take().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal executor has no speculative result to acknowledge".into(),
            )
        })?;
        if cancellation.is_cancelled() {
            self.reset_loaded_state()?;
            return Err(EngineError::Cancelled);
        }
        if accepted_drafts != pending.accepted_drafts {
            self.reset_loaded_state()?;
            return Err(EngineError::InvalidState(format!(
                "Metal engine accepted {accepted_drafts} drafts, device committed {}",
                pending.accepted_drafts
            )));
        }
        Ok(())
    }

    fn reset_session(&mut self) -> Result<()> {
        self.reset_loaded_state()
    }

    fn unload(&mut self) -> Result<()> {
        if self.is_loaded() {
            self.reset_loaded_state()?;
        }
        self.warmed = false;
        self.pending_ack = None;
        self.last_target_token = None;
        self.admitted_context = 0;
        self.admitted_draft_tokens = 0;
        drop(self.target.take());
        drop(self.mtp.take());
        drop(self.workspace.take());
        drop(self.binding_plan.take());
        drop(self.mapping.take());
        drop(self.runtime.take());
        self.allocations = AllocationSnapshot::default();
        Ok(())
    }

    fn allocations(&self) -> AllocationSnapshot {
        self.allocations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_prefix_is_strictly_causal() {
        let records = [
            MetalGreedyMtpVerification {
                target_token: 11,
                draft_token: 11,
                accepted: true,
            },
            MetalGreedyMtpVerification {
                target_token: 12,
                draft_token: 99,
                accepted: false,
            },
            MetalGreedyMtpVerification {
                target_token: 13,
                draft_token: 13,
                accepted: true,
            },
        ];
        assert_eq!(MetalModelExecutor::accepted_prefix(&records), 1);
    }

    #[test]
    fn compact_step_preserves_device_record_order() {
        let records = [
            MetalGreedyMtpVerification {
                target_token: 7,
                draft_token: 7,
                accepted: true,
            },
            MetalGreedyMtpVerification {
                target_token: 8,
                draft_token: 9,
                accepted: false,
            },
        ];
        let step = MetalModelExecutor::compact_step(&records, 10);
        assert_eq!(
            step.compact_greedy_mtp,
            Some(GreedyMtpVerification {
                draft_tokens: vec![7, 9],
                target_tokens: vec![7, 8],
                bonus_token: 10,
            })
        );
    }

    #[test]
    fn unloaded_executor_owns_no_reported_allocation() {
        let mut executor = MetalModelExecutor::new_for_profile("apple-m5-qwen38-verifier")
            .expect("compile Metal runtime");
        assert_eq!(executor.hardware_profile(), "apple-m5-qwen38-verifier");
        assert!(executor.capabilities().compact_greedy_mtp_verification);
        assert!(executor.capabilities().resident_target_selection);
        assert!(executor.allocations().is_zero());
        executor.unload().expect("drop unloaded Metal runtime");
        assert!(executor.allocations().is_zero());
    }

    #[test]
    fn empty_hardware_profile_is_rejected_before_runtime_compilation() {
        let error = MetalModelExecutor::new_for_profile("   ")
            .err()
            .expect("empty profile must fail");
        assert!(matches!(error, EngineError::InvalidArtifact(_)));
    }
}
