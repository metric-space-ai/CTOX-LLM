//! Embeddable model lifecycle shared by the library and local server.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::backend::{BackendKind, ExecutionPolicy, PromotionState};
use crate::loader::{ChecksumPolicy, ModelArtifact};
use crate::release::{MemoryProfile, ReleaseManifest};
use crate::sampler::{Sampler, SamplerConfig};
use crate::tensor_contract::validate_tensor_contract;
use crate::{EngineError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineLifecycle {
    Loaded,
    Warm,
    Active,
    UnloadFailed,
    Unloaded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorCapabilities {
    pub vocab_size: usize,
    pub maximum_context_tokens: u64,
    pub mtp: bool,
    pub maximum_draft_tokens: u32,
    pub cancellation: bool,
    pub session_reset: bool,
    pub no_hidden_fallbacks: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationSnapshot {
    pub model_bytes: u64,
    pub graph_bytes: u64,
    pub session_bytes: u64,
    pub scratch_bytes: u64,
    pub global_cache_bytes: u64,
}

impl AllocationSnapshot {
    pub fn total_bytes(self) -> Result<u64> {
        [
            self.model_bytes,
            self.graph_bytes,
            self.session_bytes,
            self.scratch_bytes,
            self.global_cache_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |sum, value| {
            sum.checked_add(value)
                .ok_or_else(|| EngineError::MemoryBudget("executor allocation overflow".into()))
        })
    }

    pub fn is_zero(self) -> bool {
        self == Self::default()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutorStep {
    pub target_logits: Vec<f32>,
    /// Unverified chained draft distributions.
    pub draft_logits: Vec<Vec<f32>>,
    /// Target distributions for the same candidate positions. Element zero
    /// must equal `target_logits`.
    pub target_verification_logits: Vec<Vec<f32>>,
    /// Target distribution after all candidate tokens. The engine consumes it
    /// only when every draft is accepted.
    pub bonus_logits: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedStep {
    pub token_id: u32,
    pub draft_tokens_proposed: u32,
    pub draft_tokens_verified: u32,
    pub accepted_draft_tokens: Vec<u32>,
}

impl ExecutorStep {
    fn validate(&self, capabilities: &ExecutorCapabilities, mtp_enabled: bool) -> Result<()> {
        if self.target_logits.len() != capabilities.vocab_size
            || self.target_logits.iter().any(|value| !value.is_finite())
        {
            return Err(EngineError::InvalidArtifact(
                "executor returned invalid target logits".into(),
            ));
        }
        if mtp_enabled {
            if !capabilities.mtp
                || self.draft_logits.is_empty()
                || self.draft_logits.len() > capabilities.maximum_draft_tokens as usize
                || self.target_verification_logits.len() != self.draft_logits.len()
                || self.target_verification_logits.first() != Some(&self.target_logits)
                || self.draft_logits.iter().any(|logits| {
                    logits.len() != capabilities.vocab_size
                        || logits.iter().any(|value| !value.is_finite())
                })
                || self.target_verification_logits.iter().any(|logits| {
                    logits.len() != capabilities.vocab_size
                        || logits.iter().any(|value| !value.is_finite())
                })
                || self.bonus_logits.as_ref().is_none_or(|logits| {
                    logits.len() != capabilities.vocab_size
                        || logits.iter().any(|value| !value.is_finite())
                })
            {
                return Err(EngineError::InvalidArtifact(
                    "MTP output contains an invalid or unverifiable draft distribution".into(),
                ));
            }
        } else if !self.draft_logits.is_empty()
            || !self.target_verification_logits.is_empty()
            || self.bonus_logits.is_some()
        {
            return Err(EngineError::InvalidArtifact(
                "executor returned MTP drafts while MTP is disabled".into(),
            ));
        }
        Ok(())
    }
}

pub trait ModelExecutor {
    fn backend_kind(&self) -> BackendKind;
    fn hardware_profile(&self) -> &str;
    fn promotion_state(&self) -> PromotionState;
    fn capabilities(&self) -> ExecutorCapabilities;
    fn load(&mut self, artifact: &ModelArtifact, profile: &MemoryProfile) -> Result<()>;
    fn warmup(&mut self) -> Result<()>;
    fn prefill(
        &mut self,
        tokens: &[u32],
        mtp_enabled: bool,
        cancellation: &CancellationToken,
    ) -> Result<ExecutorStep>;
    fn decode(
        &mut self,
        token: u32,
        mtp_enabled: bool,
        cancellation: &CancellationToken,
    ) -> Result<ExecutorStep>;
    /// Resolve the state branch prepared by an MTP decode. `accepted_drafts`
    /// is the causally verified prefix length selected by the engine.
    fn commit_speculative(
        &mut self,
        accepted_drafts: u32,
        cancellation: &CancellationToken,
    ) -> Result<()>;
    fn reset_session(&mut self) -> Result<()>;
    fn unload(&mut self) -> Result<()>;
    fn allocations(&self) -> AllocationSnapshot;
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SessionOptions {
    pub id: u64,
    pub mtp_enabled: bool,
    pub sampling: SamplerConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionStatus {
    pub id: u64,
    pub context_tokens: u64,
    pub mtp_enabled: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineMetrics {
    pub cold_load_micros: u64,
    pub warmup_micros: u64,
    pub prefill_calls: u64,
    pub decode_calls: u64,
    pub last_prefill_micros: u64,
    pub last_decode_micros: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineHealth {
    pub lifecycle: EngineLifecycle,
    pub release_id: String,
    pub pack_id: String,
    pub memory_profile_id: String,
    pub backend: BackendKind,
    pub hardware_profile: String,
    pub promotion_state: PromotionState,
    pub session: Option<SessionStatus>,
    pub allocations: AllocationSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadProgress {
    SignatureVerified,
    ArtifactOpened,
    ArtifactAdmitted,
    BackendLoaded,
}

pub struct Engine<E: ModelExecutor> {
    release_id: String,
    pack_id: String,
    memory_profile: MemoryProfile,
    backend: BackendKind,
    hardware_profile: String,
    promotion_state: PromotionState,
    capabilities: ExecutorCapabilities,
    artifact: Option<ModelArtifact>,
    executor: E,
    lifecycle: EngineLifecycle,
    session: Option<SessionStatus>,
    sampler: Option<Sampler>,
    metrics: EngineMetrics,
}

impl<E: ModelExecutor> Engine<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn load_signed(
        artifact_path: impl AsRef<Path>,
        release: &ReleaseManifest,
        pack_id: &str,
        memory_profile_id: &str,
        expected_key_id: &str,
        trusted_public_key: &[u8; 32],
        policy: ExecutionPolicy,
        executor: E,
        mut progress: impl FnMut(LoadProgress),
    ) -> Result<Self> {
        release.verify_signature(expected_key_id, trusted_public_key)?;
        progress(LoadProgress::SignatureVerified);
        Self::load_preverified_release(
            artifact_path,
            release,
            pack_id,
            memory_profile_id,
            policy,
            executor,
            progress,
        )
    }

    /// Development entry point for a manifest authenticated by a containing
    /// trusted bundle. Production download activation uses `load_signed`.
    pub fn load_preverified_release(
        artifact_path: impl AsRef<Path>,
        release: &ReleaseManifest,
        pack_id: &str,
        memory_profile_id: &str,
        policy: ExecutionPolicy,
        mut executor: E,
        mut progress: impl FnMut(LoadProgress),
    ) -> Result<Self> {
        let started = Instant::now();
        release.validate()?;
        let pack = release.backend_pack(pack_id)?;
        let memory_profile = release.memory_profile(memory_profile_id)?.clone();
        if memory_profile.pack_id != pack_id {
            return Err(EngineError::InvalidArtifact(format!(
                "memory profile {memory_profile_id} belongs to {}, not {pack_id}",
                memory_profile.pack_id
            )));
        }
        let capabilities = executor.capabilities();
        validate_executor(
            pack.backend,
            &pack.hardware_profile,
            &memory_profile,
            policy,
            &executor,
        )?;

        let artifact = ModelArtifact::open(artifact_path, ChecksumPolicy::AllTensors)?;
        progress(LoadProgress::ArtifactOpened);
        validate_tensor_contract(artifact.manifest(), &crate::Qwen38Config::default())?;
        release.admit_artifact(pack_id, &artifact)?;
        progress(LoadProgress::ArtifactAdmitted);
        if let Err(error) = executor.load(&artifact, &memory_profile) {
            let _ = executor.unload();
            return Err(error);
        }
        progress(LoadProgress::BackendLoaded);

        Ok(Self {
            release_id: release.release_id.clone(),
            pack_id: pack_id.into(),
            memory_profile,
            backend: pack.backend,
            hardware_profile: pack.hardware_profile.clone(),
            promotion_state: executor.promotion_state(),
            capabilities,
            artifact: Some(artifact),
            executor,
            lifecycle: EngineLifecycle::Loaded,
            session: None,
            sampler: None,
            metrics: EngineMetrics {
                cold_load_micros: elapsed_micros(started),
                ..EngineMetrics::default()
            },
        })
    }

    pub fn warmup(&mut self) -> Result<()> {
        match self.lifecycle {
            EngineLifecycle::Warm => return Ok(()),
            EngineLifecycle::Loaded => {}
            state => {
                return Err(EngineError::InvalidState(format!(
                    "warmup is invalid while engine is {state:?}"
                )))
            }
        }
        let started = Instant::now();
        self.executor.warmup()?;
        self.metrics.warmup_micros = elapsed_micros(started);
        self.lifecycle = EngineLifecycle::Warm;
        Ok(())
    }

    pub fn prefill(
        &mut self,
        options: SessionOptions,
        tokens: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<GeneratedStep> {
        if self.lifecycle != EngineLifecycle::Warm || self.session.is_some() {
            return Err(EngineError::InvalidState(
                "prefill requires a warm engine without an active session".into(),
            ));
        }
        if tokens.is_empty() || tokens.len() as u64 > self.memory_profile.context_tokens {
            return Err(EngineError::MemoryBudget(format!(
                "prefill has {} tokens, profile capacity is {}",
                tokens.len(),
                self.memory_profile.context_tokens
            )));
        }
        if options.mtp_enabled && !self.capabilities.mtp {
            return Err(EngineError::UnsupportedOperation {
                backend: "model_executor",
                operation: "mtp",
                reason: "executor does not advertise MTP".into(),
            });
        }
        if options.mtp_enabled && self.memory_profile.mtp_draft_tokens == 0 {
            return Err(EngineError::UnsupportedOperation {
                backend: "model_executor",
                operation: "mtp",
                reason: "the admitted memory profile reserves no MTP runtime state".into(),
            });
        }
        if options.mtp_enabled && options.sampling.temperature != 0.0 {
            return Err(EngineError::UnsupportedOperation {
                backend: "model_executor",
                operation: "mtp sampling",
                reason: "the current one-layer MTP verifier requires deterministic greedy sampling"
                    .into(),
            });
        }
        let mut sampler = Sampler::new(options.sampling)?;
        cancelled(cancellation)?;
        let started = Instant::now();
        let output = match self
            .executor
            .prefill(tokens, options.mtp_enabled, cancellation)
        {
            Ok(output) => output,
            Err(error) => {
                self.invalidate_session()?;
                return Err(error);
            }
        };
        if cancellation.is_cancelled() {
            self.invalidate_session()?;
            return Err(EngineError::Cancelled);
        }
        if let Err(error) = output.validate(&self.capabilities, false) {
            self.invalidate_session()?;
            return Err(error);
        }
        let token_id = sampler.sample(&output.target_logits)? as u32;
        self.session = Some(SessionStatus {
            id: options.id,
            context_tokens: tokens.len() as u64,
            mtp_enabled: options.mtp_enabled,
        });
        self.sampler = Some(sampler);
        self.lifecycle = EngineLifecycle::Active;
        self.metrics.prefill_calls += 1;
        self.metrics.last_prefill_micros = elapsed_micros(started);
        Ok(GeneratedStep {
            token_id,
            draft_tokens_proposed: 0,
            draft_tokens_verified: 0,
            accepted_draft_tokens: Vec::new(),
        })
    }

    pub fn decode(
        &mut self,
        session_id: u64,
        token: u32,
        cancellation: &CancellationToken,
    ) -> Result<GeneratedStep> {
        let session = *self
            .session
            .as_ref()
            .ok_or_else(|| EngineError::InvalidState("decode requires an active session".into()))?;
        if self.lifecycle != EngineLifecycle::Active || session.id != session_id {
            return Err(EngineError::InvalidState(
                "decode session id or lifecycle does not match".into(),
            ));
        }
        if session.context_tokens >= self.memory_profile.context_tokens {
            return Err(EngineError::MemoryBudget(
                "decode would exceed the admitted context capacity".into(),
            ));
        }
        let mtp_enabled = session.mtp_enabled;
        let execution_tokens = 1_u64
            .checked_add(if mtp_enabled {
                u64::from(self.memory_profile.mtp_draft_tokens)
            } else {
                0
            })
            .ok_or_else(|| EngineError::MemoryBudget("decode execution span overflows".into()))?;
        if session
            .context_tokens
            .checked_add(execution_tokens)
            .is_none_or(|tokens| tokens > self.memory_profile.context_tokens)
        {
            return Err(EngineError::MemoryBudget(
                "decode verification block would exceed the admitted context capacity".into(),
            ));
        }
        cancelled(cancellation)?;
        let started = Instant::now();
        let output = match self.executor.decode(token, mtp_enabled, cancellation) {
            Ok(output) => output,
            Err(error) => {
                self.invalidate_session()?;
                return Err(error);
            }
        };
        if cancellation.is_cancelled() {
            self.invalidate_session()?;
            return Err(EngineError::Cancelled);
        }
        if let Err(error) = output.validate(&self.capabilities, mtp_enabled) {
            self.invalidate_session()?;
            return Err(error);
        }
        if output.draft_logits.len() > self.memory_profile.mtp_draft_tokens as usize {
            self.invalidate_session()?;
            return Err(EngineError::MemoryBudget(
                "executor exceeded the admitted MTP draft depth".into(),
            ));
        }
        let draft_tokens_proposed = output.draft_logits.len() as u32;
        let (token_id, draft_tokens_verified, accepted_draft_tokens) = if mtp_enabled {
            let mut accepted = Vec::new();
            let mut verified = 0_u32;
            let mut fallback = None;
            for (draft_logits, target_logits) in output
                .draft_logits
                .iter()
                .zip(&output.target_verification_logits)
            {
                let draft = greedy_token(draft_logits)?;
                let target = greedy_token(target_logits)?;
                verified += 1;
                if draft != target {
                    fallback = Some(target);
                    break;
                }
                accepted.push(draft);
            }
            let token = match fallback {
                Some(token) => token,
                None => greedy_token(output.bonus_logits.as_ref().ok_or_else(|| {
                    EngineError::InvalidArtifact("MTP block has no bonus logits".into())
                })?)?,
            };
            if let Err(error) = self
                .executor
                .commit_speculative(accepted.len() as u32, cancellation)
            {
                self.invalidate_session()?;
                return Err(error);
            }
            (token, verified, accepted)
        } else {
            let token = self
                .sampler
                .as_mut()
                .ok_or_else(|| EngineError::InvalidState("active session has no sampler".into()))?
                .sample(&output.target_logits)? as u32;
            (token, 0, Vec::new())
        };
        let added = 1_u64
            .checked_add(accepted_draft_tokens.len() as u64)
            .ok_or_else(|| EngineError::MemoryBudget("decode token count overflows".into()))?;
        let new_context = session
            .context_tokens
            .checked_add(added)
            .ok_or_else(|| EngineError::MemoryBudget("session context overflows".into()))?;
        if new_context > self.memory_profile.context_tokens {
            self.invalidate_session()?;
            return Err(EngineError::MemoryBudget(
                "MTP decode advanced beyond the admitted context capacity".into(),
            ));
        }
        self.session
            .as_mut()
            .expect("session checked above")
            .context_tokens = new_context;
        self.metrics.decode_calls += 1;
        self.metrics.last_decode_micros = elapsed_micros(started);
        Ok(GeneratedStep {
            token_id,
            draft_tokens_proposed,
            draft_tokens_verified,
            accepted_draft_tokens,
        })
    }

    pub fn reset_session(&mut self) -> Result<()> {
        if self.lifecycle == EngineLifecycle::Unloaded {
            return Err(EngineError::InvalidState(
                "cannot reset an unloaded engine".into(),
            ));
        }
        if self.session.is_some() || self.lifecycle == EngineLifecycle::UnloadFailed {
            return self.invalidate_session();
        }
        Ok(())
    }

    fn invalidate_session(&mut self) -> Result<()> {
        let result = self.executor.reset_session();
        self.session = None;
        self.sampler = None;
        self.lifecycle = if result.is_ok() {
            EngineLifecycle::Warm
        } else {
            EngineLifecycle::UnloadFailed
        };
        result
    }

    pub fn unload(&mut self) -> Result<()> {
        if self.lifecycle == EngineLifecycle::Unloaded {
            return Ok(());
        }
        let reset_error = if self.session.is_some() {
            self.executor.reset_session().err()
        } else {
            None
        };
        self.session = None;
        self.sampler = None;
        let unload_error = self.executor.unload().err();
        let allocations = self.executor.allocations();
        if let Some(error) = reset_error.or(unload_error) {
            self.lifecycle = EngineLifecycle::UnloadFailed;
            return Err(error);
        }
        if !allocations.is_zero() {
            self.lifecycle = EngineLifecycle::UnloadFailed;
            return Err(EngineError::MemoryBudget(format!(
                "executor retained {} bytes after unload: {allocations:?}",
                allocations.total_bytes()?
            )));
        }
        self.artifact.take();
        self.lifecycle = EngineLifecycle::Unloaded;
        Ok(())
    }

    pub fn health(&self) -> EngineHealth {
        EngineHealth {
            lifecycle: self.lifecycle,
            release_id: self.release_id.clone(),
            pack_id: self.pack_id.clone(),
            memory_profile_id: self.memory_profile.profile_id.clone(),
            backend: self.backend,
            hardware_profile: self.hardware_profile.clone(),
            promotion_state: self.promotion_state,
            session: self.session,
            allocations: self.executor.allocations(),
        }
    }

    pub fn capabilities(&self) -> &ExecutorCapabilities {
        &self.capabilities
    }

    pub fn metrics(&self) -> &EngineMetrics {
        &self.metrics
    }
}

impl<E: ModelExecutor> Drop for Engine<E> {
    fn drop(&mut self) {
        if self.lifecycle != EngineLifecycle::Unloaded {
            let _ = self.executor.reset_session();
            let _ = self.executor.unload();
        }
    }
}

fn validate_executor<E: ModelExecutor>(
    backend: BackendKind,
    hardware_profile: &str,
    memory_profile: &MemoryProfile,
    policy: ExecutionPolicy,
    executor: &E,
) -> Result<()> {
    let capabilities = executor.capabilities();
    if executor.backend_kind() != backend
        || executor.hardware_profile() != hardware_profile
        || capabilities.maximum_context_tokens < memory_profile.context_tokens
        || capabilities.vocab_size == 0
        || capabilities.mtp != (capabilities.maximum_draft_tokens > 0)
        || capabilities.maximum_draft_tokens < memory_profile.mtp_draft_tokens
    {
        return Err(EngineError::UnsupportedOperation {
            backend: "model_executor",
            operation: "load",
            reason: "executor does not match the signed pack or memory profile".into(),
        });
    }
    if policy == ExecutionPolicy::Production
        && (executor.promotion_state() != PromotionState::Optimized
            || !capabilities.cancellation
            || !capabilities.session_reset
            || !capabilities.no_hidden_fallbacks)
    {
        return Err(EngineError::UnsupportedOperation {
            backend: "model_executor",
            operation: "production load",
            reason: "executor is not optimized or lacks lifecycle/fallback guarantees".into(),
        });
    }
    Ok(())
}

fn cancelled(token: &CancellationToken) -> Result<()> {
    if token.is_cancelled() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

fn greedy_token(logits: &[f32]) -> Result<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(token, _)| token as u32)
        .ok_or_else(|| EngineError::InvalidArtifact("greedy logits are empty".into()))
}

fn elapsed_micros(started: Instant) -> u64 {
    started.elapsed().as_micros().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockExecutor {
        allocations: AllocationSnapshot,
        leak_on_unload: bool,
        cancel_during_decode: bool,
        invalid_output: bool,
        reject_draft: bool,
        committed_drafts: Option<u32>,
    }

    impl MockExecutor {
        fn output(&self, mtp: bool) -> ExecutorStep {
            if self.invalid_output {
                return ExecutorStep {
                    target_logits: vec![0.0, f32::NAN, 2.0],
                    draft_logits: Vec::new(),
                    target_verification_logits: Vec::new(),
                    bonus_logits: None,
                };
            }
            let target_logits = vec![0.0, 1.0, 2.0];
            ExecutorStep {
                target_logits: target_logits.clone(),
                draft_logits: if mtp {
                    vec![if self.reject_draft {
                        vec![0.0, 2.0, 1.0]
                    } else {
                        vec![0.0, 1.0, 2.0]
                    }]
                } else {
                    Vec::new()
                },
                target_verification_logits: if mtp { vec![target_logits] } else { Vec::new() },
                bonus_logits: mtp.then_some(vec![0.0, 2.0, 1.0]),
            }
        }
    }

    impl ModelExecutor for MockExecutor {
        fn backend_kind(&self) -> BackendKind {
            BackendKind::Cuda
        }

        fn hardware_profile(&self) -> &str {
            "sm86"
        }

        fn promotion_state(&self) -> PromotionState {
            PromotionState::Optimized
        }

        fn capabilities(&self) -> ExecutorCapabilities {
            ExecutorCapabilities {
                vocab_size: 3,
                maximum_context_tokens: 16,
                mtp: true,
                maximum_draft_tokens: 1,
                cancellation: true,
                session_reset: true,
                no_hidden_fallbacks: true,
            }
        }

        fn load(&mut self, _artifact: &ModelArtifact, _profile: &MemoryProfile) -> Result<()> {
            self.allocations.model_bytes = 10;
            Ok(())
        }

        fn warmup(&mut self) -> Result<()> {
            Ok(())
        }

        fn prefill(
            &mut self,
            _tokens: &[u32],
            _mtp_enabled: bool,
            _cancellation: &CancellationToken,
        ) -> Result<ExecutorStep> {
            self.allocations.session_bytes = 4;
            Ok(self.output(false))
        }

        fn decode(
            &mut self,
            _token: u32,
            mtp_enabled: bool,
            cancellation: &CancellationToken,
        ) -> Result<ExecutorStep> {
            if self.cancel_during_decode {
                cancellation.cancel();
            }
            Ok(self.output(mtp_enabled))
        }

        fn commit_speculative(
            &mut self,
            accepted_drafts: u32,
            cancellation: &CancellationToken,
        ) -> Result<()> {
            cancelled(cancellation)?;
            self.committed_drafts = Some(accepted_drafts);
            Ok(())
        }

        fn reset_session(&mut self) -> Result<()> {
            self.allocations.session_bytes = 0;
            Ok(())
        }

        fn unload(&mut self) -> Result<()> {
            self.allocations = if self.leak_on_unload {
                AllocationSnapshot {
                    global_cache_bytes: 1,
                    ..AllocationSnapshot::default()
                }
            } else {
                AllocationSnapshot::default()
            };
            Ok(())
        }

        fn allocations(&self) -> AllocationSnapshot {
            self.allocations
        }
    }

    fn profile() -> MemoryProfile {
        MemoryProfile {
            profile_id: "test".into(),
            pack_id: "pack".into(),
            context_tokens: 16,
            sessions: 1,
            resident_model_bytes: 10,
            persistent_backend_graph_bytes: 0,
            persistent_runtime_bytes: 0,
            linear_state_dtype: crate::memory::LinearStateDType::F32,
            linear_state_bytes_per_session: 0,
            mtp_draft_tokens: 1,
            speculative_state_strategy: crate::memory::SpeculativeStateStrategy::ReplayOnReject,
            speculative_linear_state_bytes_per_session: 0,
            kv: crate::release::KvMemoryFormula {
                fixed_bytes_per_session: 0,
                bytes_per_token_per_session: 1,
                retained_q4_tokens_per_session: 0,
                q4_delta_bytes_per_token: 0,
            },
            mtp_kv: crate::release::KvMemoryFormula {
                fixed_bytes_per_session: 1,
                bytes_per_token_per_session: 0,
                retained_q4_tokens_per_session: 0,
                q4_delta_bytes_per_token: 0,
            },
            prefill_scratch_peak_bytes: 0,
            decode_scratch_peak_bytes: 0,
            loader_transient_peak_bytes: 0,
            accelerator_unattributed_reserve_bytes: 0,
            hard_limit_bytes: 1024,
        }
    }

    fn engine(executor: MockExecutor) -> Engine<MockExecutor> {
        let capabilities = executor.capabilities();
        Engine {
            release_id: "test".into(),
            pack_id: "pack".into(),
            memory_profile: profile(),
            backend: BackendKind::Cuda,
            hardware_profile: "sm86".into(),
            promotion_state: PromotionState::Optimized,
            capabilities,
            artifact: None,
            executor,
            lifecycle: EngineLifecycle::Loaded,
            session: None,
            sampler: None,
            metrics: EngineMetrics::default(),
        }
    }

    #[test]
    fn lifecycle_runs_incremental_mtp_and_releases_every_allocation() {
        let mut engine = engine(MockExecutor {
            allocations: AllocationSnapshot {
                model_bytes: 10,
                ..AllocationSnapshot::default()
            },
            leak_on_unload: false,
            cancel_during_decode: false,
            invalid_output: false,
            reject_draft: false,
            committed_drafts: None,
        });
        engine.warmup().unwrap();
        engine
            .prefill(
                SessionOptions {
                    id: 7,
                    mtp_enabled: true,
                    sampling: SamplerConfig {
                        temperature: 0.0,
                        ..SamplerConfig::default()
                    },
                },
                &[1, 2, 3],
                &CancellationToken::default(),
            )
            .unwrap();
        let step = engine.decode(7, 2, &CancellationToken::default()).unwrap();
        assert_eq!(step.token_id, 1);
        assert_eq!(step.draft_tokens_proposed, step.draft_tokens_verified);
        assert_eq!(step.accepted_draft_tokens, vec![2]);
        assert_eq!(engine.health().session.unwrap().context_tokens, 5);
        assert_eq!(engine.executor.committed_drafts, Some(1));
        engine.reset_session().unwrap();
        engine.unload().unwrap();
        assert_eq!(engine.health().lifecycle, EngineLifecycle::Unloaded);
        assert!(engine.health().allocations.is_zero());
    }

    #[test]
    fn rejected_mtp_draft_commits_only_the_target_input() {
        let mut engine = engine(MockExecutor {
            allocations: AllocationSnapshot::default(),
            leak_on_unload: false,
            cancel_during_decode: false,
            invalid_output: false,
            reject_draft: true,
            committed_drafts: None,
        });
        engine.warmup().unwrap();
        engine
            .prefill(
                SessionOptions {
                    id: 8,
                    mtp_enabled: true,
                    sampling: SamplerConfig {
                        temperature: 0.0,
                        ..SamplerConfig::default()
                    },
                },
                &[1, 2, 3],
                &CancellationToken::default(),
            )
            .unwrap();
        let step = engine.decode(8, 2, &CancellationToken::default()).unwrap();
        assert_eq!(step.token_id, 2);
        assert_eq!(step.draft_tokens_proposed, 1);
        assert_eq!(step.draft_tokens_verified, 1);
        assert!(step.accepted_draft_tokens.is_empty());
        assert_eq!(engine.executor.committed_drafts, Some(0));
        assert_eq!(engine.health().session.unwrap().context_tokens, 4);
    }

    #[test]
    fn mtp_verification_reserves_the_complete_execution_span() {
        let mut engine = engine(MockExecutor {
            allocations: AllocationSnapshot::default(),
            leak_on_unload: false,
            cancel_during_decode: false,
            invalid_output: false,
            reject_draft: false,
            committed_drafts: None,
        });
        engine.warmup().unwrap();
        engine
            .prefill(
                SessionOptions {
                    id: 9,
                    mtp_enabled: true,
                    sampling: SamplerConfig {
                        temperature: 0.0,
                        ..SamplerConfig::default()
                    },
                },
                &[0; 15],
                &CancellationToken::default(),
            )
            .unwrap();
        assert!(matches!(
            engine.decode(9, 2, &CancellationToken::default()),
            Err(EngineError::MemoryBudget(message))
                if message.contains("verification block")
        ));
        assert_eq!(engine.health().session.unwrap().context_tokens, 15);
        assert_eq!(engine.executor.committed_drafts, None);
    }

    #[test]
    fn cancellation_resets_partial_session_state() {
        let mut engine = engine(MockExecutor {
            allocations: AllocationSnapshot::default(),
            leak_on_unload: false,
            cancel_during_decode: true,
            invalid_output: false,
            reject_draft: false,
            committed_drafts: None,
        });
        engine.warmup().unwrap();
        engine
            .prefill(
                SessionOptions {
                    id: 1,
                    mtp_enabled: false,
                    sampling: SamplerConfig::default(),
                },
                &[1],
                &CancellationToken::default(),
            )
            .unwrap();
        assert!(matches!(
            engine.decode(1, 2, &CancellationToken::default()),
            Err(EngineError::Cancelled)
        ));
        assert!(engine.health().session.is_none());
        assert_eq!(engine.health().lifecycle, EngineLifecycle::Warm);
    }

    #[test]
    fn stochastic_mtp_fails_closed_until_rejection_sampling_exists() {
        let mut engine = engine(MockExecutor {
            allocations: AllocationSnapshot::default(),
            leak_on_unload: false,
            cancel_during_decode: false,
            invalid_output: false,
            reject_draft: false,
            committed_drafts: None,
        });
        engine.warmup().unwrap();
        assert!(matches!(
            engine.prefill(
                SessionOptions {
                    id: 1,
                    mtp_enabled: true,
                    sampling: SamplerConfig::default(),
                },
                &[1],
                &CancellationToken::default(),
            ),
            Err(EngineError::UnsupportedOperation {
                operation: "mtp sampling",
                ..
            })
        ));
        assert!(engine.health().session.is_none());
    }

    #[test]
    fn unload_fails_closed_on_process_global_allocator_residue() {
        let mut engine = engine(MockExecutor {
            allocations: AllocationSnapshot::default(),
            leak_on_unload: true,
            cancel_during_decode: false,
            invalid_output: false,
            reject_draft: false,
            committed_drafts: None,
        });
        assert!(engine.unload().is_err());
        assert_eq!(engine.health().lifecycle, EngineLifecycle::UnloadFailed);
        assert_eq!(engine.health().allocations.global_cache_bytes, 1);
    }

    #[test]
    fn production_executor_contract_rejects_missing_fallback_guarantees() {
        let executor = MockExecutor {
            allocations: AllocationSnapshot::default(),
            leak_on_unload: false,
            cancel_during_decode: false,
            invalid_output: false,
            reject_draft: false,
            committed_drafts: None,
        };
        validate_executor(
            BackendKind::Cuda,
            "sm86",
            &profile(),
            ExecutionPolicy::Production,
            &executor,
        )
        .unwrap();
        assert!(validate_executor(
            BackendKind::Metal,
            "sm86",
            &profile(),
            ExecutionPolicy::Production,
            &executor,
        )
        .is_err());
    }

    #[test]
    fn malformed_executor_output_invalidates_partial_session_state() {
        let mut engine = engine(MockExecutor {
            allocations: AllocationSnapshot::default(),
            leak_on_unload: false,
            cancel_during_decode: false,
            invalid_output: true,
            reject_draft: false,
            committed_drafts: None,
        });
        engine.warmup().unwrap();
        assert!(engine
            .prefill(
                SessionOptions {
                    id: 1,
                    mtp_enabled: false,
                    sampling: SamplerConfig::default(),
                },
                &[1],
                &CancellationToken::default(),
            )
            .is_err());
        assert!(engine.health().session.is_none());
        assert_eq!(engine.health().lifecycle, EngineLifecycle::Warm);
        assert_eq!(engine.health().allocations.session_bytes, 0);
    }

    #[test]
    fn session_sampler_keeps_seeded_token_decisions_in_the_shared_engine() {
        let make = || {
            engine(MockExecutor {
                allocations: AllocationSnapshot::default(),
                leak_on_unload: false,
                cancel_during_decode: false,
                invalid_output: false,
                reject_draft: false,
                committed_drafts: None,
            })
        };
        let mut left = make();
        let mut right = make();
        left.warmup().unwrap();
        right.warmup().unwrap();
        let options = SessionOptions {
            id: 4,
            mtp_enabled: false,
            sampling: SamplerConfig {
                seed: 91,
                ..SamplerConfig::default()
            },
        };
        let left_first = left
            .prefill(options, &[1], &CancellationToken::default())
            .unwrap();
        let right_first = right
            .prefill(options, &[1], &CancellationToken::default())
            .unwrap();
        assert_eq!(left_first.token_id, right_first.token_id);
        for input in [0, 1, 2, 1, 0] {
            assert_eq!(
                left.decode(4, input, &CancellationToken::default())
                    .unwrap()
                    .token_id,
                right
                    .decode(4, input, &CancellationToken::default())
                    .unwrap()
                    .token_id
            );
        }
    }
}
