//! Embeddable SM86 executor over the model-specific CUDA graph.
//!
//! This remains a verifier candidate until full-model numerical, quality,
//! unload, and roofline gates promote it. It never falls back to CPU model
//! operators. The server path returns compact device-selected MTP decisions;
//! the dedicated evidence constructor retains complete logit readbacks for
//! numerical hashes. Ordinary target selection reuses the resident LM-head
//! output through device argmax or bounded top-k/top-p sampling. MTP proposals
//! use the release-bound gathered LM head.

use std::sync::mpsc::{self, SyncSender};
use std::thread::{self, JoinHandle};

use crate::backend::cuda_graph::PreparedCudaProjectionGraph;
use crate::backend::cuda_runtime::{
    CudaCandidateRuntime, CudaDeviceF32View, CudaSubmissionStats, PreparedCudaArgmax,
    PreparedCudaTopKTopPSampler,
};
use crate::backend::{BackendKind, PromotionState};
use crate::engine::{
    AllocationSnapshot, CancellationToken, DraftDistribution, ExecutorCapabilities, ExecutorStep,
    GreedyMtpVerification, ModelExecutor,
};
use crate::loader::ModelArtifact;
use crate::memory::{LinearStateDType, SpeculativeStateStrategy};
use crate::release::MemoryProfile;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::TOKENIZER_VOCAB_SIZE;
use crate::{EngineError, Qwen38Config, Result};

const MAXIMUM_CHAINED_MTP_DRAFTS: usize = 4;
pub const CUDA_SM86_EXECUTOR_PROFILE: &str = "cuda-sm86-qwen38-verifier";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaGatheredMtpVerification {
    pub rows: usize,
    pub mismatched_rows: usize,
    pub maximum_absolute_error: f32,
}

#[derive(Debug)]
struct PendingCudaSpeculativeBranch {
    candidate_tokens: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaMtpOutputMode {
    CompactGreedy,
    FullVerifierLogits,
}

pub struct CudaModelExecutor {
    config: Qwen38Config,
    runtime: Option<CudaCandidateRuntime>,
    graph: Option<PreparedCudaProjectionGraph>,
    argmax: Option<PreparedCudaArgmax>,
    target_sampler: Option<PreparedCudaTopKTopPSampler>,
    pending_speculative: Option<PendingCudaSpeculativeBranch>,
    mtp_draft_token_ids: Vec<u32>,
    admitted_context: usize,
    admitted_draft_tokens: usize,
    free_bytes_before_graph: Option<usize>,
    mtp_output_mode: CudaMtpOutputMode,
    warmed: bool,
    allocations: AllocationSnapshot,
}

enum CudaWorkerCommand {
    Load {
        artifact: Box<ModelArtifact>,
        profile: Box<MemoryProfile>,
        mtp_draft_token_ids: Vec<u32>,
        reply: SyncSender<Result<()>>,
    },
    Warmup {
        reply: SyncSender<Result<()>>,
    },
    Prefill {
        tokens: Vec<u32>,
        mtp_enabled: bool,
        cancellation: CancellationToken,
        reply: SyncSender<Result<ExecutorStep>>,
    },
    Decode {
        token: u32,
        mtp_enabled: bool,
        cancellation: CancellationToken,
        reply: SyncSender<Result<ExecutorStep>>,
    },
    SelectTarget {
        sampling: SamplerConfig,
        draw: f32,
        reply: SyncSender<Result<Option<u32>>>,
    },
    CommitSpeculative {
        accepted_drafts: u32,
        cancellation: CancellationToken,
        reply: SyncSender<Result<()>>,
    },
    ResetSession {
        reply: SyncSender<Result<()>>,
    },
    Unload {
        reply: SyncSender<Result<()>>,
    },
    Allocations {
        reply: SyncSender<Result<AllocationSnapshot>>,
    },
    SubmissionStats {
        reply: SyncSender<Result<CudaSubmissionStats>>,
    },
    SessionTokenCounters {
        reply: SyncSender<Result<(usize, usize)>>,
    },
    VerifyGatheredMtp {
        token: u32,
        reply: SyncSender<Result<CudaGatheredMtpVerification>>,
    },
    Shutdown {
        reply: SyncSender<Result<()>>,
    },
}

/// Sendable server adapter whose worker thread exclusively owns the CUDA
/// context, graph, and allocator lifecycle. CUDA objects themselves remain
/// deliberately `!Send`; no unsafe marker trait is used to cross that boundary.
pub struct ThreadedCudaModelExecutor {
    sender: Option<mpsc::Sender<CudaWorkerCommand>>,
    worker: Option<JoinHandle<()>>,
    allocations: AllocationSnapshot,
    compact_greedy_mtp_verification: bool,
}

impl CudaModelExecutor {
    pub fn new_sm86(cubin: &[u8], device: i32) -> Result<Self> {
        Self::new_sm86_with_output_mode(cubin, device, CudaMtpOutputMode::CompactGreedy)
    }

    fn new_sm86_with_full_verifier_logits(cubin: &[u8], device: i32) -> Result<Self> {
        Self::new_sm86_with_output_mode(cubin, device, CudaMtpOutputMode::FullVerifierLogits)
    }

    fn new_sm86_with_output_mode(
        cubin: &[u8],
        device: i32,
        mtp_output_mode: CudaMtpOutputMode,
    ) -> Result<Self> {
        Ok(Self {
            config: Qwen38Config::default(),
            runtime: Some(CudaCandidateRuntime::new(cubin, device)?),
            graph: None,
            argmax: None,
            target_sampler: None,
            pending_speculative: None,
            mtp_draft_token_ids: Vec::new(),
            admitted_context: 0,
            admitted_draft_tokens: 0,
            free_bytes_before_graph: None,
            mtp_output_mode,
            warmed: false,
            allocations: AllocationSnapshot::default(),
        })
    }

    fn runtime(&self) -> Result<&CudaCandidateRuntime> {
        self.runtime.as_ref().ok_or_else(|| {
            EngineError::InvalidState("CUDA executor context is not available".into())
        })
    }

    fn validate_loaded_decode(&self, cancellation: &CancellationToken) -> Result<()> {
        if cancellation.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        if !self.warmed
            || self.graph.is_none()
            || self.argmax.is_none()
            || self.target_sampler.is_none()
            || self.runtime.is_none()
        {
            return Err(EngineError::InvalidState(
                "CUDA decode requires a warm loaded executor".into(),
            ));
        }
        if self.pending_speculative.is_some() {
            return Err(EngineError::InvalidState(
                "CUDA decode requires the previous speculative branch to be committed".into(),
            ));
        }
        Ok(())
    }

    fn canonical_draft_vocabulary(&self, ids: &[u32]) -> bool {
        !ids.is_empty()
            && ids.windows(2).all(|pair| pair[0] < pair[1])
            && ids
                .iter()
                .all(|token| (*token as usize) < TOKENIZER_VOCAB_SIZE)
    }

    /// Submission accounting is exposed only for verifier evidence. Production
    /// callers consume the stable `ModelExecutor` ABI instead.
    pub fn submission_stats(&self) -> Option<CudaSubmissionStats> {
        self.runtime
            .as_ref()
            .map(CudaCandidateRuntime::submission_stats)
    }

    /// Return the committed target/MTP token counters for lifecycle verifiers.
    pub fn session_token_counters(&self) -> Option<(usize, usize)> {
        self.graph
            .as_ref()
            .map(|graph| (graph.target_tokens(), graph.mtp_tokens()))
    }

    /// Verifier-only proof that the compact draft projection returns the
    /// bit-exact corresponding rows of the complete LM head for one identical
    /// MTP state. Both branches are restored on device before returning.
    pub fn verify_gathered_mtp(&mut self, token: u32) -> Result<CudaGatheredMtpVerification> {
        self.validate_loaded_decode(&CancellationToken::default())?;
        if token as usize >= TOKENIZER_VOCAB_SIZE {
            return Err(EngineError::Shape(
                "CUDA gathered-head verifier token exceeds vocabulary".into(),
            ));
        }
        let runtime = self.runtime.as_ref().expect("validated CUDA runtime");
        let graph = self.graph.as_mut().expect("validated CUDA graph");
        let position = graph.target_tokens();

        graph.begin_speculative_branch(runtime)?;
        let full_result = (|| {
            let view =
                graph.dispatch_mtp_draft_device(runtime, &self.config, token as usize, position)?;
            read_valid_logits(runtime, view)
        })();
        let full_restore = graph.restore_speculative_branch(runtime);
        let full = match (full_result, full_restore) {
            (Ok(full), Ok(())) => full,
            (Err(error), _) | (Ok(_), Err(error)) => return Err(error),
        };

        graph.begin_speculative_branch(runtime)?;
        let restricted_result = (|| {
            let view = graph.dispatch_mtp_restricted_draft_device(
                runtime,
                &self.config,
                token as usize,
                position,
            )?;
            read_restricted_logits(runtime, view, self.mtp_draft_token_ids.len())
        })();
        let restricted_restore = graph.restore_speculative_branch(runtime);
        let restricted = match (restricted_result, restricted_restore) {
            (Ok(restricted), Ok(())) => restricted,
            (Err(error), _) | (Ok(_), Err(error)) => return Err(error),
        };

        let mut mismatched_rows = 0_usize;
        let mut maximum_absolute_error = 0.0_f32;
        for (&token_id, &compact) in self.mtp_draft_token_ids.iter().zip(&restricted) {
            let complete = full[token_id as usize];
            if complete.to_bits() != compact.to_bits() {
                mismatched_rows += 1;
            }
            maximum_absolute_error = maximum_absolute_error.max((complete - compact).abs());
        }
        Ok(CudaGatheredMtpVerification {
            rows: restricted.len(),
            mismatched_rows,
            maximum_absolute_error,
        })
    }
}

impl ThreadedCudaModelExecutor {
    pub fn new_sm86(cubin: &[u8], device: i32) -> Result<Self> {
        Self::new_sm86_with_output_mode(cubin, device, CudaMtpOutputMode::CompactGreedy)
    }

    /// Hardware-verifier constructor retaining complete vocabulary readbacks
    /// for numerical hashes. The server path uses [`Self::new_sm86`] and
    /// receives only compact, target-verified MTP decisions.
    pub fn new_sm86_with_full_verifier_logits(cubin: &[u8], device: i32) -> Result<Self> {
        Self::new_sm86_with_output_mode(cubin, device, CudaMtpOutputMode::FullVerifierLogits)
    }

    fn new_sm86_with_output_mode(
        cubin: &[u8],
        device: i32,
        mtp_output_mode: CudaMtpOutputMode,
    ) -> Result<Self> {
        let module = cubin.to_vec();
        let (sender, receiver) = mpsc::channel();
        let (initialized_tx, initialized_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("qwen38-cuda-sm86-executor".into())
            .spawn(move || {
                let initialized = match mtp_output_mode {
                    CudaMtpOutputMode::CompactGreedy => {
                        CudaModelExecutor::new_sm86(&module, device)
                    }
                    CudaMtpOutputMode::FullVerifierLogits => {
                        CudaModelExecutor::new_sm86_with_full_verifier_logits(&module, device)
                    }
                };
                match initialized {
                    Ok(executor) => {
                        if initialized_tx.send(Ok(())).is_ok() {
                            run_cuda_worker(executor, receiver);
                        }
                    }
                    Err(error) => {
                        let _ = initialized_tx.send(Err(error));
                    }
                }
            })
            .map_err(|error| {
                EngineError::InvalidState(format!("failed to start CUDA executor worker: {error}"))
            })?;
        match initialized_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: Some(sender),
                worker: Some(worker),
                allocations: AllocationSnapshot::default(),
                compact_greedy_mtp_verification: mtp_output_mode
                    == CudaMtpOutputMode::CompactGreedy,
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(error) => {
                let _ = worker.join();
                Err(EngineError::InvalidState(format!(
                    "CUDA executor worker exited during initialization: {error}"
                )))
            }
        }
    }

    fn request<T>(
        &self,
        operation: &'static str,
        command: impl FnOnce(SyncSender<Result<T>>) -> CudaWorkerCommand,
    ) -> Result<T> {
        let sender = self.sender.as_ref().ok_or_else(|| {
            EngineError::InvalidState(format!(
                "CUDA executor {operation} requested after shutdown"
            ))
        })?;
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        sender.send(command(reply_tx)).map_err(|_| {
            EngineError::InvalidState(format!(
                "CUDA executor worker is unavailable during {operation}"
            ))
        })?;
        reply_rx.recv().map_err(|_| {
            EngineError::InvalidState(format!(
                "CUDA executor worker returned no result for {operation}"
            ))
        })?
    }

    fn refresh_allocations(&mut self) -> Result<()> {
        self.allocations = self.request("allocation query", |reply| {
            CudaWorkerCommand::Allocations { reply }
        })?;
        Ok(())
    }

    pub fn submission_stats(&self) -> Result<CudaSubmissionStats> {
        self.request("submission statistics", |reply| {
            CudaWorkerCommand::SubmissionStats { reply }
        })
    }

    pub fn session_token_counters(&self) -> Result<(usize, usize)> {
        self.request("session token counters", |reply| {
            CudaWorkerCommand::SessionTokenCounters { reply }
        })
    }

    pub fn verify_gathered_mtp(&self, token: u32) -> Result<CudaGatheredMtpVerification> {
        self.request("gathered MTP verification", |reply| {
            CudaWorkerCommand::VerifyGatheredMtp { token, reply }
        })
    }

    fn shutdown(&mut self) -> Result<()> {
        let Some(sender) = self.sender.take() else {
            return Ok(());
        };
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let requested = sender
            .send(CudaWorkerCommand::Shutdown { reply: reply_tx })
            .is_ok();
        drop(sender);
        let result = if requested {
            reply_rx.recv().map_err(|_| {
                EngineError::InvalidState("CUDA executor worker returned no shutdown result".into())
            })?
        } else {
            Err(EngineError::InvalidState(
                "CUDA executor worker exited before shutdown".into(),
            ))
        };
        let joined = self
            .worker
            .take()
            .expect("live CUDA sender has a worker")
            .join();
        self.allocations = AllocationSnapshot::default();
        if joined.is_err() {
            return Err(EngineError::InvalidState(
                "CUDA executor worker panicked during shutdown".into(),
            ));
        }
        result
    }
}

impl Drop for ThreadedCudaModelExecutor {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn run_cuda_worker(mut executor: CudaModelExecutor, receiver: mpsc::Receiver<CudaWorkerCommand>) {
    while let Ok(command) = receiver.recv() {
        match command {
            CudaWorkerCommand::Load {
                artifact,
                profile,
                mtp_draft_token_ids,
                reply,
            } => {
                let _ = reply.send(executor.load(
                    artifact.as_ref(),
                    profile.as_ref(),
                    &mtp_draft_token_ids,
                ));
            }
            CudaWorkerCommand::Warmup { reply } => {
                let _ = reply.send(executor.warmup());
            }
            CudaWorkerCommand::Prefill {
                tokens,
                mtp_enabled,
                cancellation,
                reply,
            } => {
                let _ = reply.send(executor.prefill(&tokens, mtp_enabled, &cancellation));
            }
            CudaWorkerCommand::Decode {
                token,
                mtp_enabled,
                cancellation,
                reply,
            } => {
                let _ = reply.send(executor.decode(token, mtp_enabled, &cancellation));
            }
            CudaWorkerCommand::SelectTarget {
                sampling,
                draw,
                reply,
            } => {
                let _ = reply.send(executor.select_target_token(sampling, draw));
            }
            CudaWorkerCommand::CommitSpeculative {
                accepted_drafts,
                cancellation,
                reply,
            } => {
                let _ = reply.send(executor.commit_speculative(accepted_drafts, &cancellation));
            }
            CudaWorkerCommand::ResetSession { reply } => {
                let _ = reply.send(executor.reset_session());
            }
            CudaWorkerCommand::Unload { reply } => {
                let _ = reply.send(executor.unload());
            }
            CudaWorkerCommand::Allocations { reply } => {
                let _ = reply.send(Ok(executor.allocations()));
            }
            CudaWorkerCommand::SubmissionStats { reply } => {
                let result = executor.submission_stats().ok_or_else(|| {
                    EngineError::InvalidState("CUDA runtime is not available".into())
                });
                let _ = reply.send(result);
            }
            CudaWorkerCommand::SessionTokenCounters { reply } => {
                let result = executor
                    .session_token_counters()
                    .ok_or_else(|| EngineError::InvalidState("CUDA graph is not loaded".into()));
                let _ = reply.send(result);
            }
            CudaWorkerCommand::VerifyGatheredMtp { token, reply } => {
                let _ = reply.send(executor.verify_gathered_mtp(token));
            }
            CudaWorkerCommand::Shutdown { reply } => {
                let reset = executor.reset_session();
                let unload = executor.unload();
                let _ = reply.send(reset.and(unload));
                break;
            }
        }
    }
}

impl ModelExecutor for CudaModelExecutor {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn hardware_profile(&self) -> &str {
        CUDA_SM86_EXECUTOR_PROFILE
    }

    fn promotion_state(&self) -> PromotionState {
        PromotionState::Verifier
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            vocab_size: TOKENIZER_VOCAB_SIZE,
            maximum_context_tokens: self.config.max_position_embeddings as u64,
            mtp: self.config.mtp_num_hidden_layers == 1,
            maximum_draft_tokens: MAXIMUM_CHAINED_MTP_DRAFTS as u32,
            compact_greedy_mtp_verification: self.mtp_output_mode
                == CudaMtpOutputMode::CompactGreedy,
            resident_target_selection: self.mtp_output_mode == CudaMtpOutputMode::CompactGreedy,
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
        if self.graph.is_some()
            || self.argmax.is_some()
            || self.target_sampler.is_some()
            || self.free_bytes_before_graph.is_some()
            || self.pending_speculative.is_some()
            || self.warmed
        {
            return Err(EngineError::InvalidState(
                "CUDA executor is already loaded".into(),
            ));
        }
        if profile.linear_state_dtype != LinearStateDType::F16
            || profile.mtp_draft_tokens == 0
            || profile.mtp_draft_tokens as usize > MAXIMUM_CHAINED_MTP_DRAFTS
            || profile.speculative_state_strategy != SpeculativeStateStrategy::ReplayOnReject
            || profile.speculative_linear_state_bytes_per_session
                != profile.linear_state_bytes_per_session
        {
            return Err(EngineError::UnsupportedOperation {
                backend: "cuda",
                operation: "executor memory profile",
                reason: "SM86 requires FP16 linear state and one replay-on-reject checkpoint"
                    .into(),
            });
        }
        let admitted_context = usize::try_from(profile.context_tokens)
            .map_err(|_| EngineError::MemoryBudget("CUDA context exceeds usize".into()))?;
        if admitted_context == 0 || admitted_context > self.config.max_position_embeddings {
            return Err(EngineError::MemoryBudget(
                "CUDA memory profile context exceeds model capacity".into(),
            ));
        }
        if !self.canonical_draft_vocabulary(mtp_draft_token_ids) {
            return Err(EngineError::InvalidArtifact(
                "CUDA executor received a noncanonical MTP draft vocabulary".into(),
            ));
        }

        let runtime = self.runtime()?;
        let (free_before_graph, _) = runtime.memory_info()?;
        let graph = PreparedCudaProjectionGraph::prepare(
            runtime,
            artifact,
            &self.config,
            admitted_context,
            Some(mtp_draft_token_ids),
        )?;
        let argmax = runtime.prepare_argmax_f32()?;
        let target_sampler = runtime.prepare_topk_topp_sampler(TOKENIZER_VOCAB_SIZE)?;
        let expected_checkpoint = profile
            .speculative_linear_state_bytes_per_session
            .checked_add(
                u64::try_from(self.config.hidden_size * std::mem::size_of::<f32>()).map_err(
                    |_| EngineError::MemoryBudget("CUDA hidden checkpoint exceeds u64".into()),
                )?,
            )
            .ok_or_else(|| EngineError::MemoryBudget("CUDA checkpoint bytes overflow".into()))?;
        if graph.speculative_checkpoint_bytes() != expected_checkpoint {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA graph checkpoint is {} bytes, profile admits {expected_checkpoint}",
                graph.speculative_checkpoint_bytes()
            )));
        }
        let selection_bytes = u64::try_from(
            argmax
                .resident_bytes()
                .checked_add(target_sampler.resident_bytes())
                .ok_or_else(|| {
                    EngineError::MemoryBudget("CUDA target selection bytes overflow".into())
                })?,
        )
        .map_err(|_| EngineError::MemoryBudget("CUDA target selection bytes exceed u64".into()))?;
        let graph_bytes = graph
            .graph_bytes()
            .checked_add(selection_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA graph bytes overflow".into()))?;
        self.allocations = AllocationSnapshot {
            model_bytes: graph.model_bytes(),
            graph_bytes,
            session_bytes: graph.session_bytes(),
            ..AllocationSnapshot::default()
        };
        self.graph = Some(graph);
        self.argmax = Some(argmax);
        self.target_sampler = Some(target_sampler);
        self.mtp_draft_token_ids = mtp_draft_token_ids.to_vec();
        self.admitted_context = admitted_context;
        self.admitted_draft_tokens = profile.mtp_draft_tokens as usize;
        self.free_bytes_before_graph = Some(free_before_graph);
        Ok(())
    }

    fn warmup(&mut self) -> Result<()> {
        if self.graph.is_none() || self.runtime.is_none() {
            return Err(EngineError::InvalidState(
                "CUDA executor is not loaded".into(),
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
        self.validate_loaded_decode(cancellation)?;
        if tokens.is_empty() || tokens.len() > self.admitted_context {
            return Err(EngineError::MemoryBudget(
                "CUDA prefill requires an admitted non-empty token sequence".into(),
            ));
        }
        if tokens
            .iter()
            .any(|token| (*token as usize) >= TOKENIZER_VOCAB_SIZE)
        {
            return Err(EngineError::Shape(
                "CUDA prefill token exceeds tokenizer vocabulary".into(),
            ));
        }
        if mtp_enabled && self.admitted_draft_tokens == 0 {
            return Err(EngineError::InvalidState(
                "CUDA prefill enabled MTP without admitted draft state".into(),
            ));
        }
        self.graph
            .as_mut()
            .expect("validated CUDA graph")
            .reset_session()?;
        let mut target_logits = Vec::new();
        let runtime = self.runtime.as_ref().expect("validated CUDA runtime");
        let graph = self.graph.as_mut().expect("validated CUDA graph");
        let chunk_tokens = graph.prefill_workspaces().max_chunk_tokens();
        for chunk in tokens.chunks(chunk_tokens) {
            if cancellation.is_cancelled() {
                graph.reset_session()?;
                return Err(EngineError::Cancelled);
            }
            let start_position = graph.target_tokens();
            let final_chunk = start_position + chunk.len() == tokens.len();
            if final_chunk {
                let view = if mtp_enabled {
                    graph.dispatch_target_prefill_chunk_with_mtp_device(
                        runtime,
                        &self.config,
                        chunk,
                        start_position,
                    )?
                } else {
                    graph.dispatch_target_prefill_chunk_without_mtp_device(
                        runtime,
                        &self.config,
                        chunk,
                        start_position,
                    )?
                };
                target_logits = match self.mtp_output_mode {
                    CudaMtpOutputMode::CompactGreedy => Vec::new(),
                    CudaMtpOutputMode::FullVerifierLogits => read_valid_logits(runtime, view)?,
                };
            } else if mtp_enabled {
                graph.dispatch_target_prefill_state_with_mtp_device(
                    runtime,
                    &self.config,
                    chunk,
                    start_position,
                )?;
            } else {
                graph.dispatch_target_prefill_state_without_mtp_device(
                    runtime,
                    &self.config,
                    chunk,
                    start_position,
                )?;
            }
        }
        Ok(ExecutorStep {
            target_logits,
            draft_logits: Vec::new(),
            target_verification_logits: Vec::new(),
            bonus_logits: None,
            compact_greedy_mtp: None,
        })
    }

    fn decode(
        &mut self,
        token: u32,
        mtp_enabled: bool,
        cancellation: &CancellationToken,
    ) -> Result<ExecutorStep> {
        self.validate_loaded_decode(cancellation)?;
        if token as usize >= TOKENIZER_VOCAB_SIZE {
            return Err(EngineError::Shape(
                "CUDA decode token exceeds vocabulary".into(),
            ));
        }
        let runtime = self.runtime.as_ref().expect("validated CUDA runtime");
        let argmax = self.argmax.as_ref().expect("validated CUDA argmax");
        let graph = self.graph.as_mut().expect("validated CUDA graph");
        if graph.target_tokens() >= self.admitted_context {
            return Err(EngineError::MemoryBudget(
                "CUDA decode reached admitted context capacity".into(),
            ));
        }
        let speculative_draft_count = if mtp_enabled {
            let remaining_after_target = self
                .admitted_context
                .saturating_sub(graph.target_tokens().saturating_add(1));
            let count = self.admitted_draft_tokens.min(remaining_after_target);
            if count == 0 {
                return Err(EngineError::MemoryBudget(
                    "CUDA MTP decode has no admitted target slot for a draft".into(),
                ));
            }
            count
        } else {
            0
        };
        let first_draft = if mtp_enabled {
            if self.admitted_draft_tokens == 0 {
                return Err(EngineError::InvalidState(
                    "CUDA decode enabled MTP without admitted draft state".into(),
                ));
            }
            let absolute_position = graph.target_tokens();
            let view = graph.dispatch_mtp_restricted_draft_device(
                runtime,
                &self.config,
                token as usize,
                absolute_position,
            )?;
            Some(match self.mtp_output_mode {
                CudaMtpOutputMode::CompactGreedy => (
                    None,
                    select_device_restricted_draft(
                        runtime,
                        argmax,
                        view,
                        &self.mtp_draft_token_ids,
                    )?,
                ),
                CudaMtpOutputMode::FullVerifierLogits => {
                    let (logits, token) = read_device_restricted_draft(
                        runtime,
                        argmax,
                        view,
                        &self.mtp_draft_token_ids,
                    )?;
                    (Some(logits), token)
                }
            })
        } else {
            None
        };
        let position = graph.target_tokens();
        let target_view =
            graph.dispatch_target_token_device(runtime, &self.config, token as usize, position)?;
        let compact_target_token = (self.mtp_output_mode == CudaMtpOutputMode::CompactGreedy)
            .then(|| runtime.dispatch_argmax_f32_device(argmax, target_view))
            .transpose()?;
        let target_logits = match self.mtp_output_mode {
            CudaMtpOutputMode::CompactGreedy => Vec::new(),
            CudaMtpOutputMode::FullVerifierLogits => read_valid_logits(runtime, target_view)?,
        };
        if !mtp_enabled {
            return Ok(ExecutorStep {
                target_logits,
                draft_logits: Vec::new(),
                target_verification_logits: Vec::new(),
                bonus_logits: None,
                compact_greedy_mtp: None,
            });
        }

        graph.begin_speculative_branch(runtime)?;
        if self.mtp_output_mode == CudaMtpOutputMode::CompactGreedy {
            let (_, mut current_draft) = first_draft.ok_or_else(|| {
                EngineError::InvalidState("CUDA MTP draft chain ended before verification".into())
            })?;
            let mut current_target = compact_target_token.ok_or_else(|| {
                EngineError::InvalidState("CUDA compact MTP target selection is missing".into())
            })?;
            let mut candidate_tokens = Vec::with_capacity(speculative_draft_count);
            let mut target_tokens = Vec::with_capacity(speculative_draft_count);
            for depth in 0..speculative_draft_count {
                if cancellation.is_cancelled() {
                    return Err(EngineError::Cancelled);
                }
                let candidate = current_draft;
                candidate_tokens.push(candidate);
                target_tokens.push(current_target);
                if depth + 1 < speculative_draft_count {
                    let absolute_position = graph.target_tokens();
                    let next_draft_view = graph.dispatch_mtp_restricted_draft_device(
                        runtime,
                        &self.config,
                        candidate as usize,
                        absolute_position,
                    )?;
                    current_draft = select_device_restricted_draft(
                        runtime,
                        argmax,
                        next_draft_view,
                        &self.mtp_draft_token_ids,
                    )?;
                }
                let candidate_position = graph.target_tokens();
                let next_target_view = graph.dispatch_target_token_device(
                    runtime,
                    &self.config,
                    candidate as usize,
                    candidate_position,
                )?;
                current_target = runtime.dispatch_argmax_f32_device(argmax, next_target_view)?;
            }
            self.pending_speculative = Some(PendingCudaSpeculativeBranch {
                candidate_tokens: candidate_tokens.clone(),
            });
            return Ok(ExecutorStep {
                target_logits,
                draft_logits: Vec::new(),
                target_verification_logits: Vec::new(),
                bonus_logits: None,
                compact_greedy_mtp: Some(GreedyMtpVerification {
                    draft_tokens: candidate_tokens,
                    target_tokens,
                    bonus_token: current_target,
                }),
            });
        }

        let mut draft_logits = Vec::with_capacity(speculative_draft_count);
        let mut target_verification_logits = Vec::with_capacity(speculative_draft_count);
        let mut candidate_tokens = Vec::with_capacity(speculative_draft_count);
        let mut current_draft = first_draft;
        let mut current_target = target_logits.clone();
        for depth in 0..speculative_draft_count {
            if cancellation.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            let (draft, candidate) = current_draft.take().ok_or_else(|| {
                EngineError::InvalidState("CUDA MTP draft chain ended early".into())
            })?;
            draft_logits.push(DraftDistribution::Restricted {
                token_ids: self.mtp_draft_token_ids.clone(),
                logits: draft.ok_or_else(|| {
                    EngineError::InvalidState("CUDA verifier MTP logits are missing".into())
                })?,
            });
            target_verification_logits.push(current_target);
            candidate_tokens.push(candidate);

            if depth + 1 < speculative_draft_count {
                let absolute_position = graph.target_tokens();
                let next_draft_view = graph.dispatch_mtp_restricted_draft_device(
                    runtime,
                    &self.config,
                    candidate as usize,
                    absolute_position,
                )?;
                let (logits, token) = read_device_restricted_draft(
                    runtime,
                    argmax,
                    next_draft_view,
                    &self.mtp_draft_token_ids,
                )?;
                current_draft = Some((Some(logits), token));
            }
            let candidate_position = graph.target_tokens();
            let next_target_view = graph.dispatch_target_token_device(
                runtime,
                &self.config,
                candidate as usize,
                candidate_position,
            )?;
            current_target = read_valid_logits(runtime, next_target_view)?;
        }
        self.pending_speculative = Some(PendingCudaSpeculativeBranch { candidate_tokens });
        Ok(ExecutorStep {
            target_logits,
            draft_logits,
            target_verification_logits,
            bonus_logits: Some(current_target),
            compact_greedy_mtp: None,
        })
    }

    fn select_target_token(&mut self, sampling: SamplerConfig, draw: f32) -> Result<Option<u32>> {
        let _ = Sampler::new(sampling)?;
        if !self.warmed || self.pending_speculative.is_some() {
            return Err(EngineError::InvalidState(
                "CUDA target selection requires a warm committed target distribution".into(),
            ));
        }
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            EngineError::InvalidState("CUDA target selection has no runtime".into())
        })?;
        let graph = self.graph.as_ref().ok_or_else(|| {
            EngineError::InvalidState("CUDA target selection has no graph".into())
        })?;
        let target_logits = graph
            .target_logits_device()?
            .slice(0, TOKENIZER_VOCAB_SIZE)?;
        let token = if sampling.temperature == 0.0 {
            let argmax = self.argmax.as_ref().ok_or_else(|| {
                EngineError::InvalidState("CUDA target selection has no argmax".into())
            })?;
            runtime.dispatch_argmax_f32_device(argmax, target_logits)?
        } else {
            let sampler = self.target_sampler.as_ref().ok_or_else(|| {
                EngineError::InvalidState("CUDA target selection has no sampler".into())
            })?;
            runtime
                .dispatch_topk_topp_sample_f32_device(sampler, target_logits, sampling, draw)?
                .token
        };
        Ok(Some(token))
    }

    fn commit_speculative(
        &mut self,
        accepted_drafts: u32,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        let branch = self.pending_speculative.take().ok_or_else(|| {
            EngineError::InvalidState("CUDA executor has no pending speculative branch".into())
        })?;
        let accepted = usize::try_from(accepted_drafts)
            .map_err(|_| EngineError::InvalidState("CUDA accepted depth exceeds usize".into()))?;
        if accepted > branch.candidate_tokens.len() {
            return Err(EngineError::InvalidState(
                "CUDA accepted depth exceeds pending candidate block".into(),
            ));
        }
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            EngineError::InvalidState("CUDA speculative commit has no runtime".into())
        })?;
        let graph = self.graph.as_mut().ok_or_else(|| {
            EngineError::InvalidState("CUDA speculative commit has no graph".into())
        })?;
        graph.restore_speculative_branch(runtime)?;
        for candidate in branch.candidate_tokens.into_iter().take(accepted) {
            if cancellation.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            let absolute_position = graph.target_tokens();
            let _ = graph.dispatch_mtp_draft_device(
                runtime,
                &self.config,
                candidate as usize,
                absolute_position,
            )?;
            let target_position = graph.target_tokens();
            let _ = graph.dispatch_target_token_device(
                runtime,
                &self.config,
                candidate as usize,
                target_position,
            )?;
        }
        Ok(())
    }

    fn reset_session(&mut self) -> Result<()> {
        self.pending_speculative = None;
        if let Some(graph) = self.graph.as_mut() {
            graph.reset_session()?;
        }
        Ok(())
    }

    fn unload(&mut self) -> Result<()> {
        self.pending_speculative = None;
        self.warmed = false;
        self.mtp_draft_token_ids.clear();
        self.admitted_context = 0;
        self.admitted_draft_tokens = 0;
        let graph = self.graph.take();
        drop(graph);
        let argmax = self.argmax.take();
        drop(argmax);
        let target_sampler = self.target_sampler.take();
        drop(target_sampler);
        let expected_free = self.free_bytes_before_graph.take();
        let observed_free = self
            .runtime
            .as_ref()
            .map(CudaCandidateRuntime::memory_info)
            .transpose();
        self.runtime.take();
        self.allocations = AllocationSnapshot::default();
        let observed_free = observed_free?.map(|(free, _)| free);
        if let (Some(expected), Some(observed)) = (expected_free, observed_free) {
            if observed != expected {
                return Err(EngineError::MemoryBudget(format!(
                    "CUDA executor retained {} bytes after unload",
                    expected.saturating_sub(observed)
                )));
            }
        }
        Ok(())
    }

    fn allocations(&self) -> AllocationSnapshot {
        self.allocations
    }
}

impl ModelExecutor for ThreadedCudaModelExecutor {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn hardware_profile(&self) -> &str {
        CUDA_SM86_EXECUTOR_PROFILE
    }

    fn promotion_state(&self) -> PromotionState {
        PromotionState::Verifier
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            vocab_size: TOKENIZER_VOCAB_SIZE,
            maximum_context_tokens: Qwen38Config::default().max_position_embeddings as u64,
            mtp: Qwen38Config::default().mtp_num_hidden_layers == 1,
            maximum_draft_tokens: MAXIMUM_CHAINED_MTP_DRAFTS as u32,
            compact_greedy_mtp_verification: self.compact_greedy_mtp_verification,
            resident_target_selection: self.compact_greedy_mtp_verification,
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
        self.request("load", |reply| CudaWorkerCommand::Load {
            artifact: Box::new(artifact.clone()),
            profile: Box::new(profile.clone()),
            mtp_draft_token_ids: mtp_draft_token_ids.to_vec(),
            reply,
        })?;
        self.refresh_allocations()
    }

    fn warmup(&mut self) -> Result<()> {
        self.request("warmup", |reply| CudaWorkerCommand::Warmup { reply })
    }

    fn prefill(
        &mut self,
        tokens: &[u32],
        mtp_enabled: bool,
        cancellation: &CancellationToken,
    ) -> Result<ExecutorStep> {
        self.request("prefill", |reply| CudaWorkerCommand::Prefill {
            tokens: tokens.to_vec(),
            mtp_enabled,
            cancellation: cancellation.clone(),
            reply,
        })
    }

    fn decode(
        &mut self,
        token: u32,
        mtp_enabled: bool,
        cancellation: &CancellationToken,
    ) -> Result<ExecutorStep> {
        self.request("decode", |reply| CudaWorkerCommand::Decode {
            token,
            mtp_enabled,
            cancellation: cancellation.clone(),
            reply,
        })
    }

    fn select_target_token(&mut self, sampling: SamplerConfig, draw: f32) -> Result<Option<u32>> {
        self.request("target selection", |reply| {
            CudaWorkerCommand::SelectTarget {
                sampling,
                draw,
                reply,
            }
        })
    }

    fn commit_speculative(
        &mut self,
        accepted_drafts: u32,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        self.request("speculative commit", |reply| {
            CudaWorkerCommand::CommitSpeculative {
                accepted_drafts,
                cancellation: cancellation.clone(),
                reply,
            }
        })
    }

    fn reset_session(&mut self) -> Result<()> {
        self.request("session reset", |reply| CudaWorkerCommand::ResetSession {
            reply,
        })
    }

    fn unload(&mut self) -> Result<()> {
        let result = self.request("unload", |reply| CudaWorkerCommand::Unload { reply });
        let allocation_result = self.refresh_allocations();
        result.and(allocation_result)
    }

    fn allocations(&self) -> AllocationSnapshot {
        self.allocations
    }
}

fn greedy_restricted_token(token_ids: &[u32], logits: &[f32]) -> Result<u32> {
    if token_ids.len() != logits.len() || token_ids.is_empty() {
        return Err(EngineError::InvalidArtifact(
            "CUDA restricted token IDs and logits differ".into(),
        ));
    }
    let index = logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .ok_or_else(|| EngineError::InvalidArtifact("CUDA logits are empty".into()))?;
    Ok(token_ids[index])
}

fn read_restricted_logits(
    runtime: &CudaCandidateRuntime,
    view: crate::backend::cuda_runtime::CudaDeviceF32View<'_>,
    expected: usize,
) -> Result<Vec<f32>> {
    let logits = runtime.verifier_read_f32_device(view)?;
    if logits.len() != expected {
        return Err(EngineError::InvalidArtifact(format!(
            "CUDA restricted logits contain {} rows, expected {expected}",
            logits.len()
        )));
    }
    Ok(logits)
}

fn read_device_restricted_draft(
    runtime: &CudaCandidateRuntime,
    argmax: &PreparedCudaArgmax,
    view: CudaDeviceF32View<'_>,
    token_ids: &[u32],
) -> Result<(Vec<f32>, u32)> {
    let local_index = usize::try_from(runtime.dispatch_argmax_f32_device(argmax, view)?)
        .map_err(|_| EngineError::Shape("CUDA argmax index exceeds usize".into()))?;
    let logits = read_restricted_logits(runtime, view, token_ids.len())?;
    let token = token_ids.get(local_index).copied().ok_or_else(|| {
        EngineError::InvalidArtifact("CUDA argmax index exceeds restricted vocabulary".into())
    })?;
    let host_token = greedy_restricted_token(token_ids, &logits)?;
    if token != host_token {
        return Err(EngineError::InvalidState(format!(
            "CUDA device argmax selected {token}, host oracle selected {host_token}"
        )));
    }
    Ok((logits, token))
}

fn select_device_restricted_draft(
    runtime: &CudaCandidateRuntime,
    argmax: &PreparedCudaArgmax,
    view: CudaDeviceF32View<'_>,
    token_ids: &[u32],
) -> Result<u32> {
    let local_index = usize::try_from(runtime.dispatch_argmax_f32_device(argmax, view)?)
        .map_err(|_| EngineError::Shape("CUDA argmax index exceeds usize".into()))?;
    token_ids.get(local_index).copied().ok_or_else(|| {
        EngineError::InvalidArtifact("CUDA argmax index exceeds restricted vocabulary".into())
    })
}

fn read_valid_logits(
    runtime: &CudaCandidateRuntime,
    view: crate::backend::cuda_runtime::CudaDeviceF32View<'_>,
) -> Result<Vec<f32>> {
    let mut logits = runtime.verifier_read_f32_device(view)?;
    if logits.len() < TOKENIZER_VOCAB_SIZE {
        return Err(EngineError::InvalidArtifact(format!(
            "CUDA logits contain {} rows, tokenizer requires {TOKENIZER_VOCAB_SIZE}",
            logits.len()
        )));
    }
    logits.truncate(TOKENIZER_VOCAB_SIZE);
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn threaded_cuda_adapter_is_send_without_moving_the_driver_context() {
        assert_send::<ThreadedCudaModelExecutor>();
    }

    #[test]
    fn restricted_greedy_maps_compact_logits_back_to_global_tokens() {
        let token_ids = [7_u32, 101, 40_000, 151_664];
        assert_eq!(
            greedy_restricted_token(&token_ids, &[-1.0, 3.5, 2.0, 3.0]).unwrap(),
            101
        );
        assert!(greedy_restricted_token(&token_ids, &[1.0]).is_err());
        assert!(greedy_restricted_token(&[], &[]).is_err());
    }
}
