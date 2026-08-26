//! Model-specific artifact graph composition.
//!
//! This module wires the frozen Qwen tensor names to backend operations. It is
//! intentionally not a generic neural-network runtime. The mmap-backed
//! correctness executor covers the complete target graph and native one-layer
//! MTP graph. Production promotion still requires fused token-mixer kernels,
//! full-artifact golden evidence, and same-device roofline measurements.

use crate::backend::cpu::CpuBackend;
use crate::backend::{
    Activation, Backend, BackendKind, ExecutionPolicy, PromotionState, RecoveredRowMatVec,
};
use crate::config::LayerKind;
use crate::engine::{
    AllocationSnapshot, CancellationToken, DraftDistribution, ExecutorCapabilities, ExecutorStep,
    ModelExecutor,
};
use crate::kv_cache::PagedKvCache;
use crate::loader::ModelArtifact;
use crate::reference::{
    apply_partial_rope, causal_conv_silu_update, grouped_query_attention,
    recurrent_gated_delta_step, rms_norm_1p_weight, rms_norm_gated, sigmoid_gate, swiglu,
};
use crate::release::MemoryProfile;
use crate::{EngineError, Qwen38Config, Result};

const MAXIMUM_CHAINED_MTP_DRAFTS: usize = 4;

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

#[cfg(test)]
fn greedy_token(logits: &[f32]) -> Result<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(token, _)| token as u32)
        .ok_or_else(|| EngineError::Shape("MTP draft logits are empty".into()))
}

fn repeat_heads(
    values: &[f32],
    heads: usize,
    dimension: usize,
    repeats: usize,
) -> Result<Vec<f32>> {
    if heads == 0 || dimension == 0 || repeats == 0 || values.len() != heads * dimension {
        return Err(EngineError::Shape(
            "linear-attention head repetition shape differs".into(),
        ));
    }
    let mut output = Vec::with_capacity(values.len() * repeats);
    for head in values.chunks_exact(dimension) {
        for _ in 0..repeats {
            output.extend_from_slice(head);
        }
    }
    Ok(output)
}

#[derive(Debug, Clone)]
pub struct FullAttentionState {
    storage: FullAttentionStorage,
    key_value_heads: usize,
    head_dim: usize,
    maximum_tokens: usize,
    tokens: usize,
}

#[derive(Debug, Clone)]
enum FullAttentionStorage {
    /// Small non-Qwen geometries remain useful as an exact test oracle.
    Dense {
        key_heads: Vec<Vec<f32>>,
        value_heads: Vec<Vec<f32>>,
    },
    /// The real Qwen geometry is always stored as mixed Q2/Q4 pages.
    Paged(PagedKvCache),
}

#[derive(Debug, Clone)]
pub struct LinearAttentionState {
    convolution: Vec<f32>,
    recurrent: Vec<f32>,
    channels: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel: usize,
    maximum_tokens: usize,
    tokens: usize,
}

impl LinearAttentionState {
    pub fn new(config: &Qwen38Config, maximum_tokens: usize) -> Result<Self> {
        let key_width = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)
            .ok_or_else(|| EngineError::Shape("linear key width overflows".into()))?;
        let value_width = config
            .linear_num_value_heads
            .checked_mul(config.linear_value_head_dim)
            .ok_or_else(|| EngineError::Shape("linear value width overflows".into()))?;
        let channels = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or_else(|| EngineError::Shape("linear convolution width overflows".into()))?;
        let convolution_values = channels
            .checked_mul(config.linear_conv_kernel_dim)
            .ok_or_else(|| EngineError::Shape("linear convolution state overflows".into()))?;
        let recurrent_values = config
            .linear_num_value_heads
            .checked_mul(config.linear_key_head_dim)
            .and_then(|values| values.checked_mul(config.linear_value_head_dim))
            .ok_or_else(|| EngineError::Shape("linear recurrent state overflows".into()))?;
        if maximum_tokens == 0
            || config.linear_num_key_heads == 0
            || config.linear_num_value_heads == 0
            || !config
                .linear_num_value_heads
                .is_multiple_of(config.linear_num_key_heads)
        {
            return Err(EngineError::Shape(
                "linear-attention state geometry differs".into(),
            ));
        }
        Ok(Self {
            convolution: vec![0.0; convolution_values],
            recurrent: vec![0.0; recurrent_values],
            channels,
            value_heads: config.linear_num_value_heads,
            key_dim: config.linear_key_head_dim,
            value_dim: config.linear_value_head_dim,
            kernel: config.linear_conv_kernel_dim,
            maximum_tokens,
            tokens: 0,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn allocated_bytes(&self) -> usize {
        (self.convolution.len() + self.recurrent.len()) * std::mem::size_of::<f32>()
    }

    pub fn reset(&mut self) {
        self.convolution.fill(0.0);
        self.recurrent.fill(0.0);
        self.tokens = 0;
    }

    fn validate_step(&self, position: usize) -> Result<()> {
        if position != self.tokens || self.tokens >= self.maximum_tokens {
            return Err(EngineError::Shape(
                "linear-attention position exceeds or differs from state".into(),
            ));
        }
        Ok(())
    }
}

impl FullAttentionState {
    pub fn new(key_value_heads: usize, head_dim: usize, maximum_tokens: usize) -> Result<Self> {
        if key_value_heads == 0 || head_dim == 0 || maximum_tokens == 0 {
            return Err(EngineError::Shape(
                "full-attention state dimensions must be non-zero".into(),
            ));
        }
        let component_values = key_value_heads
            .checked_mul(head_dim)
            .ok_or_else(|| EngineError::Shape("full-attention KV width overflows".into()))?;
        let storage = if component_values.is_multiple_of(crate::quant::BLOCK_LEN) {
            FullAttentionStorage::Paged(PagedKvCache::qwen_default(
                maximum_tokens,
                component_values,
            )?)
        } else {
            FullAttentionStorage::Dense {
                key_heads: (0..key_value_heads).map(|_| Vec::new()).collect(),
                value_heads: (0..key_value_heads).map(|_| Vec::new()).collect(),
            }
        };
        Ok(Self {
            storage,
            key_value_heads,
            head_dim,
            maximum_tokens,
            tokens: 0,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn reset(&mut self) {
        match &mut self.storage {
            FullAttentionStorage::Dense {
                key_heads,
                value_heads,
            } => {
                key_heads.iter_mut().for_each(Vec::clear);
                value_heads.iter_mut().for_each(Vec::clear);
            }
            FullAttentionStorage::Paged(cache) => cache.reset(),
        }
        self.tokens = 0;
    }

    pub fn allocated_bytes(&self) -> usize {
        match &self.storage {
            FullAttentionStorage::Dense {
                key_heads,
                value_heads,
            } => key_heads
                .iter()
                .chain(value_heads)
                .map(|head| head.capacity() * std::mem::size_of::<f32>())
                .sum(),
            FullAttentionStorage::Paged(cache) => cache.allocated_bytes(),
        }
    }

    fn append(&mut self, position: usize, key: &[f32], value: &[f32]) -> Result<()> {
        let expected = self
            .key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| EngineError::Shape("full-attention state shape overflows".into()))?;
        if position != self.tokens
            || self.tokens >= self.maximum_tokens
            || key.len() != expected
            || value.len() != expected
        {
            return Err(EngineError::Shape(
                "full-attention append position or shape differs".into(),
            ));
        }
        match &mut self.storage {
            FullAttentionStorage::Dense {
                key_heads,
                value_heads,
            } => {
                for head in 0..key_heads.len() {
                    let start = head * self.head_dim;
                    key_heads[head].extend_from_slice(&key[start..start + self.head_dim]);
                    value_heads[head].extend_from_slice(&value[start..start + self.head_dim]);
                }
            }
            FullAttentionStorage::Paged(cache) => {
                cache.push(key, value)?;
            }
        }
        self.tokens += 1;
        Ok(())
    }

    fn flattened_key(&self) -> Result<Vec<f32>> {
        match &self.storage {
            FullAttentionStorage::Dense { key_heads, .. } => {
                Ok(key_heads.iter().flatten().copied().collect())
            }
            FullAttentionStorage::Paged(cache) => {
                cache.flattened_key(self.key_value_heads, self.head_dim)
            }
        }
    }

    fn flattened_value(&self) -> Result<Vec<f32>> {
        match &self.storage {
            FullAttentionStorage::Dense { value_heads, .. } => {
                Ok(value_heads.iter().flatten().copied().collect())
            }
            FullAttentionStorage::Paged(cache) => {
                cache.flattened_value(self.key_value_heads, self.head_dim)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum DecoderLayerState {
    Linear(LinearAttentionState),
    Full(FullAttentionState),
}

#[derive(Debug, Clone)]
pub struct DecoderState {
    layers: Vec<DecoderLayerState>,
    maximum_tokens: usize,
    position: usize,
}

#[derive(Debug, Clone)]
pub struct MtpState {
    attention: FullAttentionState,
    maximum_tokens: usize,
    tokens: usize,
}

#[derive(Debug)]
struct PendingSpeculativeBranch {
    candidate_tokens: Vec<u32>,
}

impl PendingSpeculativeBranch {
    fn allocated_bytes(&self) -> Result<usize> {
        self.candidate_tokens
            .capacity()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| EngineError::MemoryBudget("CPU MTP candidate bytes overflow".into()))
    }
}

impl MtpState {
    pub fn new(config: &Qwen38Config, maximum_tokens: usize) -> Result<Self> {
        if config.mtp_num_hidden_layers != 1
            || maximum_tokens < 2
            || maximum_tokens > config.max_position_embeddings
        {
            return Err(EngineError::Shape(
                "MTP state requires one layer and at least two admitted tokens".into(),
            ));
        }
        Ok(Self {
            attention: FullAttentionState::new(
                config.num_key_value_heads,
                config.head_dim,
                maximum_tokens - 1,
            )?,
            maximum_tokens,
            tokens: 0,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn allocated_bytes(&self) -> usize {
        self.attention.allocated_bytes()
    }

    pub fn reset(&mut self) {
        self.attention.reset();
        self.tokens = 0;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TargetStep {
    /// The target model's post-final-norm hidden state. This is precisely the
    /// `last_hidden_state` consumed by Transformers' Qwen MTP layer.
    pub final_hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RestrictedDraftStep {
    pub final_hidden: Vec<f32>,
    pub distribution: DraftDistribution,
}

impl DecoderState {
    pub fn new(config: &Qwen38Config, maximum_tokens: usize) -> Result<Self> {
        if maximum_tokens == 0 || maximum_tokens > config.max_position_embeddings {
            return Err(EngineError::Shape(
                "decoder state capacity is zero or exceeds model context".into(),
            ));
        }
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            layers.push(
                match config.layer_kind(layer).ok_or_else(|| {
                    EngineError::Shape(format!("decoder layer {layer} is outside config"))
                })? {
                    LayerKind::LinearAttention => DecoderLayerState::Linear(
                        LinearAttentionState::new(config, maximum_tokens)?,
                    ),
                    LayerKind::FullAttention => DecoderLayerState::Full(FullAttentionState::new(
                        config.num_key_value_heads,
                        config.head_dim,
                        maximum_tokens,
                    )?),
                },
            );
        }
        Ok(Self {
            layers,
            maximum_tokens,
            position: 0,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn allocated_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| match layer {
                DecoderLayerState::Linear(state) => state.allocated_bytes(),
                DecoderLayerState::Full(state) => state.allocated_bytes(),
            })
            .sum()
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            match layer {
                DecoderLayerState::Linear(state) => state.reset(),
                DecoderLayerState::Full(state) => state.reset(),
            }
        }
        self.position = 0;
    }
}

pub struct ArtifactDecoder<'a, B: Backend> {
    artifact: &'a ModelArtifact,
    backend: &'a B,
    rms_epsilon: f32,
}

pub struct CpuCorrectnessExecutor {
    config: Qwen38Config,
    backend: CpuBackend,
    artifact: Option<ModelArtifact>,
    state: Option<DecoderState>,
    mtp_state: Option<MtpState>,
    last_final_hidden: Option<Vec<f32>>,
    pending_speculative: Option<PendingSpeculativeBranch>,
    mtp_draft_token_ids: Vec<u32>,
    admitted_context: usize,
    admitted_draft_tokens: usize,
    warmed: bool,
    allocations: AllocationSnapshot,
}

impl CpuCorrectnessExecutor {
    pub fn scalar(config: Qwen38Config) -> Self {
        Self {
            config,
            backend: CpuBackend::scalar_verifier(),
            artifact: None,
            state: None,
            mtp_state: None,
            last_final_hidden: None,
            pending_speculative: None,
            mtp_draft_token_ids: Vec::new(),
            admitted_context: 0,
            admitted_draft_tokens: 0,
            warmed: false,
            allocations: AllocationSnapshot::default(),
        }
    }

    pub fn detected(config: Qwen38Config) -> Result<Self> {
        Ok(Self {
            config,
            backend: CpuBackend::detect(ExecutionPolicy::Verifier)?,
            artifact: None,
            state: None,
            mtp_state: None,
            last_final_hidden: None,
            pending_speculative: None,
            mtp_draft_token_ids: Vec::new(),
            admitted_context: 0,
            admitted_draft_tokens: 0,
            warmed: false,
            allocations: AllocationSnapshot::default(),
        })
    }

    fn update_session_allocation(&mut self) -> Result<()> {
        let pending_bytes = self
            .pending_speculative
            .as_ref()
            .map(PendingSpeculativeBranch::allocated_bytes)
            .transpose()?
            .unwrap_or(0);
        self.allocations.session_bytes = self
            .state
            .as_ref()
            .map(DecoderState::allocated_bytes)
            .unwrap_or(0)
            .checked_add(
                self.mtp_state
                    .as_ref()
                    .map(MtpState::allocated_bytes)
                    .unwrap_or(0),
            )
            .and_then(|bytes| {
                bytes.checked_add(
                    self.last_final_hidden
                        .as_ref()
                        .map(|hidden| hidden.capacity() * std::mem::size_of::<f32>())
                        .unwrap_or(0),
                )
            })
            .and_then(|bytes| bytes.checked_add(pending_bytes))
            .ok_or_else(|| EngineError::MemoryBudget("CPU decoder state overflows".into()))?
            .try_into()
            .map_err(|_| EngineError::MemoryBudget("CPU decoder state exceeds u64".into()))?;
        Ok(())
    }
}

impl<'a, B: Backend> ArtifactDecoder<'a, B> {
    pub fn new(artifact: &'a ModelArtifact, backend: &'a B, rms_epsilon: f32) -> Result<Self> {
        if !rms_epsilon.is_finite() || rms_epsilon <= 0.0 {
            return Err(EngineError::Shape(
                "decoder RMS epsilon must be finite and positive".into(),
            ));
        }
        Ok(Self {
            artifact,
            backend,
            rms_epsilon,
        })
    }

    pub fn embedding(&self, token_id: u32) -> Result<Vec<f32>> {
        let matrix = self
            .artifact
            .recovered_matrix("model.language_model.embed_tokens.weight")?;
        let row = matrix.row_operation(token_id as usize)?;
        self.backend.recovered_row(&row)
    }

    pub fn projection(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let matrix = self.artifact.recovered_matrix(name)?;
        if input.len() != matrix.matrix.columns {
            return Err(EngineError::Shape(format!(
                "projection {name} received {} values, expected {}",
                input.len(),
                matrix.matrix.columns
            )));
        }
        self.backend
            .fused_matvec(&matrix.operation(input, Activation::Identity)?)
    }

    pub fn projection_fanout(&self, names: &[String], input: &[f32]) -> Result<Vec<Vec<f32>>> {
        if names.is_empty() {
            return Err(EngineError::Shape(
                "projection fan-out requires at least one matrix".into(),
            ));
        }
        let matrices = names
            .iter()
            .map(|name| {
                let matrix = self.artifact.recovered_matrix(name)?;
                if input.len() != matrix.matrix.columns {
                    return Err(EngineError::Shape(format!(
                        "projection {name} received {} values, expected {}",
                        input.len(),
                        matrix.matrix.columns
                    )));
                }
                Ok(matrix)
            })
            .collect::<Result<Vec<_>>>()?;
        let operations = matrices
            .iter()
            .map(|matrix| (*matrix).operation(input, Activation::Identity))
            .collect::<Result<Vec<_>>>()?;
        let outputs = self.backend.fused_matvec_fanout(&operations)?;
        if outputs.len() != names.len() {
            return Err(EngineError::InvalidState(format!(
                "projection fan-out returned {} outputs for {} matrices",
                outputs.len(),
                names.len()
            )));
        }
        Ok(outputs)
    }

    /// Evaluate only release-bound LM-head rows for an MTP proposal. The
    /// complete target LM head remains a separate mandatory operation.
    pub fn restricted_logits_from_final_hidden(
        &self,
        final_hidden: &[f32],
        token_ids: &[u32],
    ) -> Result<DraftDistribution> {
        let matrix = self.artifact.recovered_matrix("lm_head.weight")?;
        if final_hidden.len() != matrix.matrix.columns
            || token_ids.is_empty()
            || token_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || token_ids
                .iter()
                .any(|token| *token as usize >= matrix.matrix.rows)
        {
            return Err(EngineError::Shape(
                "restricted LM-head input or canonical token IDs differ".into(),
            ));
        }
        let s_in = matrix.s_in.as_recovery_scales()?;
        let corrected_input = final_hidden
            .iter()
            .enumerate()
            .map(|(column, value)| Ok(value * s_in.value(column)?))
            .collect::<Result<Vec<f32>>>()?;
        let mut logits = Vec::with_capacity(token_ids.len());
        for token in token_ids {
            let row_index = *token as usize;
            let row = matrix.matrix.row(row_index)?;
            logits.push(self.backend.recovered_row_matvec(&RecoveredRowMatVec {
                dtype: row.dtype,
                weights: row.weights,
                corrected_input: &corrected_input,
                s_out: matrix.s_out.value(row_index)?,
            })?);
        }
        Ok(DraftDistribution::Restricted {
            token_ids: token_ids.to_vec(),
            logits,
        })
    }

    pub fn rms_norm(&self, name: &str, hidden: &[f32]) -> Result<Vec<f32>> {
        let weight = self.artifact.float_tensor(name)?.to_f32_vec()?;
        rms_norm_1p_weight(hidden, 1, hidden.len(), &weight, self.rms_epsilon)
    }

    /// Execute Qwen's post-token-mixer residual MLP for one token:
    /// `x + down(silu(gate(norm(x))) * up(norm(x)))`.
    pub fn decoder_mlp_residual(&self, layer_prefix: &str, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.is_empty() || hidden.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::Shape(
                "decoder MLP hidden state is empty or non-finite".into(),
            ));
        }
        let normalized = self.rms_norm(
            &format!("{layer_prefix}.post_attention_layernorm.weight"),
            hidden,
        )?;
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let mut fanout = self.projection_fanout(
            &[
                format!("{mlp_prefix}.gate_proj.weight"),
                format!("{mlp_prefix}.up_proj.weight"),
            ],
            &normalized,
        )?;
        let up = fanout.pop().expect("two fan-out outputs validated");
        let gate = fanout.pop().expect("two fan-out outputs validated");
        let activated = swiglu(&gate, &up)?;
        let down = self.projection(&format!("{mlp_prefix}.down_proj.weight"), &activated)?;
        if down.len() != hidden.len() {
            return Err(EngineError::Shape(format!(
                "decoder MLP produced {} values, expected {}",
                down.len(),
                hidden.len()
            )));
        }
        Ok(hidden
            .iter()
            .zip(down)
            .map(|(residual, value)| residual + value)
            .collect())
    }

    /// Execute one cached full-attention token mixer and its residual. This is
    /// the exact scalar composition oracle around backend quantized
    /// projections; production backends replace the attention/state internals
    /// with paged fused kernels.
    pub fn full_attention_residual(
        &self,
        layer_prefix: &str,
        hidden: &[f32],
        position: usize,
        state: &mut FullAttentionState,
        config: &Qwen38Config,
    ) -> Result<Vec<f32>> {
        self.full_attention_residual_at(layer_prefix, hidden, position, position, state, config)
    }

    fn full_attention_residual_at(
        &self,
        layer_prefix: &str,
        hidden: &[f32],
        rope_position: usize,
        cache_position: usize,
        state: &mut FullAttentionState,
        config: &Qwen38Config,
    ) -> Result<Vec<f32>> {
        if hidden.len() != config.hidden_size
            || state.key_value_heads != config.num_key_value_heads
            || state.head_dim != config.head_dim
        {
            return Err(EngineError::Shape(
                "full-attention hidden state or cache geometry differs".into(),
            ));
        }
        let normalized =
            self.rms_norm(&format!("{layer_prefix}.input_layernorm.weight"), hidden)?;
        let prefix = format!("{layer_prefix}.self_attn");
        let query_width = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| EngineError::Shape("query width overflows".into()))?;
        let key_value_width = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| EngineError::Shape("KV width overflows".into()))?;
        let mut fanout = self.projection_fanout(
            &[
                format!("{prefix}.q_proj.weight"),
                format!("{prefix}.k_proj.weight"),
                format!("{prefix}.v_proj.weight"),
            ],
            &normalized,
        )?;
        let value = fanout.pop().expect("three fan-out outputs validated");
        let mut key = fanout.pop().expect("three fan-out outputs validated");
        let query_gate = fanout.pop().expect("three fan-out outputs validated");
        if query_gate.len() != query_width * 2 {
            return Err(EngineError::Shape(
                "full-attention query/gate projection shape differs".into(),
            ));
        }
        let mut query = Vec::with_capacity(query_width);
        let mut gate = Vec::with_capacity(query_width);
        for head in query_gate.chunks_exact(config.head_dim * 2) {
            query.extend_from_slice(&head[..config.head_dim]);
            gate.extend_from_slice(&head[config.head_dim..]);
        }
        if key.len() != key_value_width || value.len() != key_value_width {
            return Err(EngineError::Shape(
                "full-attention key/value projection shape differs".into(),
            ));
        }
        let q_norm = self
            .artifact
            .float_tensor(&format!("{prefix}.q_norm.weight"))?
            .to_f32_vec()?;
        let k_norm = self
            .artifact
            .float_tensor(&format!("{prefix}.k_norm.weight"))?
            .to_f32_vec()?;
        query = rms_norm_1p_weight(
            &query,
            config.num_attention_heads,
            config.head_dim,
            &q_norm,
            self.rms_epsilon,
        )?;
        key = rms_norm_1p_weight(
            &key,
            config.num_key_value_heads,
            config.head_dim,
            &k_norm,
            self.rms_epsilon,
        )?;
        apply_partial_rope(
            &mut query,
            &mut key,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rotary_dim,
            rope_position as u64,
            config.rope_theta,
        )?;
        state.append(cache_position, &key, &value)?;
        let key = state.flattened_key()?;
        let value = state.flattened_value()?;
        let mut attention = grouped_query_attention(
            &query,
            &key,
            &value,
            config.num_attention_heads,
            config.num_key_value_heads,
            1,
            state.tokens(),
            config.head_dim,
            cache_position,
        )?;
        sigmoid_gate(&mut attention, &gate)?;
        let projected = self.projection(&format!("{prefix}.o_proj.weight"), &attention)?;
        if projected.len() != hidden.len() {
            return Err(EngineError::Shape(
                "full-attention output projection shape differs".into(),
            ));
        }
        Ok(hidden
            .iter()
            .zip(projected)
            .map(|(residual, value)| residual + value)
            .collect())
    }

    /// Execute one cached Qwen GatedDeltaNet token mixer and residual using
    /// the pinned recurrent rule. Production backends replace the scalar
    /// convolution/recurrent core with fused platform kernels while retaining
    /// this exact tensor order and state transition.
    pub fn linear_attention_residual(
        &self,
        layer_prefix: &str,
        hidden: &[f32],
        position: usize,
        state: &mut LinearAttentionState,
        config: &Qwen38Config,
    ) -> Result<Vec<f32>> {
        state.validate_step(position)?;
        let key_width = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_width = config.linear_num_value_heads * config.linear_value_head_dim;
        let convolution_width = key_width * 2 + value_width;
        if hidden.len() != config.hidden_size
            || state.channels != convolution_width
            || state.value_heads != config.linear_num_value_heads
            || state.key_dim != config.linear_key_head_dim
            || state.value_dim != config.linear_value_head_dim
            || state.kernel != config.linear_conv_kernel_dim
        {
            return Err(EngineError::Shape(
                "linear-attention hidden state or cache geometry differs".into(),
            ));
        }
        let normalized =
            self.rms_norm(&format!("{layer_prefix}.input_layernorm.weight"), hidden)?;
        let prefix = format!("{layer_prefix}.linear_attn");
        let mut fanout = self.projection_fanout(
            &[
                format!("{prefix}.in_proj_qkv.weight"),
                format!("{prefix}.in_proj_z.weight"),
                format!("{prefix}.in_proj_a.weight"),
                format!("{prefix}.in_proj_b.weight"),
            ],
            &normalized,
        )?;
        let b = fanout.pop().expect("four fan-out outputs validated");
        let a = fanout.pop().expect("four fan-out outputs validated");
        let z = fanout.pop().expect("four fan-out outputs validated");
        let mixed_qkv = fanout.pop().expect("four fan-out outputs validated");
        if mixed_qkv.len() != convolution_width
            || z.len() != value_width
            || a.len() != config.linear_num_value_heads
            || b.len() != config.linear_num_value_heads
        {
            return Err(EngineError::Shape(
                "linear-attention projection shape differs".into(),
            ));
        }
        let convolution_weight = self
            .artifact
            .float_tensor(&format!("{prefix}.conv1d.weight"))?
            .to_f32_vec()?;
        let convolved = causal_conv_silu_update(
            &mixed_qkv,
            &mut state.convolution,
            &convolution_weight,
            convolution_width,
            config.linear_conv_kernel_dim,
        )?;
        let query = &convolved[..key_width];
        let key = &convolved[key_width..key_width * 2];
        let value = &convolved[key_width * 2..];
        let repeats = config.linear_num_value_heads / config.linear_num_key_heads;
        let query = repeat_heads(
            query,
            config.linear_num_key_heads,
            config.linear_key_head_dim,
            repeats,
        )?;
        let key = repeat_heads(
            key,
            config.linear_num_key_heads,
            config.linear_key_head_dim,
            repeats,
        )?;
        let beta: Vec<f32> = b.iter().map(|value| 1.0 / (1.0 + (-value).exp())).collect();
        let a_log = self
            .artifact
            .float_tensor(&format!("{prefix}.A_log"))?
            .to_f32_vec()?;
        let dt_bias = self
            .artifact
            .float_tensor(&format!("{prefix}.dt_bias"))?
            .to_f32_vec()?;
        if a_log.len() != config.linear_num_value_heads
            || dt_bias.len() != config.linear_num_value_heads
        {
            return Err(EngineError::Shape(
                "linear-attention decay parameter shape differs".into(),
            ));
        }
        let log_decay: Vec<f32> = a
            .iter()
            .zip(a_log)
            .zip(dt_bias)
            .map(|((a, a_log), dt_bias)| -a_log.exp() * softplus(a + dt_bias))
            .collect();
        let core = recurrent_gated_delta_step(
            &query,
            &key,
            value,
            &log_decay,
            &beta,
            &mut state.recurrent,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        )?;
        let norm_weight = self
            .artifact
            .float_tensor(&format!("{prefix}.norm.weight"))?
            .to_f32_vec()?;
        let core = rms_norm_gated(
            &core,
            &z,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            &norm_weight,
            self.rms_epsilon,
        )?;
        let projected = self.projection(&format!("{prefix}.out_proj.weight"), &core)?;
        if projected.len() != hidden.len() {
            return Err(EngineError::Shape(
                "linear-attention output projection shape differs".into(),
            ));
        }
        state.tokens += 1;
        Ok(hidden
            .iter()
            .zip(projected)
            .map(|(residual, value)| residual + value)
            .collect())
    }

    pub fn final_logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let normalized = self.final_hidden(hidden)?;
        self.logits_from_final_hidden(&normalized)
    }

    pub fn final_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        self.rms_norm("model.language_model.norm.weight", hidden)
    }

    pub fn logits_from_final_hidden(&self, final_hidden: &[f32]) -> Result<Vec<f32>> {
        self.projection("lm_head.weight", final_hidden)
    }

    /// Complete target-model decode for one input token. The method is a
    /// correctness composition over the frozen 64-layer topology; optimized
    /// executors batch/fuse these same transitions per backend.
    pub fn decode_target_step(
        &self,
        token_id: u32,
        state: &mut DecoderState,
        config: &Qwen38Config,
    ) -> Result<TargetStep> {
        if state.layers.len() != config.num_hidden_layers || state.position >= state.maximum_tokens
        {
            return Err(EngineError::Shape(
                "decoder state topology or context capacity differs".into(),
            ));
        }
        let position = state.position;
        let mut hidden = self.embedding(token_id)?;
        if hidden.len() != config.hidden_size {
            return Err(EngineError::Shape(
                "embedding width differs from decoder hidden size".into(),
            ));
        }
        for (layer, layer_state) in state.layers.iter_mut().enumerate() {
            let prefix = format!("model.language_model.layers.{layer}");
            hidden = match (config.layer_kind(layer), layer_state) {
                (Some(LayerKind::LinearAttention), DecoderLayerState::Linear(linear_state)) => self
                    .linear_attention_residual(&prefix, &hidden, position, linear_state, config)?,
                (Some(LayerKind::FullAttention), DecoderLayerState::Full(full_state)) => {
                    self.full_attention_residual(&prefix, &hidden, position, full_state, config)?
                }
                _ => {
                    return Err(EngineError::InvalidState(format!(
                        "decoder layer {layer} state kind differs from config"
                    )))
                }
            };
            hidden = self.decoder_mlp_residual(&prefix, &hidden)?;
        }
        let final_hidden = self.final_hidden(&hidden)?;
        let logits = self.logits_from_final_hidden(&final_hidden)?;
        state.position += 1;
        Ok(TargetStep {
            final_hidden,
            logits,
        })
    }

    pub fn decode_target_token(
        &self,
        token_id: u32,
        state: &mut DecoderState,
        config: &Qwen38Config,
    ) -> Result<Vec<f32>> {
        Ok(self.decode_target_step(token_id, state, config)?.logits)
    }

    fn mtp_final_hidden(
        &self,
        next_token_id: u32,
        previous_final_hidden: &[f32],
        absolute_position: usize,
        state: &mut MtpState,
        config: &Qwen38Config,
    ) -> Result<Vec<f32>> {
        if previous_final_hidden.len() != config.hidden_size
            || absolute_position != state.tokens + 1
            || absolute_position >= state.maximum_tokens
        {
            return Err(EngineError::Shape(
                "MTP hidden width or absolute position differs from its cache".into(),
            ));
        }
        let embedding = self.embedding(next_token_id)?;
        let embedding = self.rms_norm("mtp.pre_fc_norm_embedding.weight", &embedding)?;
        let hidden = self.rms_norm("mtp.pre_fc_norm_hidden.weight", previous_final_hidden)?;
        // Qwen3.8 leaves `mtp_hidden_states_first` unset. Transformers therefore
        // uses [normalized embedding, normalized previous hidden] in this order.
        let mut projection_input = Vec::with_capacity(config.hidden_size * 2);
        projection_input.extend_from_slice(&embedding);
        projection_input.extend_from_slice(&hidden);
        let mut draft_hidden = self.projection("mtp.fc.weight", &projection_input)?;
        draft_hidden = self.full_attention_residual_at(
            "mtp.layers.0",
            &draft_hidden,
            absolute_position,
            state.tokens,
            &mut state.attention,
            config,
        )?;
        draft_hidden = self.decoder_mlp_residual("mtp.layers.0", &draft_hidden)?;
        let final_hidden = self.rms_norm("mtp.norm.weight", &draft_hidden)?;
        state.tokens += 1;
        Ok(final_hidden)
    }

    /// Execute the native Qwen3.8 one-layer MTP graph for base hidden state
    /// `p` and the already target-selected token `p+1`. The result drafts
    /// token `p+2`; acceptance is deliberately outside this method because a
    /// target-model transition must verify every proposal.
    pub fn mtp_draft(
        &self,
        next_token_id: u32,
        previous_final_hidden: &[f32],
        absolute_position: usize,
        state: &mut MtpState,
        config: &Qwen38Config,
    ) -> Result<TargetStep> {
        let final_hidden = self.mtp_final_hidden(
            next_token_id,
            previous_final_hidden,
            absolute_position,
            state,
            config,
        )?;
        let logits = self.logits_from_final_hidden(&final_hidden)?;
        Ok(TargetStep {
            final_hidden,
            logits,
        })
    }

    pub fn mtp_restricted_draft(
        &self,
        next_token_id: u32,
        previous_final_hidden: &[f32],
        absolute_position: usize,
        state: &mut MtpState,
        config: &Qwen38Config,
        token_ids: &[u32],
    ) -> Result<RestrictedDraftStep> {
        let final_hidden = self.mtp_final_hidden(
            next_token_id,
            previous_final_hidden,
            absolute_position,
            state,
            config,
        )?;
        let distribution = self.restricted_logits_from_final_hidden(&final_hidden, token_ids)?;
        Ok(RestrictedDraftStep {
            final_hidden,
            distribution,
        })
    }

    fn mtp_advance(
        &self,
        next_token_id: u32,
        previous_final_hidden: &[f32],
        absolute_position: usize,
        state: &mut MtpState,
        config: &Qwen38Config,
    ) -> Result<()> {
        self.mtp_final_hidden(
            next_token_id,
            previous_final_hidden,
            absolute_position,
            state,
            config,
        )?;
        Ok(())
    }
}

impl ModelExecutor for CpuCorrectnessExecutor {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn hardware_profile(&self) -> &str {
        self.backend.profile()
    }

    fn promotion_state(&self) -> PromotionState {
        // Quantized projections may use AVX2/NEON, but token mixers in this
        // executor deliberately remain the scalar correctness composition.
        PromotionState::Verifier
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            vocab_size: self.config.vocab_size,
            maximum_context_tokens: self.config.max_position_embeddings as u64,
            mtp: self.config.mtp_num_hidden_layers == 1,
            maximum_draft_tokens: if self.config.mtp_num_hidden_layers == 1 {
                MAXIMUM_CHAINED_MTP_DRAFTS as u32
            } else {
                0
            },
            cancellation: true,
            session_reset: true,
            no_hidden_fallbacks: false,
        }
    }

    fn load(
        &mut self,
        artifact: &ModelArtifact,
        profile: &MemoryProfile,
        mtp_draft_token_ids: &[u32],
    ) -> Result<()> {
        if self.artifact.is_some()
            || self.state.is_some()
            || self.mtp_state.is_some()
            || self.last_final_hidden.is_some()
            || self.pending_speculative.is_some()
            || !self.mtp_draft_token_ids.is_empty()
        {
            return Err(EngineError::InvalidState(
                "CPU correctness executor is already loaded".into(),
            ));
        }
        if profile.linear_state_dtype != crate::memory::LinearStateDType::F32
            || profile.mtp_draft_tokens as usize > MAXIMUM_CHAINED_MTP_DRAFTS
        {
            return Err(EngineError::UnsupportedOperation {
                backend: "cpu",
                operation: "correctness executor memory profile",
                reason: "the scalar oracle owns FP32 state and at most four chained MTP drafts"
                    .into(),
            });
        }
        let admitted_context = usize::try_from(profile.context_tokens)
            .map_err(|_| EngineError::MemoryBudget("CPU context capacity exceeds usize".into()))?;
        if admitted_context == 0 || admitted_context > self.config.max_position_embeddings {
            return Err(EngineError::MemoryBudget(
                "CPU memory profile context exceeds model capacity".into(),
            ));
        }
        if mtp_draft_token_ids.is_empty()
            || mtp_draft_token_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || mtp_draft_token_ids
                .iter()
                .any(|token| *token as usize >= self.config.vocab_size)
        {
            return Err(EngineError::InvalidArtifact(
                "CPU executor received a noncanonical MTP draft vocabulary".into(),
            ));
        }
        self.artifact = Some(artifact.clone());
        self.mtp_draft_token_ids = mtp_draft_token_ids.to_vec();
        self.admitted_context = admitted_context;
        self.admitted_draft_tokens = profile.mtp_draft_tokens as usize;
        self.warmed = false;
        self.allocations = AllocationSnapshot {
            model_bytes: profile.resident_model_bytes,
            ..AllocationSnapshot::default()
        };
        Ok(())
    }

    fn warmup(&mut self) -> Result<()> {
        if self.artifact.is_none() {
            return Err(EngineError::InvalidState(
                "CPU correctness executor is not loaded".into(),
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
        if !self.warmed || tokens.is_empty() || tokens.len() > self.admitted_context {
            return Err(EngineError::InvalidState(
                "CPU prefill requires a warm executor and admitted non-empty tokens".into(),
            ));
        }
        if mtp_enabled && self.admitted_draft_tokens == 0 {
            return Err(EngineError::InvalidState(
                "CPU prefill enabled MTP without an admitted draft block".into(),
            ));
        }
        self.state = Some(DecoderState::new(&self.config, self.admitted_context)?);
        self.mtp_state = if mtp_enabled {
            Some(MtpState::new(&self.config, self.admitted_context)?)
        } else {
            None
        };
        self.last_final_hidden = None;
        self.pending_speculative = None;
        let artifact = self.artifact.as_ref().ok_or_else(|| {
            EngineError::InvalidState("CPU prefill has no loaded artifact".into())
        })?;
        let decoder = ArtifactDecoder::new(artifact, &self.backend, self.config.rms_norm_epsilon)?;
        let state = self.state.as_mut().expect("state created above");
        let mut target_logits = Vec::new();
        for (position, token) in tokens.iter().enumerate() {
            if cancellation.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            if mtp_enabled && position > 0 {
                let previous = self.last_final_hidden.as_ref().ok_or_else(|| {
                    EngineError::InvalidState("MTP prefill lost target final hidden state".into())
                })?;
                decoder.mtp_advance(
                    *token,
                    previous,
                    position,
                    self.mtp_state.as_mut().expect("MTP state created above"),
                    &self.config,
                )?;
            }
            let step = decoder.decode_target_step(*token, state, &self.config)?;
            target_logits = step.logits;
            self.last_final_hidden = Some(step.final_hidden);
        }
        self.update_session_allocation()?;
        Ok(ExecutorStep {
            target_logits,
            draft_logits: Vec::new(),
            target_verification_logits: Vec::new(),
            bonus_logits: None,
        })
    }

    fn decode(
        &mut self,
        token: u32,
        mtp_enabled: bool,
        cancellation: &CancellationToken,
    ) -> Result<ExecutorStep> {
        if !self.warmed || cancellation.is_cancelled() {
            return if cancellation.is_cancelled() {
                Err(EngineError::Cancelled)
            } else {
                Err(EngineError::InvalidState(
                    "CPU correctness decode is not warm".into(),
                ))
            };
        }
        if self.pending_speculative.is_some() {
            return Err(EngineError::InvalidState(
                "CPU decode requires the previous speculative branch to be committed".into(),
            ));
        }
        let artifact = self
            .artifact
            .as_ref()
            .ok_or_else(|| EngineError::InvalidState("CPU decode has no loaded artifact".into()))?;
        let decoder = ArtifactDecoder::new(artifact, &self.backend, self.config.rms_norm_epsilon)?;
        let state = self.state.as_mut().ok_or_else(|| {
            EngineError::InvalidState("CPU decode requires a prefilled state".into())
        })?;
        if mtp_enabled && self.admitted_draft_tokens == 0 {
            return Err(EngineError::InvalidState(
                "CPU decode enabled MTP without an admitted draft block".into(),
            ));
        }
        let first_draft_logits = if mtp_enabled {
            let previous = self.last_final_hidden.as_ref().ok_or_else(|| {
                EngineError::InvalidState("MTP decode has no previous target hidden state".into())
            })?;
            let absolute_position = state.position();
            let draft = decoder.mtp_restricted_draft(
                token,
                previous,
                absolute_position,
                self.mtp_state
                    .as_mut()
                    .ok_or_else(|| EngineError::InvalidState("MTP decode has no cache".into()))?,
                &self.config,
                &self.mtp_draft_token_ids,
            )?;
            Some(draft.distribution)
        } else {
            None
        };
        let target = decoder.decode_target_step(token, state, &self.config)?;
        let (draft_logits, target_verification_logits, bonus_logits) = if mtp_enabled {
            let mut draft_logits = Vec::with_capacity(self.admitted_draft_tokens);
            let mut target_verification_logits = Vec::with_capacity(self.admitted_draft_tokens);
            let mut candidate_tokens = Vec::with_capacity(self.admitted_draft_tokens);
            let mut target_branch = state.clone();
            let mut mtp_branch = self
                .mtp_state
                .as_ref()
                .ok_or_else(|| EngineError::InvalidState("MTP decode has no cache".into()))?
                .clone();
            let mut current_draft = first_draft_logits;
            let mut current_target_hidden = target.final_hidden.clone();
            let mut current_target_logits = target.logits.clone();
            for depth in 0..self.admitted_draft_tokens {
                if cancellation.is_cancelled() {
                    return Err(EngineError::Cancelled);
                }
                target_verification_logits.push(current_target_logits);
                let draft = current_draft.take().ok_or_else(|| {
                    EngineError::InvalidState("MTP draft chain ended early".into())
                })?;
                let candidate = draft.greedy_token()?;
                draft_logits.push(draft);
                candidate_tokens.push(candidate);
                let absolute_position = target_branch.position();
                let next_draft = if depth + 1 < self.admitted_draft_tokens {
                    Some(decoder.mtp_restricted_draft(
                        candidate,
                        &current_target_hidden,
                        absolute_position,
                        &mut mtp_branch,
                        &self.config,
                        &self.mtp_draft_token_ids,
                    )?)
                } else {
                    None
                };
                let next_target =
                    decoder.decode_target_step(candidate, &mut target_branch, &self.config)?;
                current_target_hidden = next_target.final_hidden;
                current_target_logits = next_target.logits;
                if let Some(next_draft) = next_draft {
                    current_draft = Some(next_draft.distribution);
                }
            }
            self.pending_speculative = Some(PendingSpeculativeBranch { candidate_tokens });
            (
                draft_logits,
                target_verification_logits,
                Some(current_target_logits),
            )
        } else {
            (Vec::new(), Vec::new(), None)
        };
        self.last_final_hidden = Some(target.final_hidden);
        self.update_session_allocation()?;
        Ok(ExecutorStep {
            target_logits: target.logits,
            draft_logits,
            target_verification_logits,
            bonus_logits,
        })
    }

    fn commit_speculative(
        &mut self,
        accepted_drafts: u32,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        let branch = self.pending_speculative.take().ok_or_else(|| {
            EngineError::InvalidState("CPU executor has no pending speculative branch".into())
        })?;
        let accepted_drafts = usize::try_from(accepted_drafts).map_err(|_| {
            EngineError::InvalidState("CPU accepted MTP depth exceeds usize".into())
        })?;
        if accepted_drafts > branch.candidate_tokens.len() {
            return Err(EngineError::InvalidState(
                "CPU accepted MTP depth exceeds the pending candidate block".into(),
            ));
        }
        if accepted_drafts == 0 {
            self.update_session_allocation()?;
            return Ok(());
        }
        if cancellation.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let artifact = self.artifact.as_ref().ok_or_else(|| {
            EngineError::InvalidState("CPU speculative commit has no loaded artifact".into())
        })?;
        let decoder = ArtifactDecoder::new(artifact, &self.backend, self.config.rms_norm_epsilon)?;
        for candidate in branch.candidate_tokens.into_iter().take(accepted_drafts) {
            if cancellation.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            let previous_final_hidden = self.last_final_hidden.as_ref().ok_or_else(|| {
                EngineError::InvalidState("CPU speculative commit lost target hidden state".into())
            })?;
            let absolute_position = self
                .state
                .as_ref()
                .ok_or_else(|| {
                    EngineError::InvalidState("CPU speculative commit has no target state".into())
                })?
                .position();
            decoder.mtp_advance(
                candidate,
                previous_final_hidden,
                absolute_position,
                self.mtp_state.as_mut().ok_or_else(|| {
                    EngineError::InvalidState("CPU speculative commit has no MTP state".into())
                })?,
                &self.config,
            )?;
            let target = decoder.decode_target_step(
                candidate,
                self.state.as_mut().ok_or_else(|| {
                    EngineError::InvalidState("CPU speculative commit has no target state".into())
                })?,
                &self.config,
            )?;
            self.last_final_hidden = Some(target.final_hidden);
        }
        self.update_session_allocation()?;
        Ok(())
    }

    fn reset_session(&mut self) -> Result<()> {
        self.state = None;
        self.mtp_state = None;
        self.last_final_hidden = None;
        self.pending_speculative = None;
        self.allocations.session_bytes = 0;
        Ok(())
    }

    fn unload(&mut self) -> Result<()> {
        self.state = None;
        self.mtp_state = None;
        self.last_final_hidden = None;
        self.pending_speculative = None;
        self.mtp_draft_token_ids.clear();
        self.artifact = None;
        self.admitted_context = 0;
        self.admitted_draft_tokens = 0;
        self.warmed = false;
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
    use crate::backend::cpu::CpuBackend;
    use crate::format::{ArtifactBuilder, PackedTensor, TensorDType, DEFAULT_ALIGNMENT};
    use crate::loader::ChecksumPolicy;
    use crate::quant::{Q2Block64, BLOCK_LEN};
    use crate::release::KvMemoryFormula;
    use half::f16;

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn recovered_matrix(name: &str, rows: usize, columns: usize, value: f32) -> Vec<PackedTensor> {
        assert!(columns.is_multiple_of(BLOCK_LEN));
        let mut weights = Vec::new();
        for _ in 0..rows * (columns / BLOCK_LEN) {
            weights.extend_from_slice(&Q2Block64::quantize(&[value; BLOCK_LEN]).unwrap().encode());
        }
        vec![
            PackedTensor {
                name: name.into(),
                dtype: TensorDType::Q2B64,
                shape: vec![rows as u64, columns as u64],
                bytes: weights,
            },
            PackedTensor {
                name: format!("{name}.s_in"),
                dtype: TensorDType::F16,
                shape: vec![columns as u64],
                bytes: f16_bytes(&vec![1.0; columns]),
            },
            PackedTensor {
                name: format!("{name}.s_out"),
                dtype: TensorDType::F16,
                shape: vec![rows as u64],
                bytes: f16_bytes(&vec![1.0; rows]),
            },
        ]
    }

    #[test]
    fn mmap_graph_executes_embedding_mlp_norm_and_lm_head() {
        let hidden = BLOCK_LEN;
        let intermediate = BLOCK_LEN;
        let layer = "model.language_model.layers.0";
        let mut tensors = Vec::new();
        tensors.extend(recovered_matrix(
            "model.language_model.embed_tokens.weight",
            2,
            hidden,
            0.5,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.mlp.gate_proj.weight"),
            intermediate,
            hidden,
            1.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.mlp.up_proj.weight"),
            intermediate,
            hidden,
            2.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.mlp.down_proj.weight"),
            hidden,
            intermediate,
            1.0 / intermediate as f32,
        ));
        let head_dim = 32;
        let query_heads = 2;
        let key_value_heads = 1;
        let query_width = query_heads * head_dim;
        let key_value_width = key_value_heads * head_dim;
        tensors.extend(recovered_matrix(
            &format!("{layer}.self_attn.q_proj.weight"),
            query_width * 2,
            hidden,
            1.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.self_attn.k_proj.weight"),
            key_value_width,
            hidden,
            1.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.self_attn.v_proj.weight"),
            key_value_width,
            hidden,
            2.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{layer}.self_attn.o_proj.weight"),
            hidden,
            query_width,
            1.0 / query_width as f32,
        ));
        let linear_key_heads = 1;
        let linear_value_heads = 2;
        let linear_key_dim = 32;
        let linear_value_dim = 32;
        let linear_kernel = 2;
        let linear_key_width = linear_key_heads * linear_key_dim;
        let linear_value_width = linear_value_heads * linear_value_dim;
        let linear_convolution_width = linear_key_width * 2 + linear_value_width;
        let linear = format!("{layer}.linear_attn");
        tensors.extend(recovered_matrix(
            &format!("{linear}.in_proj_qkv.weight"),
            linear_convolution_width,
            hidden,
            1.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{linear}.in_proj_z.weight"),
            linear_value_width,
            hidden,
            1.0 / hidden as f32,
        ));
        for projection in ["in_proj_a", "in_proj_b"] {
            tensors.extend(recovered_matrix(
                &format!("{linear}.{projection}.weight"),
                linear_value_heads,
                hidden,
                1.0 / hidden as f32,
            ));
        }
        tensors.extend(recovered_matrix(
            &format!("{linear}.out_proj.weight"),
            hidden,
            linear_value_width,
            1.0 / linear_value_width as f32,
        ));
        let convolution_weights: Vec<f32> = (0..linear_convolution_width)
            .flat_map(|_| [0.0_f32, 1.0_f32])
            .collect();
        tensors.push(PackedTensor {
            name: format!("{linear}.conv1d.weight"),
            dtype: TensorDType::F16,
            shape: vec![linear_convolution_width as u64, 1, linear_kernel as u64],
            bytes: f16_bytes(&convolution_weights),
        });
        for name in ["A_log", "dt_bias"] {
            tensors.push(PackedTensor {
                name: format!("{linear}.{name}"),
                dtype: TensorDType::F32,
                shape: vec![linear_value_heads as u64],
                bytes: vec![0_u8; linear_value_heads * 4],
            });
        }
        tensors.push(PackedTensor {
            name: format!("{linear}.norm.weight"),
            dtype: TensorDType::F16,
            shape: vec![linear_value_dim as u64],
            bytes: f16_bytes(&vec![1.0; linear_value_dim]),
        });
        tensors.extend(recovered_matrix(
            "lm_head.weight",
            2,
            hidden,
            1.0 / hidden as f32,
        ));
        tensors.extend(recovered_matrix(
            "mtp.fc.weight",
            hidden,
            hidden * 2,
            1.0 / (hidden * 2) as f32,
        ));
        let mtp_layer = "mtp.layers.0";
        for projection in ["gate_proj", "up_proj"] {
            tensors.extend(recovered_matrix(
                &format!("{mtp_layer}.mlp.{projection}.weight"),
                intermediate,
                hidden,
                1.0 / hidden as f32,
            ));
        }
        tensors.extend(recovered_matrix(
            &format!("{mtp_layer}.mlp.down_proj.weight"),
            hidden,
            intermediate,
            1.0 / intermediate as f32,
        ));
        tensors.extend(recovered_matrix(
            &format!("{mtp_layer}.self_attn.q_proj.weight"),
            query_width * 2,
            hidden,
            1.0 / hidden as f32,
        ));
        for projection in ["k_proj", "v_proj"] {
            tensors.extend(recovered_matrix(
                &format!("{mtp_layer}.self_attn.{projection}.weight"),
                key_value_width,
                hidden,
                1.0 / hidden as f32,
            ));
        }
        tensors.extend(recovered_matrix(
            &format!("{mtp_layer}.self_attn.o_proj.weight"),
            hidden,
            query_width,
            1.0 / query_width as f32,
        ));
        for name in [
            format!("{layer}.input_layernorm.weight"),
            format!("{layer}.post_attention_layernorm.weight"),
            format!("{mtp_layer}.input_layernorm.weight"),
            format!("{mtp_layer}.post_attention_layernorm.weight"),
            "model.language_model.norm.weight".into(),
            "mtp.norm.weight".into(),
            "mtp.pre_fc_norm_embedding.weight".into(),
            "mtp.pre_fc_norm_hidden.weight".into(),
        ] {
            tensors.push(PackedTensor {
                name,
                dtype: TensorDType::F16,
                shape: vec![hidden as u64],
                bytes: f16_bytes(&vec![0.0; hidden]),
            });
        }
        for name in [
            format!("{layer}.self_attn.q_norm.weight"),
            format!("{layer}.self_attn.k_norm.weight"),
            format!("{mtp_layer}.self_attn.q_norm.weight"),
            format!("{mtp_layer}.self_attn.k_norm.weight"),
        ] {
            tensors.push(PackedTensor {
                name,
                dtype: TensorDType::F16,
                shape: vec![head_dim as u64],
                bytes: f16_bytes(&vec![0.0; head_dim]),
            });
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("decoder-slice.ctoxq");
        ArtifactBuilder {
            model: "test/qwen".into(),
            revision: "0123456789abcdef".into(),
            target: "cpu-test".into(),
            alignment: DEFAULT_ALIGNMENT,
            tensors,
        }
        .write_new(&path)
        .unwrap();
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors).unwrap();
        let backend = CpuBackend::scalar_verifier();
        let decoder = ArtifactDecoder::new(&artifact, &backend, 1e-6).unwrap();

        let embedding = decoder.embedding(0).unwrap();
        assert_eq!(embedding.len(), hidden);
        assert!(embedding.iter().all(|value| (*value - 0.5).abs() < 1e-6));
        assert!(decoder.embedding(2).is_err());

        let residual = vec![1.0_f32; hidden];
        let attention_config = Qwen38Config {
            hidden_size: hidden,
            num_attention_heads: query_heads,
            num_key_value_heads: key_value_heads,
            head_dim,
            rotary_dim: 16,
            rope_theta: 10_000.0,
            ..Qwen38Config::default()
        };
        let mut attention_state = FullAttentionState::new(key_value_heads, head_dim, 2).unwrap();
        let attention = decoder
            .full_attention_residual(layer, &residual, 0, &mut attention_state, &attention_config)
            .unwrap();
        let normalized = (1.0_f32 + 1e-6).sqrt().recip();
        let expected_attention = 1.0 + 2.0 * normalized / (1.0 + (-normalized).exp());
        assert!(attention
            .iter()
            .all(|value| (*value - expected_attention).abs() < 2e-4));
        assert_eq!(attention_state.tokens(), 1);
        assert!(decoder
            .full_attention_residual(layer, &residual, 2, &mut attention_state, &attention_config,)
            .is_err());
        attention_state.reset();
        assert_eq!(attention_state.tokens(), 0);

        let linear_config = Qwen38Config {
            hidden_size: hidden,
            num_attention_heads: query_heads,
            num_key_value_heads: key_value_heads,
            head_dim,
            rotary_dim: 16,
            rope_theta: 10_000.0,
            linear_num_key_heads: linear_key_heads,
            linear_num_value_heads: linear_value_heads,
            linear_key_head_dim: linear_key_dim,
            linear_value_head_dim: linear_value_dim,
            linear_conv_kernel_dim: linear_kernel,
            ..Qwen38Config::default()
        };
        let mut linear_state = LinearAttentionState::new(&linear_config, 2).unwrap();
        assert_eq!(linear_state.allocated_bytes(), 9_216);
        let linear_output = decoder
            .linear_attention_residual(layer, &residual, 0, &mut linear_state, &linear_config)
            .unwrap();
        assert_eq!(linear_output.len(), hidden);
        assert!(linear_output.iter().all(|value| value.is_finite()));
        assert_eq!(linear_state.tokens(), 1);
        assert!(decoder
            .linear_attention_residual(layer, &residual, 2, &mut linear_state, &linear_config)
            .is_err());
        linear_state.reset();
        let repeated = decoder
            .linear_attention_residual(layer, &residual, 0, &mut linear_state, &linear_config)
            .unwrap();
        assert_eq!(linear_output, repeated);

        let output = decoder.decoder_mlp_residual(layer, &residual).unwrap();
        let inverse = (1.0_f32 + 1e-6).sqrt().recip();
        let gate = inverse;
        let up = 2.0 * inverse;
        let activated = gate / (1.0 + (-gate).exp()) * up;
        let expected = 1.0 + activated;
        assert!(output.iter().all(|value| (*value - expected).abs() < 2e-4));

        let logits = decoder.final_logits(&output).unwrap();
        assert_eq!(logits.len(), 2);
        assert!(logits.iter().all(|value| value.is_finite()));
        assert!((logits[0] - logits[1]).abs() < 1e-6);
        let final_hidden = decoder.final_hidden(&output).unwrap();
        let restricted = decoder
            .restricted_logits_from_final_hidden(&final_hidden, &[1])
            .unwrap();
        match restricted {
            DraftDistribution::Restricted {
                token_ids,
                logits: scores,
            } => {
                assert_eq!(token_ids, vec![1]);
                assert_eq!(scores, vec![logits[1]]);
            }
            DraftDistribution::Full(_) => panic!("restricted LM head returned full logits"),
        }
        assert!(decoder
            .restricted_logits_from_final_hidden(&final_hidden, &[1, 1])
            .is_err());
        assert!(decoder
            .restricted_logits_from_final_hidden(&final_hidden, &[2])
            .is_err());

        let linear_decode_config = Qwen38Config {
            vocab_size: 2,
            num_hidden_layers: 1,
            full_attention_interval: 4,
            max_position_embeddings: 2,
            ..linear_config.clone()
        };
        let mut target_state = DecoderState::new(&linear_decode_config, 2).unwrap();
        assert_eq!(target_state.allocated_bytes(), 9_216);
        let target_logits = decoder
            .decode_target_token(0, &mut target_state, &linear_decode_config)
            .unwrap();
        assert_eq!(target_logits.len(), 2);
        assert_eq!(target_state.position(), 1);
        target_state.reset();
        assert_eq!(target_state.position(), 0);
        assert_eq!(
            decoder
                .decode_target_token(0, &mut target_state, &linear_decode_config)
                .unwrap(),
            target_logits
        );

        let full_decode_config = Qwen38Config {
            vocab_size: 2,
            num_hidden_layers: 1,
            full_attention_interval: 1,
            max_position_embeddings: 2,
            ..attention_config
        };
        let mut full_target_state = DecoderState::new(&full_decode_config, 2).unwrap();
        let target_step = decoder
            .decode_target_step(0, &mut full_target_state, &full_decode_config)
            .unwrap();
        assert_eq!(target_step.final_hidden.len(), hidden);
        assert_eq!(target_step.logits.len(), 2);
        assert!(target_step.logits.iter().all(|value| value.is_finite()));
        assert_eq!(full_target_state.position(), 1);

        let mut mtp_state = MtpState::new(&full_decode_config, 2).unwrap();
        let draft = decoder
            .mtp_draft(
                0,
                &target_step.final_hidden,
                1,
                &mut mtp_state,
                &full_decode_config,
            )
            .unwrap();
        assert_eq!(draft.final_hidden.len(), hidden);
        assert_eq!(draft.logits.len(), 2);
        assert!(draft.logits.iter().all(|value| value.is_finite()));
        assert_eq!(mtp_state.tokens(), 1);
        assert!(decoder
            .mtp_draft(
                0,
                &target_step.final_hidden,
                2,
                &mut mtp_state,
                &full_decode_config,
            )
            .is_err());
        mtp_state.reset();
        assert_eq!(mtp_state.tokens(), 0);
        assert_eq!(
            decoder
                .mtp_draft(
                    0,
                    &target_step.final_hidden,
                    1,
                    &mut mtp_state,
                    &full_decode_config,
                )
                .unwrap(),
            draft
        );

        let executor_config = Qwen38Config {
            max_position_embeddings: 12,
            ..linear_decode_config.clone()
        };
        let profile = MemoryProfile {
            profile_id: "cpu-correctness-mtp4".into(),
            pack_id: "test".into(),
            context_tokens: 12,
            sessions: 1,
            resident_model_bytes: artifact.file_bytes(),
            persistent_backend_graph_bytes: 0,
            persistent_runtime_bytes: 0,
            linear_state_dtype: crate::memory::LinearStateDType::F32,
            linear_state_bytes_per_session: 9_216,
            mtp_draft_tokens: 4,
            speculative_state_strategy: crate::memory::SpeculativeStateStrategy::ReplayOnReject,
            speculative_linear_state_bytes_per_session: 9_216,
            kv: KvMemoryFormula {
                fixed_bytes_per_session: 0,
                bytes_per_token_per_session: 0,
                retained_q4_tokens_per_session: 0,
                q4_delta_bytes_per_token: 0,
            },
            mtp_kv: KvMemoryFormula {
                fixed_bytes_per_session: 0,
                bytes_per_token_per_session: 0,
                retained_q4_tokens_per_session: 0,
                q4_delta_bytes_per_token: 0,
            },
            prefill_scratch_peak_bytes: 0,
            decode_scratch_peak_bytes: 0,
            loader_transient_peak_bytes: 0,
            accelerator_unattributed_reserve_bytes: 0,
            hard_limit_bytes: 1 << 30,
        };
        let mut executor = CpuCorrectnessExecutor::scalar(executor_config);
        executor.load(&artifact, &profile, &[0, 1]).unwrap();
        executor.warmup().unwrap();
        let cancellation = CancellationToken::default();
        let prefill = executor.prefill(&[0], false, &cancellation).unwrap();
        assert_eq!(prefill.target_logits.len(), 2);
        assert!(executor.allocations().session_bytes > 0);
        let decode = executor.decode(0, false, &cancellation).unwrap();
        assert_eq!(decode.target_logits.len(), 2);
        executor.reset_session().unwrap();
        assert_eq!(executor.allocations().session_bytes, 0);
        let mtp_prefill = executor.prefill(&[0], true, &cancellation).unwrap();
        assert!(mtp_prefill.draft_logits.is_empty());
        let mtp_decode = executor.decode(0, true, &cancellation).unwrap();
        assert_eq!(mtp_decode.target_logits.len(), 2);
        assert_eq!(mtp_decode.draft_logits.len(), 4);
        assert!(mtp_decode
            .draft_logits
            .iter()
            .all(|logits| logits.is_restricted() && logits.len() == 2));
        assert_eq!(mtp_decode.target_verification_logits.len(), 4);
        assert_eq!(mtp_decode.bonus_logits.as_ref().unwrap().len(), 2);
        assert!(executor.pending_speculative.is_some());
        executor
            .commit_speculative(0, &CancellationToken::default())
            .unwrap();
        assert_eq!(executor.state.as_ref().unwrap().position(), 2);
        assert_eq!(executor.mtp_state.as_ref().unwrap().tokens(), 1);

        executor.reset_session().unwrap();
        executor.prefill(&[0], true, &cancellation).unwrap();
        let accepted = executor.decode(0, true, &cancellation).unwrap();
        for (draft, target) in accepted
            .draft_logits
            .iter()
            .zip(&accepted.target_verification_logits)
        {
            assert_eq!(draft.greedy_token().unwrap(), greedy_token(target).unwrap());
        }
        executor
            .commit_speculative(2, &CancellationToken::default())
            .unwrap();
        assert_eq!(executor.state.as_ref().unwrap().position(), 4);
        assert_eq!(executor.mtp_state.as_ref().unwrap().tokens(), 3);
        let fallback_token = greedy_token(&accepted.target_verification_logits[2]).unwrap();
        let aligned = executor
            .decode(fallback_token, true, &cancellation)
            .unwrap();
        assert_eq!(aligned.target_verification_logits.len(), 4);
        executor
            .commit_speculative(0, &CancellationToken::default())
            .unwrap();
        assert_eq!(executor.state.as_ref().unwrap().position(), 5);
        assert_eq!(executor.mtp_state.as_ref().unwrap().tokens(), 4);
        executor.reset_session().unwrap();
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            executor.prefill(&[0], false, &cancelled),
            Err(EngineError::Cancelled)
        ));
        executor.reset_session().unwrap();
        executor.unload().unwrap();
        assert!(executor.allocations().is_zero());
        assert!(executor.mtp_draft_token_ids.is_empty());
    }
}
