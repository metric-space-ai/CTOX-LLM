//! Artifact-backed CUDA projection graph for the frozen Qwen3.8-27B model.
//!
//! This module is deliberately model-specific. It resolves every target and
//! MTP projection into the exact shared-correction group used by the recovery
//! pipeline, then prepares immutable packed weights and recovery scales
//! directly from the CTOXQ mapping. It does not implement a generic graph
//! runtime. The bounded prefill ownership also includes the dedicated batched
//! embedding row-ID/output workspace; no token-wise host launch is admitted.

use std::collections::{BTreeMap, BTreeSet};

use crate::backend::cuda_runtime::{
    CudaCandidateRuntime, CudaCausalConvConfig, CudaDeviceF32View, CudaGatedDeltaConfig,
    CudaGatedRmsNormConfig, CudaPagedGqaConfig, CudaPartialRopeConfig, CudaQueryGateConfig,
    CudaRmsNormConfig, PreparedCudaA8Activation, PreparedCudaA8Projection,
    PreparedCudaBatchedA8OutputArena, PreparedCudaBatchedA8Workspace,
    PreparedCudaBatchedEmbeddingWorkspace, PreparedCudaBatchedGatedRmsNormOutput,
    PreparedCudaBatchedQueryGateOutput, PreparedCudaBatchedRmsNormWorkspace,
    PreparedCudaBatchedRopeWorkspace, PreparedCudaCausalConv, PreparedCudaCausalConvScanOutput,
    PreparedCudaEmbedding, PreparedCudaF32Checkpoint, PreparedCudaF32Concat,
    PreparedCudaGatedDelta, PreparedCudaGatedDeltaInputs, PreparedCudaGatedDeltaScanInputs,
    PreparedCudaGatedDeltaScanOutput, PreparedCudaGatedRmsNorm, PreparedCudaGatheredA8Projection,
    PreparedCudaPagedGqa, PreparedCudaPagedGqaPrefillOutput, PreparedCudaPartialRope,
    PreparedCudaQueryGate, PreparedCudaResidualRmsNorm, PreparedCudaRmsNorm,
};
use crate::backend::cuda_schedule::{
    CudaDecodeOperation, CudaDecodeSchedule, CudaDecodeStep, CudaMtpPrefillAlignment,
    CudaNormBinding, CudaPrefillChunk, CudaPrefillOperation, CudaPrefillSchedule, CudaPrefillStep,
};
use crate::backend::{Activation, ScaleSlice};
use crate::config::LayerKind;
use crate::fanout::qwen38_fanout_groups;
use crate::kv_cache::{DEFAULT_KV_PAGE_TOKENS, DEFAULT_KV_RECENT_TOKENS, DEFAULT_KV_SINK_TOKENS};
use crate::loader::{FloatTensorView, ModelArtifact};
use crate::quant::BLOCK_LEN;
use crate::tensor_contract::{expected_tensor_contract, validate_tensor_contract, TensorClass};
use crate::{EngineError, Qwen38Config, Result};

const EMBEDDING_MATRIX: &str = "model.language_model.embed_tokens.weight";
const CUDA_PREFILL_CHUNK_TOKENS: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaProjectionGroupPlan {
    pub key: String,
    pub projection_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaProjectionPlan {
    groups: Vec<CudaProjectionGroupPlan>,
    projection_groups: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CudaPreparedResource {
    Embedding,
    Activation(String),
    Projection(String),
    LinearMixer(usize),
    FullAttention(String),
    RegularNorm(String),
    ResidualNorm(String),
    TokenBarrier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaBoundDecodeStep {
    pub schedule_index: usize,
    pub layer: Option<usize>,
    pub operation: CudaDecodeOperation,
    pub resources: Vec<CudaPreparedResource>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDecodeBindingPlan {
    steps: Vec<CudaBoundDecodeStep>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaBoundPrefillStep {
    pub schedule_index: usize,
    pub layer: Option<usize>,
    pub operation: CudaPrefillOperation,
    pub resources: Vec<CudaPreparedResource>,
}

/// Exact resident-resource contract for the layer-major prompt program.
///
/// This deliberately does not claim that the executor already dispatches the
/// chunked program. It proves that every scheduled operation resolves to the
/// same immutable model/state owners used by decode, before transient chunk
/// workspaces are admitted and the sequential prefill path is replaced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaPrefillBindingPlan {
    steps: Vec<CudaBoundPrefillStep>,
    max_chunk_tokens: usize,
    projection_workspace: CudaPrefillProjectionWorkspacePlan,
}

/// Fail-closed progress tracker for one bound prefill chunk.
///
/// A caller may commit the returned token position only after advancing every
/// bound operation, including the sole final chunk barrier. This object does
/// not execute kernels itself; it prevents the production executor from
/// silently skipping, duplicating, or reordering a model-specific dispatch.
#[derive(Debug)]
pub struct CudaPrefillExecutionCursor<'a> {
    plan: &'a CudaPrefillBindingPlan,
    chunk: CudaPrefillChunk,
    next_step: usize,
}

/// Exact bytes for the shared full-attention prefill scratch admitted with a
/// loaded graph. These buffers are independent of layer count: the layer-major
/// schedule overwrites the same slots after each consumer has completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPrefillWorkspaceBudget {
    pub embedding_row_id_bytes: u64,
    pub embedding_output_bytes: u64,
    pub hidden_norm_bytes: u64,
    pub key_norm_bytes: u64,
    pub rope_table_bytes: u64,
    pub query_gate_bytes: u64,
    pub paged_gqa_output_bytes: u64,
    pub total_bytes: u64,
}

/// Fixed arena geometry for every chunk-wide Q2/Q4 projection. `lm_head` is
/// intentionally excluded: the schedule evaluates only the final prompt row
/// through its already-resident one-token output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPrefillProjectionWorkspacePlan {
    pub max_chunk_tokens: usize,
    pub activation_columns: usize,
    pub output_slot_rows: [usize; 4],
    pub chunk_projection_count: usize,
    pub last_token_lm_head_rows: usize,
    pub activation_code_bytes: u64,
    pub activation_scale_bytes: u64,
    pub output_arena_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPrefillLinearWorkspaceBudget {
    pub causal_conv_output_bytes: u64,
    pub gated_delta_input_bytes: u64,
    pub gated_delta_output_bytes: u64,
    pub gated_rms_norm_output_bytes: u64,
    pub total_bytes: u64,
}

impl CudaPrefillLinearWorkspaceBudget {
    pub fn qwen38(config: &Qwen38Config, max_chunk_tokens: usize) -> Result<Self> {
        if max_chunk_tokens == 0 || max_chunk_tokens > 65_535 {
            return Err(EngineError::MemoryBudget(
                "CUDA linear workspace chunk capacity must be in 1..=65535".into(),
            ));
        }
        let chunk = u64::try_from(max_chunk_tokens)
            .map_err(|_| EngineError::MemoryBudget("CUDA prefill chunk exceeds u64".into()))?;
        let key_width = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA linear key width overflows".into()))?;
        let value_width = config
            .linear_num_value_heads
            .checked_mul(config.linear_value_head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA linear value width overflows".into()))?;
        let convolution_width = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or_else(|| EngineError::MemoryBudget("CUDA convolution width overflows".into()))?;
        let heads = u64::try_from(config.linear_num_value_heads)
            .map_err(|_| EngineError::MemoryBudget("CUDA linear heads exceed u64".into()))?;
        let key_dim = u64::try_from(config.linear_key_head_dim)
            .map_err(|_| EngineError::MemoryBudget("CUDA linear key dim exceeds u64".into()))?;
        let value_dim = u64::try_from(config.linear_value_head_dim)
            .map_err(|_| EngineError::MemoryBudget("CUDA linear value dim exceeds u64".into()))?;
        let f32_bytes =
            u64::try_from(std::mem::size_of::<f32>()).expect("f32 byte count always fits u64");
        let causal_conv_output_bytes = checked_product(
            &[
                chunk,
                u64::try_from(convolution_width).map_err(|_| {
                    EngineError::MemoryBudget("CUDA convolution width exceeds u64".into())
                })?,
                f32_bytes,
            ],
            "CUDA convolution scan output bytes",
        )?;
        let gated_delta_input_values = checked_add(
            checked_product(&[heads, key_dim, 2], "CUDA gated-delta Q/K values")?,
            checked_add(
                checked_product(&[heads, value_dim], "CUDA gated-delta V values")?,
                checked_product(&[heads, 2], "CUDA gated-delta scalar values")?,
                "CUDA gated-delta V/scalar values",
            )?,
            "CUDA gated-delta input values",
        )?;
        let gated_delta_input_bytes = checked_product(
            &[chunk, gated_delta_input_values, f32_bytes],
            "CUDA gated-delta input bytes",
        )?;
        let gated_delta_output_bytes = checked_product(
            &[chunk, heads, value_dim, f32_bytes],
            "CUDA gated-delta output bytes",
        )?;
        let gated_rms_norm_output_bytes = gated_delta_output_bytes;
        let total_bytes = [
            causal_conv_output_bytes,
            gated_delta_input_bytes,
            gated_delta_output_bytes,
            gated_rms_norm_output_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |total, value| {
            checked_add(total, value, "CUDA linear workspace bytes")
        })?;
        Ok(Self {
            causal_conv_output_bytes,
            gated_delta_input_bytes,
            gated_delta_output_bytes,
            gated_rms_norm_output_bytes,
            total_bytes,
        })
    }
}

impl CudaPrefillProjectionWorkspacePlan {
    fn qwen38(
        config: &Qwen38Config,
        projections: &CudaProjectionPlan,
        max_chunk_tokens: usize,
    ) -> Result<Self> {
        if max_chunk_tokens == 0 || max_chunk_tokens > 65_535 {
            return Err(EngineError::MemoryBudget(
                "CUDA projection workspace chunk capacity must be in 1..=65535".into(),
            ));
        }
        let output_slot_rows = [
            config.intermediate_size,
            config.intermediate_size,
            (config.num_key_value_heads * config.head_dim).max(config.linear_num_value_heads),
            config.linear_num_value_heads,
        ];
        let contract = expected_tensor_contract(config);
        let mut activation_columns = 0_usize;
        let mut chunk_projection_count = 0_usize;
        let mut last_token_lm_head_rows = None;
        for name in projections.projection_groups.keys() {
            let spec = contract.get(name).ok_or_else(|| {
                EngineError::InvalidState(format!(
                    "CUDA projection workspace cannot resolve tensor {name}"
                ))
            })?;
            let [rows, columns] = spec.shape.as_slice() else {
                return Err(EngineError::Shape(format!(
                    "CUDA projection {name} is not a rank-two matrix"
                )));
            };
            let rows = usize::try_from(*rows).map_err(|_| {
                EngineError::MemoryBudget(format!("CUDA projection {name} rows exceed usize"))
            })?;
            let columns = usize::try_from(*columns).map_err(|_| {
                EngineError::MemoryBudget(format!("CUDA projection {name} columns exceed usize"))
            })?;
            if name == "lm_head.weight" {
                last_token_lm_head_rows = Some(rows);
                continue;
            }
            let slot = prefill_projection_output_slot(name)?;
            if rows > output_slot_rows[slot] {
                return Err(EngineError::MemoryBudget(format!(
                    "CUDA projection {name} needs {rows} output rows, slot {slot} admits {}",
                    output_slot_rows[slot]
                )));
            }
            activation_columns = activation_columns.max(columns);
            chunk_projection_count += 1;
        }
        for group in projections.groups() {
            let mut slots = BTreeSet::new();
            for name in &group.projection_names {
                if name == "lm_head.weight" {
                    continue;
                }
                let slot = prefill_projection_output_slot(name)?;
                if !slots.insert(slot) {
                    return Err(EngineError::InvalidState(format!(
                        "CUDA projection group {} aliases live output slot {slot}",
                        group.key
                    )));
                }
            }
        }
        let last_token_lm_head_rows = last_token_lm_head_rows.ok_or_else(|| {
            EngineError::InvalidState("CUDA projection workspace omits LM head".into())
        })?;
        if chunk_projection_count + 1 != projections.projection_count() {
            return Err(EngineError::InvalidState(
                "CUDA projection workspace does not cover the complete graph".into(),
            ));
        }
        let chunk = u64::try_from(max_chunk_tokens)
            .map_err(|_| EngineError::MemoryBudget("CUDA prefill chunk exceeds u64".into()))?;
        let columns = u64::try_from(activation_columns)
            .map_err(|_| EngineError::MemoryBudget("CUDA projection columns exceed u64".into()))?;
        let f32_bytes =
            u64::try_from(std::mem::size_of::<f32>()).expect("f32 byte count always fits u64");
        let activation_code_bytes =
            checked_product(&[chunk, columns], "CUDA projection activation-code bytes")?;
        let scale_blocks = u64::try_from(activation_columns.div_ceil(BLOCK_LEN)).map_err(|_| {
            EngineError::MemoryBudget("CUDA projection scale blocks exceed u64".into())
        })?;
        let activation_scale_bytes = checked_product(
            &[chunk, scale_blocks, f32_bytes],
            "CUDA projection activation-scale bytes",
        )?;
        let output_rows = output_slot_rows
            .into_iter()
            .try_fold(0_u64, |total, rows| {
                let rows = u64::try_from(rows).map_err(|_| {
                    EngineError::MemoryBudget("CUDA projection output rows exceed u64".into())
                })?;
                checked_add(total, rows, "CUDA projection output rows")
            })?;
        let output_arena_bytes = checked_product(
            &[chunk, output_rows, f32_bytes],
            "CUDA projection output arena bytes",
        )?;
        let total_bytes = [
            activation_code_bytes,
            activation_scale_bytes,
            output_arena_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |total, value| {
            checked_add(total, value, "CUDA projection workspace bytes")
        })?;
        Ok(Self {
            max_chunk_tokens,
            activation_columns,
            output_slot_rows,
            chunk_projection_count,
            last_token_lm_head_rows,
            activation_code_bytes,
            activation_scale_bytes,
            output_arena_bytes,
            total_bytes,
        })
    }
}

impl CudaPrefillWorkspaceBudget {
    pub fn qwen38(config: &Qwen38Config, max_chunk_tokens: usize) -> Result<Self> {
        if max_chunk_tokens == 0 || max_chunk_tokens > 65_535 {
            return Err(EngineError::MemoryBudget(
                "CUDA prefill workspace chunk capacity must be in 1..=65535".into(),
            ));
        }
        let f32_bytes =
            u64::try_from(std::mem::size_of::<f32>()).expect("f32 byte count always fits u64");
        let u32_bytes =
            u64::try_from(std::mem::size_of::<u32>()).expect("u32 byte count always fits u64");
        let chunk = u64::try_from(max_chunk_tokens)
            .map_err(|_| EngineError::MemoryBudget("CUDA prefill chunk exceeds u64".into()))?;
        let hidden = u64::try_from(config.hidden_size)
            .map_err(|_| EngineError::MemoryBudget("CUDA hidden size exceeds u64".into()))?;
        let kv_heads = u64::try_from(config.num_key_value_heads)
            .map_err(|_| EngineError::MemoryBudget("CUDA KV-head count exceeds u64".into()))?;
        let query_heads = u64::try_from(config.num_attention_heads)
            .map_err(|_| EngineError::MemoryBudget("CUDA query-head count exceeds u64".into()))?;
        let head_dim = u64::try_from(config.head_dim)
            .map_err(|_| EngineError::MemoryBudget("CUDA head dimension exceeds u64".into()))?;
        let rope_half = u64::try_from(config.rotary_dim / 2)
            .map_err(|_| EngineError::MemoryBudget("CUDA rotary dimension exceeds u64".into()))?;

        let embedding_row_id_bytes =
            checked_product(&[chunk, u32_bytes], "CUDA embedding row-ID workspace bytes")?;
        let embedding_output_bytes = checked_product(
            &[chunk, hidden, f32_bytes],
            "CUDA embedding output workspace bytes",
        )?;
        let hidden_norm_bytes = checked_product(
            &[chunk, hidden, f32_bytes, 2],
            "CUDA hidden-norm workspace bytes",
        )?;
        let key_norm_bytes = checked_product(
            &[chunk, kv_heads, head_dim, f32_bytes, 2],
            "CUDA key-norm workspace bytes",
        )?;
        let rope_table_bytes = checked_product(
            &[chunk, rope_half, f32_bytes, 2],
            "CUDA RoPE-table workspace bytes",
        )?;
        let query_gate_bytes = checked_product(
            &[chunk, query_heads, head_dim, f32_bytes, 2],
            "CUDA query/gate workspace bytes",
        )?;
        let paged_gqa_output_bytes = checked_product(
            &[chunk, query_heads, head_dim, f32_bytes],
            "CUDA paged-GQA output bytes",
        )?;
        let total_bytes = [
            embedding_row_id_bytes,
            embedding_output_bytes,
            hidden_norm_bytes,
            key_norm_bytes,
            rope_table_bytes,
            query_gate_bytes,
            paged_gqa_output_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |total, value| {
            checked_add(total, value, "CUDA prefill workspace bytes")
        })?;
        Ok(Self {
            embedding_row_id_bytes,
            embedding_output_bytes,
            hidden_norm_bytes,
            key_norm_bytes,
            rope_table_bytes,
            query_gate_bytes,
            paged_gqa_output_bytes,
            total_bytes,
        })
    }
}

impl CudaDecodeBindingPlan {
    pub fn qwen38(
        schedule: &CudaDecodeSchedule,
        projections: &CudaProjectionPlan,
        config: &Qwen38Config,
    ) -> Result<Self> {
        schedule.validate()?;
        let mut steps = Vec::with_capacity(schedule.steps.len());
        for (schedule_index, step) in schedule.steps.iter().enumerate() {
            steps.push(bind_decode_step(schedule_index, step, projections, config)?);
        }
        let plan = Self { steps };
        plan.validate_complete_ownership(projections, config)?;
        Ok(plan)
    }

    pub fn steps(&self) -> &[CudaBoundDecodeStep] {
        &self.steps
    }

    pub fn resource_count(&self, expected: fn(&CudaPreparedResource) -> bool) -> usize {
        self.steps
            .iter()
            .flat_map(|step| &step.resources)
            .filter(|resource| expected(resource))
            .collect::<BTreeSet<_>>()
            .len()
    }

    fn validate_complete_ownership(
        &self,
        projections: &CudaProjectionPlan,
        config: &Qwen38Config,
    ) -> Result<()> {
        if self.steps.is_empty() {
            return Err(EngineError::InvalidState(
                "CUDA decode binding plan cannot be empty".into(),
            ));
        }
        let resources: BTreeSet<_> = self
            .steps
            .iter()
            .flat_map(|step| step.resources.iter().cloned())
            .collect();
        let bound_projections: BTreeSet<_> = resources
            .iter()
            .filter_map(|resource| match resource {
                CudaPreparedResource::Projection(name) => Some(name.clone()),
                _ => None,
            })
            .collect();
        let expected_projections: BTreeSet<_> =
            projections.projection_groups.keys().cloned().collect();
        compare_resource_set("projection", &bound_projections, &expected_projections)?;

        let bound_activations: BTreeSet<_> = resources
            .iter()
            .filter_map(|resource| match resource {
                CudaPreparedResource::Activation(key) => Some(key.clone()),
                _ => None,
            })
            .collect();
        let expected_activations: BTreeSet<_> = projections
            .groups
            .iter()
            .map(|group| group.key.clone())
            .collect();
        compare_resource_set("activation", &bound_activations, &expected_activations)?;

        let bound_linear_mixers: BTreeSet<_> = resources
            .iter()
            .filter_map(|resource| match resource {
                CudaPreparedResource::LinearMixer(layer) => Some(*layer),
                _ => None,
            })
            .collect();
        let expected_linear_mixers: BTreeSet<_> = (0..config.num_hidden_layers)
            .filter(|layer| config.layer_kind(*layer) == Some(LayerKind::LinearAttention))
            .collect();
        compare_resource_set(
            "linear mixer",
            &bound_linear_mixers,
            &expected_linear_mixers,
        )?;

        let bound_full_attention: BTreeSet<_> = resources
            .iter()
            .filter_map(|resource| match resource {
                CudaPreparedResource::FullAttention(key) => Some(key.clone()),
                _ => None,
            })
            .collect();
        let mut expected_full_attention: BTreeSet<_> = (0..config.num_hidden_layers)
            .filter(|layer| config.layer_kind(*layer) == Some(LayerKind::FullAttention))
            .map(|layer| format!("target:{layer}"))
            .collect();
        expected_full_attention.insert("mtp:0".into());
        compare_resource_set(
            "full attention",
            &bound_full_attention,
            &expected_full_attention,
        )?;

        let bound_regular_norms: BTreeSet<_> = resources
            .iter()
            .filter_map(|resource| match resource {
                CudaPreparedResource::RegularNorm(key) => Some(key.clone()),
                _ => None,
            })
            .collect();
        let expected_regular_norms = BTreeSet::from([
            "target:initial".to_owned(),
            "mtp:pre_embedding".to_owned(),
            "mtp:pre_hidden".to_owned(),
            "mtp:input".to_owned(),
        ]);
        compare_resource_set(
            "regular norm",
            &bound_regular_norms,
            &expected_regular_norms,
        )?;

        let bound_residual_norms: BTreeSet<_> = resources
            .iter()
            .filter_map(|resource| match resource {
                CudaPreparedResource::ResidualNorm(key) => Some(key.clone()),
                _ => None,
            })
            .collect();
        let mut expected_residual_norms = BTreeSet::new();
        for layer in 0..config.num_hidden_layers {
            expected_residual_norms.insert(format!("target:{layer}:post_attention"));
            if layer + 1 == config.num_hidden_layers {
                expected_residual_norms.insert(format!("target:{layer}:post_ffn:final"));
            } else {
                expected_residual_norms
                    .insert(format!("target:{layer}:post_ffn:layer_{}", layer + 1));
            }
        }
        expected_residual_norms.insert("mtp:post_attention".to_owned());
        expected_residual_norms.insert("mtp:final".to_owned());
        compare_resource_set(
            "residual norm",
            &bound_residual_norms,
            &expected_residual_norms,
        )?;

        if !resources.contains(&CudaPreparedResource::Embedding)
            || !resources.contains(&CudaPreparedResource::TokenBarrier)
        {
            return Err(EngineError::InvalidState(
                "CUDA decode binding omits embedding or token barrier".into(),
            ));
        }
        Ok(())
    }
}

impl CudaPrefillBindingPlan {
    pub fn qwen38(
        schedule: &CudaPrefillSchedule,
        projections: &CudaProjectionPlan,
        config: &Qwen38Config,
    ) -> Result<Self> {
        schedule.validate()?;
        let mut steps = Vec::with_capacity(schedule.steps.len());
        for (schedule_index, step) in schedule.steps.iter().enumerate() {
            let previous = schedule_index
                .checked_sub(1)
                .and_then(|index| schedule.steps.get(index));
            steps.push(bind_prefill_step(
                schedule_index,
                step,
                previous,
                projections,
                config,
            )?);
        }
        let plan = Self {
            steps,
            max_chunk_tokens: schedule.max_chunk_tokens,
            projection_workspace: CudaPrefillProjectionWorkspacePlan::qwen38(
                config,
                projections,
                schedule.max_chunk_tokens,
            )?,
        };
        plan.validate_complete_ownership(projections, config)?;
        Ok(plan)
    }

    pub fn steps(&self) -> &[CudaBoundPrefillStep] {
        &self.steps
    }

    pub fn max_chunk_tokens(&self) -> usize {
        self.max_chunk_tokens
    }

    pub fn projection_workspace(&self) -> CudaPrefillProjectionWorkspacePlan {
        self.projection_workspace
    }

    pub fn execution_cursor(
        &self,
        chunk: CudaPrefillChunk,
        committed_tokens: usize,
        admitted_context: usize,
    ) -> Result<CudaPrefillExecutionCursor<'_>> {
        if chunk.token_count == 0 || chunk.token_count > self.max_chunk_tokens {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA prefill chunk has {} tokens, capacity is {}",
                chunk.token_count, self.max_chunk_tokens
            )));
        }
        if chunk.start_position != committed_tokens {
            return Err(EngineError::InvalidState(format!(
                "CUDA prefill chunk starts at {}, but {} tokens are committed",
                chunk.start_position, committed_tokens
            )));
        }
        let end_position = chunk
            .start_position
            .checked_add(chunk.token_count)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA prefill chunk end overflows".into()))?;
        if end_position > admitted_context {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA prefill chunk ends at {end_position}, admitted context is {admitted_context}"
            )));
        }
        if self.steps.last().is_none_or(|step| {
            step.operation != CudaPrefillOperation::ChunkBarrier
                || step.schedule_index + 1 != self.steps.len()
        }) {
            return Err(EngineError::InvalidState(
                "CUDA prefill binding has no sole final chunk barrier".into(),
            ));
        }
        Ok(CudaPrefillExecutionCursor {
            plan: self,
            chunk,
            next_step: 0,
        })
    }

    pub fn resource_count(&self, expected: fn(&CudaPreparedResource) -> bool) -> usize {
        self.steps
            .iter()
            .flat_map(|step| &step.resources)
            .filter(|resource| expected(resource))
            .collect::<BTreeSet<_>>()
            .len()
    }

    fn validate_complete_ownership(
        &self,
        projections: &CudaProjectionPlan,
        config: &Qwen38Config,
    ) -> Result<()> {
        if self.steps.is_empty() {
            return Err(EngineError::InvalidState(
                "CUDA prefill binding plan cannot be empty".into(),
            ));
        }
        let resources: BTreeSet<_> = self
            .steps
            .iter()
            .flat_map(|step| step.resources.iter().cloned())
            .collect();
        validate_complete_resource_ownership("prefill", &resources, projections, config)
    }
}

impl<'a> CudaPrefillExecutionCursor<'a> {
    pub fn chunk(&self) -> CudaPrefillChunk {
        self.chunk
    }

    pub fn next_step(&self) -> Option<&'a CudaBoundPrefillStep> {
        self.plan.steps.get(self.next_step)
    }

    /// Record one successfully dispatched operation. The caller must pass the
    /// identity of the operation it actually launched, after the CUDA driver
    /// accepted that launch.
    pub fn advance(
        &mut self,
        schedule_index: usize,
        layer: Option<usize>,
        operation: CudaPrefillOperation,
    ) -> Result<()> {
        let expected = self.next_step().ok_or_else(|| {
            EngineError::InvalidState("CUDA prefill chunk is already complete".into())
        })?;
        if expected.schedule_index != schedule_index
            || expected.layer != layer
            || expected.operation != operation
        {
            return Err(EngineError::InvalidState(format!(
                "CUDA prefill expected step {} {:?} layer {:?}, received step {schedule_index} {operation:?} layer {layer:?}",
                expected.schedule_index, expected.operation, expected.layer
            )));
        }
        self.next_step += 1;
        Ok(())
    }

    /// Skip only the LM-head read of an intermediate prompt chunk. Target and
    /// optional MTP state still have to execute and the final chunk must use
    /// the logits-producing API.
    pub fn skip_intermediate_lm_head(&mut self) -> Result<()> {
        let expected = self.next_step().ok_or_else(|| {
            EngineError::InvalidState("CUDA prefill chunk is already complete".into())
        })?;
        if expected.layer.is_some() || expected.operation != CudaPrefillOperation::LastTokenLmHead {
            return Err(EngineError::InvalidState(format!(
                "CUDA prefill can skip only an intermediate LM head, next step is {:?} layer {:?}",
                expected.operation, expected.layer
            )));
        }
        self.next_step += 1;
        Ok(())
    }

    /// Records the one schedule step that is intentionally inactive when the
    /// caller requested target-only prefill. No other CUDA operation may be
    /// skipped through this boundary.
    pub fn skip_disabled_mtp(&mut self) -> Result<()> {
        let expected = self.next_step().ok_or_else(|| {
            EngineError::InvalidState("CUDA prefill chunk is already complete".into())
        })?;
        if expected.layer.is_some()
            || expected.operation != CudaPrefillOperation::MtpPrefillCausalScan
        {
            return Err(EngineError::InvalidState(format!(
                "CUDA prefill can skip only disabled MTP, next step is {:?} layer {:?}",
                expected.operation, expected.layer
            )));
        }
        self.next_step += 1;
        Ok(())
    }

    /// Records the MTP causal scan only when it is bound to the exact target
    /// chunk owned by this cursor. The alignment separately proves the
    /// target-one-ahead cache/RoPE contract before mutable MTP state commits.
    pub fn advance_mtp(&mut self, alignment: CudaMtpPrefillAlignment) -> Result<()> {
        if alignment.chunk != self.chunk {
            return Err(EngineError::InvalidState(format!(
                "CUDA MTP alignment {:?} differs from active prefill chunk {:?}",
                alignment.chunk, self.chunk
            )));
        }
        let expected = self.next_step().ok_or_else(|| {
            EngineError::InvalidState("CUDA prefill chunk is already complete".into())
        })?;
        self.advance(
            expected.schedule_index,
            None,
            CudaPrefillOperation::MtpPrefillCausalScan,
        )
    }

    /// Return the new committed token position only after all 645 operations
    /// and the final host-visible barrier have completed in order.
    pub fn finish(self) -> Result<usize> {
        if self.next_step != self.plan.steps.len() {
            return Err(EngineError::InvalidState(format!(
                "CUDA prefill chunk completed {} of {} bound steps",
                self.next_step,
                self.plan.steps.len()
            )));
        }
        self.chunk
            .start_position
            .checked_add(self.chunk.token_count)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA prefill commit overflows".into()))
    }

    pub fn finish_with_mtp(self, alignment: CudaMtpPrefillAlignment) -> Result<(usize, usize)> {
        if alignment.chunk != self.chunk {
            return Err(EngineError::InvalidState(format!(
                "CUDA MTP alignment {:?} differs from completed prefill chunk {:?}",
                alignment.chunk, self.chunk
            )));
        }
        let target_tokens = self.finish()?;
        let expected_target_tokens = alignment
            .committed_mtp_tokens
            .checked_add(1)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA MTP token count overflows".into()))?;
        if target_tokens != expected_target_tokens {
            return Err(EngineError::InvalidState(format!(
                "CUDA MTP prefill completed target/MTP {target_tokens}/{}",
                alignment.committed_mtp_tokens
            )));
        }
        Ok((target_tokens, alignment.committed_mtp_tokens))
    }
}

impl CudaProjectionPlan {
    pub fn qwen38(config: &Qwen38Config) -> Result<Self> {
        let expected = expected_tensor_contract(config);
        let quantized: BTreeSet<String> = expected
            .iter()
            .filter(|(_, spec)| spec.class == TensorClass::QuantizedMatrix)
            .map(|(name, _)| name.clone())
            .collect();
        if !quantized.contains(EMBEDDING_MATRIX) {
            return Err(EngineError::InvalidState(
                "CUDA graph contract is missing the embedding matrix".into(),
            ));
        }

        let mut groups = Vec::new();
        let mut projection_groups = BTreeMap::new();
        for fanout in qwen38_fanout_groups(config) {
            let key = format!("{}:{}", fanout.kind, fanout.prefix);
            let projection_names: Vec<String> = fanout
                .scale_names
                .iter()
                .map(|name| {
                    name.strip_suffix(".s_in")
                        .expect("frozen fan-out scale name has s_in suffix")
                        .to_owned()
                })
                .collect();
            for name in &projection_names {
                if !quantized.contains(name) {
                    return Err(EngineError::InvalidState(format!(
                        "CUDA fan-out group {key} references non-projection {name}"
                    )));
                }
                if projection_groups
                    .insert(name.clone(), key.clone())
                    .is_some()
                {
                    return Err(EngineError::InvalidState(format!(
                        "CUDA projection {name} belongs to multiple activation groups"
                    )));
                }
            }
            groups.push(CudaProjectionGroupPlan {
                key,
                projection_names,
            });
        }

        for name in quantized {
            if name == EMBEDDING_MATRIX || projection_groups.contains_key(&name) {
                continue;
            }
            let key = format!("independent:{name}");
            projection_groups.insert(name.clone(), key.clone());
            groups.push(CudaProjectionGroupPlan {
                key,
                projection_names: vec![name],
            });
        }
        groups.sort_by(|left, right| left.key.cmp(&right.key));

        let plan = Self {
            groups,
            projection_groups,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn groups(&self) -> &[CudaProjectionGroupPlan] {
        &self.groups
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub fn projection_count(&self) -> usize {
        self.projection_groups.len()
    }

    pub fn group_for_projection(&self, name: &str) -> Result<&str> {
        self.projection_groups
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| {
                EngineError::InvalidState(format!(
                    "CUDA projection plan does not contain projection {name}"
                ))
            })
    }

    fn validate(&self) -> Result<()> {
        if self.groups.is_empty() || self.projection_groups.is_empty() {
            return Err(EngineError::InvalidState(
                "CUDA projection graph cannot be empty".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for group in &self.groups {
            if group.key.is_empty() || group.projection_names.is_empty() {
                return Err(EngineError::InvalidState(
                    "CUDA projection group has no key or projections".into(),
                ));
            }
            for name in &group.projection_names {
                if !seen.insert(name.as_str()) {
                    return Err(EngineError::InvalidState(format!(
                        "CUDA projection {name} occurs more than once"
                    )));
                }
                if self.projection_groups.get(name) != Some(&group.key) {
                    return Err(EngineError::InvalidState(format!(
                        "CUDA projection {name} has inconsistent activation ownership"
                    )));
                }
            }
        }
        if seen.len() != self.projection_groups.len() || seen.contains(EMBEDDING_MATRIX) {
            return Err(EngineError::InvalidState(
                "CUDA projection ownership is incomplete or includes embedding".into(),
            ));
        }
        Ok(())
    }
}

fn bind_decode_step(
    schedule_index: usize,
    step: &CudaDecodeStep,
    projections: &CudaProjectionPlan,
    config: &Qwen38Config,
) -> Result<CudaBoundDecodeStep> {
    if step.operation != CudaDecodeOperation::RmsNorm
        && step.operation != CudaDecodeOperation::ResidualRmsNorm
        && step.norm.is_some()
    {
        return Err(EngineError::InvalidState(format!(
            "CUDA decode step {schedule_index} attaches a norm to {:?}",
            step.operation
        )));
    }
    let mut resources = Vec::new();
    match step.operation {
        CudaDecodeOperation::Embedding => {
            require_global_step(schedule_index, step)?;
            resources.push(CudaPreparedResource::Embedding);
        }
        CudaDecodeOperation::RmsNorm => {
            if step.layer != Some(0) || step.norm != Some(CudaNormBinding::LayerInput(0)) {
                return Err(EngineError::InvalidState(format!(
                    "CUDA decode step {schedule_index} is not the frozen initial norm"
                )));
            }
            resources.push(CudaPreparedResource::RegularNorm("target:initial".into()));
        }
        CudaDecodeOperation::FullAttentionFanout => {
            let layer = require_layer_kind(schedule_index, step, config, LayerKind::FullAttention)?;
            let prefix = format!("model.language_model.layers.{layer}.self_attn");
            add_projection_resources(
                projections,
                [
                    format!("{prefix}.q_proj.weight"),
                    format!("{prefix}.k_proj.weight"),
                    format!("{prefix}.v_proj.weight"),
                ],
                &mut resources,
            )?;
        }
        CudaDecodeOperation::QueryGateNormRope
        | CudaDecodeOperation::KeyRope
        | CudaDecodeOperation::PagedKvAppend
        | CudaDecodeOperation::PagedGqa => {
            let layer = require_layer_kind(schedule_index, step, config, LayerKind::FullAttention)?;
            resources.push(CudaPreparedResource::FullAttention(format!(
                "target:{layer}"
            )));
        }
        CudaDecodeOperation::AttentionGateA8OutputProjection => {
            let layer = require_layer_kind(schedule_index, step, config, LayerKind::FullAttention)?;
            let name = format!("model.language_model.layers.{layer}.self_attn.o_proj.weight");
            resources.push(CudaPreparedResource::FullAttention(format!(
                "target:{layer}"
            )));
            add_projection_resources(projections, [name], &mut resources)?;
        }
        CudaDecodeOperation::LinearFanout => {
            let layer =
                require_layer_kind(schedule_index, step, config, LayerKind::LinearAttention)?;
            let prefix = format!("model.language_model.layers.{layer}.linear_attn");
            add_projection_resources(
                projections,
                [
                    format!("{prefix}.in_proj_qkv.weight"),
                    format!("{prefix}.in_proj_z.weight"),
                    format!("{prefix}.in_proj_a.weight"),
                    format!("{prefix}.in_proj_b.weight"),
                ],
                &mut resources,
            )?;
        }
        CudaDecodeOperation::CausalConvolution
        | CudaDecodeOperation::GatedDeltaPrepare
        | CudaDecodeOperation::GatedDeltaRecurrent
        | CudaDecodeOperation::GatedRmsNorm => {
            let layer =
                require_layer_kind(schedule_index, step, config, LayerKind::LinearAttention)?;
            resources.push(CudaPreparedResource::LinearMixer(layer));
        }
        CudaDecodeOperation::LinearOutputProjection => {
            let layer =
                require_layer_kind(schedule_index, step, config, LayerKind::LinearAttention)?;
            resources.push(CudaPreparedResource::LinearMixer(layer));
            add_projection_resources(
                projections,
                [format!(
                    "model.language_model.layers.{layer}.linear_attn.out_proj.weight"
                )],
                &mut resources,
            )?;
        }
        CudaDecodeOperation::ResidualRmsNorm => {
            let layer = require_layer(schedule_index, step, config)?;
            let key = match step.norm {
                Some(CudaNormBinding::LayerPostAttention(bound)) if bound == layer => {
                    format!("target:{layer}:post_attention")
                }
                Some(CudaNormBinding::LayerInput(next)) if next == layer + 1 => {
                    format!("target:{layer}:post_ffn:layer_{next}")
                }
                Some(CudaNormBinding::Final) if layer + 1 == config.num_hidden_layers => {
                    format!("target:{layer}:post_ffn:final")
                }
                _ => {
                    return Err(EngineError::InvalidState(format!(
                        "CUDA residual norm at step {schedule_index} has incompatible binding {:?}",
                        step.norm
                    )))
                }
            };
            resources.push(CudaPreparedResource::ResidualNorm(key));
        }
        CudaDecodeOperation::FfnGateUpFanout => {
            let layer = require_layer(schedule_index, step, config)?;
            let prefix = format!("model.language_model.layers.{layer}.mlp");
            add_projection_resources(
                projections,
                [
                    format!("{prefix}.gate_proj.weight"),
                    format!("{prefix}.up_proj.weight"),
                ],
                &mut resources,
            )?;
        }
        CudaDecodeOperation::SwiGluA8DownProjection => {
            let layer = require_layer(schedule_index, step, config)?;
            add_projection_resources(
                projections,
                [format!(
                    "model.language_model.layers.{layer}.mlp.down_proj.weight"
                )],
                &mut resources,
            )?;
        }
        CudaDecodeOperation::LmHead => {
            require_global_step(schedule_index, step)?;
            add_projection_resources(projections, ["lm_head.weight".to_owned()], &mut resources)?;
        }
        CudaDecodeOperation::MtpDraftAndTargetVerify => {
            require_global_step(schedule_index, step)?;
            resources.extend([
                CudaPreparedResource::Embedding,
                CudaPreparedResource::FullAttention("mtp:0".into()),
                CudaPreparedResource::RegularNorm("mtp:pre_embedding".into()),
                CudaPreparedResource::RegularNorm("mtp:pre_hidden".into()),
                CudaPreparedResource::RegularNorm("mtp:input".into()),
                CudaPreparedResource::ResidualNorm("mtp:post_attention".into()),
                CudaPreparedResource::ResidualNorm("mtp:final".into()),
            ]);
            add_projection_resources(
                projections,
                [
                    "mtp.fc.weight".to_owned(),
                    "mtp.layers.0.self_attn.q_proj.weight".to_owned(),
                    "mtp.layers.0.self_attn.k_proj.weight".to_owned(),
                    "mtp.layers.0.self_attn.v_proj.weight".to_owned(),
                    "mtp.layers.0.self_attn.o_proj.weight".to_owned(),
                    "mtp.layers.0.mlp.gate_proj.weight".to_owned(),
                    "mtp.layers.0.mlp.up_proj.weight".to_owned(),
                    "mtp.layers.0.mlp.down_proj.weight".to_owned(),
                    "lm_head.weight".to_owned(),
                ],
                &mut resources,
            )?;
        }
        CudaDecodeOperation::TokenBarrier => {
            require_global_step(schedule_index, step)?;
            resources.push(CudaPreparedResource::TokenBarrier);
        }
    }
    resources.sort();
    resources.dedup();
    if resources.is_empty() {
        return Err(EngineError::InvalidState(format!(
            "CUDA decode step {schedule_index} has no prepared resource"
        )));
    }
    Ok(CudaBoundDecodeStep {
        schedule_index,
        layer: step.layer,
        operation: step.operation,
        resources,
    })
}

fn bind_prefill_step(
    schedule_index: usize,
    step: &CudaPrefillStep,
    previous: Option<&CudaPrefillStep>,
    projections: &CudaProjectionPlan,
    config: &Qwen38Config,
) -> Result<CudaBoundPrefillStep> {
    let mut resources = Vec::new();
    match step.operation {
        CudaPrefillOperation::EmbeddingBatch => {
            require_global_prefill_step(schedule_index, step)?;
            resources.push(CudaPreparedResource::Embedding);
        }
        CudaPrefillOperation::RmsNormBatch => {
            if step.layer != Some(0) {
                return Err(EngineError::InvalidState(format!(
                    "CUDA prefill RMSNorm step {schedule_index} is not the frozen initial norm"
                )));
            }
            resources.push(CudaPreparedResource::RegularNorm("target:initial".into()));
        }
        CudaPrefillOperation::FullAttentionFanoutBatch => {
            let layer =
                require_prefill_layer_kind(schedule_index, step, config, LayerKind::FullAttention)?;
            let prefix = format!("model.language_model.layers.{layer}.self_attn");
            add_projection_resources(
                projections,
                [
                    format!("{prefix}.q_proj.weight"),
                    format!("{prefix}.k_proj.weight"),
                    format!("{prefix}.v_proj.weight"),
                ],
                &mut resources,
            )?;
        }
        CudaPrefillOperation::QueryGateNormRopeBatch
        | CudaPrefillOperation::KeyRopeBatch
        | CudaPrefillOperation::PagedKvAppendBatch
        | CudaPrefillOperation::PagedGqaCausalScan => {
            let layer =
                require_prefill_layer_kind(schedule_index, step, config, LayerKind::FullAttention)?;
            resources.push(CudaPreparedResource::FullAttention(format!(
                "target:{layer}"
            )));
        }
        CudaPrefillOperation::AttentionGateOutputProjectionBatch => {
            let layer =
                require_prefill_layer_kind(schedule_index, step, config, LayerKind::FullAttention)?;
            resources.push(CudaPreparedResource::FullAttention(format!(
                "target:{layer}"
            )));
            add_projection_resources(
                projections,
                [format!(
                    "model.language_model.layers.{layer}.self_attn.o_proj.weight"
                )],
                &mut resources,
            )?;
        }
        CudaPrefillOperation::LinearFanoutBatch => {
            let layer = require_prefill_layer_kind(
                schedule_index,
                step,
                config,
                LayerKind::LinearAttention,
            )?;
            let prefix = format!("model.language_model.layers.{layer}.linear_attn");
            add_projection_resources(
                projections,
                [
                    format!("{prefix}.in_proj_qkv.weight"),
                    format!("{prefix}.in_proj_z.weight"),
                    format!("{prefix}.in_proj_a.weight"),
                    format!("{prefix}.in_proj_b.weight"),
                ],
                &mut resources,
            )?;
        }
        CudaPrefillOperation::CausalConvolutionScan
        | CudaPrefillOperation::GatedDeltaPrepareBatch
        | CudaPrefillOperation::GatedDeltaCausalScan
        | CudaPrefillOperation::GatedRmsNormBatch => {
            let layer = require_prefill_layer_kind(
                schedule_index,
                step,
                config,
                LayerKind::LinearAttention,
            )?;
            resources.push(CudaPreparedResource::LinearMixer(layer));
        }
        CudaPrefillOperation::LinearOutputProjectionBatch => {
            let layer = require_prefill_layer_kind(
                schedule_index,
                step,
                config,
                LayerKind::LinearAttention,
            )?;
            resources.push(CudaPreparedResource::LinearMixer(layer));
            add_projection_resources(
                projections,
                [format!(
                    "model.language_model.layers.{layer}.linear_attn.out_proj.weight"
                )],
                &mut resources,
            )?;
        }
        CudaPrefillOperation::ResidualRmsNormBatch => {
            let layer = require_prefill_layer(schedule_index, step, config)?;
            let key = match previous.map(|candidate| candidate.operation) {
                Some(CudaPrefillOperation::AttentionGateOutputProjectionBatch)
                | Some(CudaPrefillOperation::LinearOutputProjectionBatch) => {
                    format!("target:{layer}:post_attention")
                }
                Some(CudaPrefillOperation::SwiGluDownProjectionBatch)
                    if layer + 1 == config.num_hidden_layers =>
                {
                    format!("target:{layer}:post_ffn:final")
                }
                Some(CudaPrefillOperation::SwiGluDownProjectionBatch) => {
                    format!("target:{layer}:post_ffn:layer_{}", layer + 1)
                }
                _ => {
                    return Err(EngineError::InvalidState(format!(
                        "CUDA prefill residual norm at step {schedule_index} has no compatible producer"
                    )))
                }
            };
            resources.push(CudaPreparedResource::ResidualNorm(key));
        }
        CudaPrefillOperation::FfnGateUpFanoutBatch => {
            let layer = require_prefill_layer(schedule_index, step, config)?;
            let prefix = format!("model.language_model.layers.{layer}.mlp");
            add_projection_resources(
                projections,
                [
                    format!("{prefix}.gate_proj.weight"),
                    format!("{prefix}.up_proj.weight"),
                ],
                &mut resources,
            )?;
        }
        CudaPrefillOperation::SwiGluDownProjectionBatch => {
            let layer = require_prefill_layer(schedule_index, step, config)?;
            add_projection_resources(
                projections,
                [format!(
                    "model.language_model.layers.{layer}.mlp.down_proj.weight"
                )],
                &mut resources,
            )?;
        }
        CudaPrefillOperation::LastTokenLmHead => {
            require_global_prefill_step(schedule_index, step)?;
            add_projection_resources(projections, ["lm_head.weight".to_owned()], &mut resources)?;
        }
        CudaPrefillOperation::MtpPrefillCausalScan => {
            require_global_prefill_step(schedule_index, step)?;
            resources.extend([
                CudaPreparedResource::Embedding,
                CudaPreparedResource::FullAttention("mtp:0".into()),
                CudaPreparedResource::RegularNorm("mtp:pre_embedding".into()),
                CudaPreparedResource::RegularNorm("mtp:pre_hidden".into()),
                CudaPreparedResource::RegularNorm("mtp:input".into()),
                CudaPreparedResource::ResidualNorm("mtp:post_attention".into()),
                CudaPreparedResource::ResidualNorm("mtp:final".into()),
            ]);
            add_projection_resources(
                projections,
                [
                    "mtp.fc.weight".to_owned(),
                    "mtp.layers.0.self_attn.q_proj.weight".to_owned(),
                    "mtp.layers.0.self_attn.k_proj.weight".to_owned(),
                    "mtp.layers.0.self_attn.v_proj.weight".to_owned(),
                    "mtp.layers.0.self_attn.o_proj.weight".to_owned(),
                    "mtp.layers.0.mlp.gate_proj.weight".to_owned(),
                    "mtp.layers.0.mlp.up_proj.weight".to_owned(),
                    "mtp.layers.0.mlp.down_proj.weight".to_owned(),
                ],
                &mut resources,
            )?;
        }
        CudaPrefillOperation::ChunkBarrier => {
            require_global_prefill_step(schedule_index, step)?;
            resources.push(CudaPreparedResource::TokenBarrier);
        }
    }
    resources.sort();
    resources.dedup();
    if resources.is_empty() {
        return Err(EngineError::InvalidState(format!(
            "CUDA prefill step {schedule_index} has no prepared resource"
        )));
    }
    Ok(CudaBoundPrefillStep {
        schedule_index,
        layer: step.layer,
        operation: step.operation,
        resources,
    })
}

fn add_projection_resources<const N: usize>(
    projections: &CudaProjectionPlan,
    names: [String; N],
    resources: &mut Vec<CudaPreparedResource>,
) -> Result<()> {
    for name in names {
        let group = projections.group_for_projection(&name)?;
        resources.push(CudaPreparedResource::Activation(group.to_owned()));
        resources.push(CudaPreparedResource::Projection(name));
    }
    Ok(())
}

fn require_global_step(schedule_index: usize, step: &CudaDecodeStep) -> Result<()> {
    if step.layer.is_some() || step.norm.is_some() {
        return Err(EngineError::InvalidState(format!(
            "CUDA global decode step {schedule_index} carries a layer or norm"
        )));
    }
    Ok(())
}

fn require_layer(
    schedule_index: usize,
    step: &CudaDecodeStep,
    config: &Qwen38Config,
) -> Result<usize> {
    let layer = step.layer.ok_or_else(|| {
        EngineError::InvalidState(format!("CUDA decode step {schedule_index} has no layer"))
    })?;
    if layer >= config.num_hidden_layers {
        return Err(EngineError::InvalidState(format!(
            "CUDA decode step {schedule_index} references layer {layer}"
        )));
    }
    Ok(layer)
}

fn require_global_prefill_step(schedule_index: usize, step: &CudaPrefillStep) -> Result<()> {
    if step.layer.is_some() {
        return Err(EngineError::InvalidState(format!(
            "CUDA global prefill step {schedule_index} carries a layer"
        )));
    }
    Ok(())
}

fn require_prefill_layer(
    schedule_index: usize,
    step: &CudaPrefillStep,
    config: &Qwen38Config,
) -> Result<usize> {
    let layer = step.layer.ok_or_else(|| {
        EngineError::InvalidState(format!("CUDA prefill step {schedule_index} has no layer"))
    })?;
    if layer >= config.num_hidden_layers {
        return Err(EngineError::InvalidState(format!(
            "CUDA prefill step {schedule_index} references layer {layer}"
        )));
    }
    Ok(layer)
}

fn require_prefill_layer_kind(
    schedule_index: usize,
    step: &CudaPrefillStep,
    config: &Qwen38Config,
    expected: LayerKind,
) -> Result<usize> {
    let layer = require_prefill_layer(schedule_index, step, config)?;
    if config.layer_kind(layer) != Some(expected) {
        return Err(EngineError::InvalidState(format!(
            "CUDA prefill step {schedule_index} binds {expected:?} resources to layer {layer}"
        )));
    }
    Ok(layer)
}

fn validate_complete_resource_ownership(
    program: &str,
    resources: &BTreeSet<CudaPreparedResource>,
    projections: &CudaProjectionPlan,
    config: &Qwen38Config,
) -> Result<()> {
    let bound_projections: BTreeSet<_> = resources
        .iter()
        .filter_map(|resource| match resource {
            CudaPreparedResource::Projection(name) => Some(name.clone()),
            _ => None,
        })
        .collect();
    let expected_projections: BTreeSet<_> = projections.projection_groups.keys().cloned().collect();
    compare_resource_set(
        &format!("{program} projection"),
        &bound_projections,
        &expected_projections,
    )?;

    let bound_activations: BTreeSet<_> = resources
        .iter()
        .filter_map(|resource| match resource {
            CudaPreparedResource::Activation(key) => Some(key.clone()),
            _ => None,
        })
        .collect();
    let expected_activations: BTreeSet<_> = projections
        .groups
        .iter()
        .map(|group| group.key.clone())
        .collect();
    compare_resource_set(
        &format!("{program} activation"),
        &bound_activations,
        &expected_activations,
    )?;

    let bound_linear_mixers: BTreeSet<_> = resources
        .iter()
        .filter_map(|resource| match resource {
            CudaPreparedResource::LinearMixer(layer) => Some(*layer),
            _ => None,
        })
        .collect();
    let expected_linear_mixers: BTreeSet<_> = (0..config.num_hidden_layers)
        .filter(|layer| config.layer_kind(*layer) == Some(LayerKind::LinearAttention))
        .collect();
    compare_resource_set(
        &format!("{program} linear mixer"),
        &bound_linear_mixers,
        &expected_linear_mixers,
    )?;

    let bound_full_attention: BTreeSet<_> = resources
        .iter()
        .filter_map(|resource| match resource {
            CudaPreparedResource::FullAttention(key) => Some(key.clone()),
            _ => None,
        })
        .collect();
    let mut expected_full_attention: BTreeSet<_> = (0..config.num_hidden_layers)
        .filter(|layer| config.layer_kind(*layer) == Some(LayerKind::FullAttention))
        .map(|layer| format!("target:{layer}"))
        .collect();
    expected_full_attention.insert("mtp:0".into());
    compare_resource_set(
        &format!("{program} full attention"),
        &bound_full_attention,
        &expected_full_attention,
    )?;

    let bound_regular_norms: BTreeSet<_> = resources
        .iter()
        .filter_map(|resource| match resource {
            CudaPreparedResource::RegularNorm(key) => Some(key.clone()),
            _ => None,
        })
        .collect();
    let expected_regular_norms = BTreeSet::from([
        "target:initial".to_owned(),
        "mtp:pre_embedding".to_owned(),
        "mtp:pre_hidden".to_owned(),
        "mtp:input".to_owned(),
    ]);
    compare_resource_set(
        &format!("{program} regular norm"),
        &bound_regular_norms,
        &expected_regular_norms,
    )?;

    let bound_residual_norms: BTreeSet<_> = resources
        .iter()
        .filter_map(|resource| match resource {
            CudaPreparedResource::ResidualNorm(key) => Some(key.clone()),
            _ => None,
        })
        .collect();
    let mut expected_residual_norms = BTreeSet::new();
    for layer in 0..config.num_hidden_layers {
        expected_residual_norms.insert(format!("target:{layer}:post_attention"));
        if layer + 1 == config.num_hidden_layers {
            expected_residual_norms.insert(format!("target:{layer}:post_ffn:final"));
        } else {
            expected_residual_norms.insert(format!("target:{layer}:post_ffn:layer_{}", layer + 1));
        }
    }
    expected_residual_norms.insert("mtp:post_attention".to_owned());
    expected_residual_norms.insert("mtp:final".to_owned());
    compare_resource_set(
        &format!("{program} residual norm"),
        &bound_residual_norms,
        &expected_residual_norms,
    )?;

    if !resources.contains(&CudaPreparedResource::Embedding)
        || !resources.contains(&CudaPreparedResource::TokenBarrier)
    {
        return Err(EngineError::InvalidState(format!(
            "CUDA {program} binding omits embedding or token barrier"
        )));
    }
    Ok(())
}

fn require_layer_kind(
    schedule_index: usize,
    step: &CudaDecodeStep,
    config: &Qwen38Config,
    expected: LayerKind,
) -> Result<usize> {
    let layer = require_layer(schedule_index, step, config)?;
    if config.layer_kind(layer) != Some(expected) {
        return Err(EngineError::InvalidState(format!(
            "CUDA decode step {schedule_index} binds {expected:?} resources to layer {layer}"
        )));
    }
    Ok(layer)
}

fn compare_resource_set<T: Ord>(
    label: &str,
    actual: &BTreeSet<T>,
    expected: &BTreeSet<T>,
) -> Result<()> {
    if actual != expected {
        return Err(EngineError::InvalidState(format!(
            "CUDA decode binding owns {} {label} resources, expected {}",
            actual.len(),
            expected.len()
        )));
    }
    Ok(())
}

/// Resident CUDA projection state. `artifact` intentionally keeps the
/// immutable mapping alive, while all accelerator allocations are owned by
/// the prepared objects' private driver context.
pub struct PreparedCudaProjectionGraph {
    artifact: ModelArtifact,
    plan: CudaProjectionPlan,
    decode_bindings: CudaDecodeBindingPlan,
    prefill_bindings: CudaPrefillBindingPlan,
    embedding: PreparedCudaEmbedding,
    activations: BTreeMap<String, PreparedCudaA8Activation>,
    projections: BTreeMap<String, PreparedCudaA8Projection>,
    mtp_draft_projection: Option<PreparedCudaGatheredA8Projection>,
    linear_mixers: BTreeMap<usize, PreparedCudaLinearMixerLayer>,
    full_attention: BTreeMap<String, PreparedCudaFullAttentionLayer>,
    norms: PreparedCudaNormGraph,
    prefill_workspaces: PreparedCudaPrefillWorkspaces,
    mtp_concat: PreparedCudaF32Concat,
    target_hidden_checkpoint: PreparedCudaF32Checkpoint,
    model_bytes: u64,
    graph_bytes: u64,
    session_bytes: u64,
    speculative_checkpoint_bytes: u64,
    target_tokens: usize,
    poisoned: bool,
    mtp_tokens: usize,
    mtp_poisoned: bool,
    speculative_base: Option<CudaSpeculativeBase>,
}

/// One fixed set of full-attention prompt buffers shared by all sixteen target
/// layers and the MTP layer. It intentionally owns no weights, KV pages, or
/// per-layer state.
pub struct PreparedCudaPrefillWorkspaces {
    max_chunk_tokens: usize,
    budget: CudaPrefillWorkspaceBudget,
    projection_budget: CudaPrefillProjectionWorkspacePlan,
    linear_budget: CudaPrefillLinearWorkspaceBudget,
    embedding: PreparedCudaBatchedEmbeddingWorkspace,
    hidden_norm: PreparedCudaBatchedRmsNormWorkspace,
    key_norm: PreparedCudaBatchedRmsNormWorkspace,
    rope: PreparedCudaBatchedRopeWorkspace,
    query_gate: PreparedCudaBatchedQueryGateOutput,
    paged_gqa_output: PreparedCudaPagedGqaPrefillOutput,
    projection_activation: PreparedCudaBatchedA8Workspace,
    projection_outputs: PreparedCudaBatchedA8OutputArena,
    causal_conv_output: PreparedCudaCausalConvScanOutput,
    gated_delta_inputs: PreparedCudaGatedDeltaScanInputs,
    gated_delta_output: PreparedCudaGatedDeltaScanOutput,
    gated_rms_norm_output: PreparedCudaBatchedGatedRmsNormOutput,
}

impl PreparedCudaPrefillWorkspaces {
    fn prepare(
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        max_chunk_tokens: usize,
        maximum_context_tokens: usize,
        projection_budget: CudaPrefillProjectionWorkspacePlan,
        embedding_table: &PreparedCudaEmbedding,
    ) -> Result<Self> {
        let budget = CudaPrefillWorkspaceBudget::qwen38(config, max_chunk_tokens)?;
        let linear_budget = CudaPrefillLinearWorkspaceBudget::qwen38(config, max_chunk_tokens)?;
        if projection_budget.max_chunk_tokens != max_chunk_tokens {
            return Err(EngineError::InvalidState(
                "CUDA attention/projection workspace chunk capacities differ".into(),
            ));
        }
        let embedding =
            runtime.prepare_batched_embedding_workspace(embedding_table, max_chunk_tokens)?;
        let hidden_norm =
            runtime.prepare_batched_rms_norm_workspace(max_chunk_tokens, config.hidden_size)?;
        let key_rows = max_chunk_tokens
            .checked_mul(config.num_key_value_heads)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA key-norm rows overflow".into()))?;
        let key_norm = runtime.prepare_batched_rms_norm_workspace(key_rows, config.head_dim)?;
        let rope_config = CudaPartialRopeConfig {
            heads: config.num_attention_heads,
            head_dim: config.head_dim,
            rotary_dim: config.rotary_dim,
            theta: config.rope_theta,
        };
        let rope = runtime.prepare_batched_rope_workspace(rope_config, max_chunk_tokens)?;
        let query_gate = runtime
            .prepare_batched_query_gate_output(CudaQueryGateConfig::QWEN38_27B, max_chunk_tokens)?;
        let paged_gqa_output = runtime.prepare_paged_q2q4_gqa_prefill_output(
            CudaPagedGqaConfig {
                query_heads: config.num_attention_heads,
                key_value_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                maximum_tokens: maximum_context_tokens,
                page_tokens: DEFAULT_KV_PAGE_TOKENS,
                sink_tokens: DEFAULT_KV_SINK_TOKENS,
                recent_tokens: DEFAULT_KV_RECENT_TOKENS,
            },
            max_chunk_tokens,
        )?;
        let projection_activation = runtime
            .prepare_batched_a8_workspace(max_chunk_tokens, projection_budget.activation_columns)?;
        let projection_outputs = runtime.prepare_batched_a8_output_arena(
            max_chunk_tokens,
            projection_budget.output_slot_rows,
        )?;
        let causal_conv_output = runtime
            .prepare_causal_conv_scan_output(CudaCausalConvConfig::QWEN38_27B, max_chunk_tokens)?;
        let gated_delta_inputs = runtime.prepare_gated_delta_scan_inputs(max_chunk_tokens)?;
        let gated_delta_output = runtime
            .prepare_gated_delta_scan_output(CudaGatedDeltaConfig::QWEN38_27B, max_chunk_tokens)?;
        let gated_rms_norm_output = runtime.prepare_batched_gated_rms_norm_output(
            CudaGatedRmsNormConfig::QWEN38_27B,
            max_chunk_tokens,
        )?;
        let actual = [
            embedding.transient_bytes(),
            hidden_norm.transient_bytes(),
            key_norm.transient_bytes(),
            rope.transient_bytes(),
            query_gate.transient_bytes(),
            paged_gqa_output.transient_bytes(),
            projection_activation.transient_bytes(),
            projection_outputs.transient_bytes(),
            causal_conv_output.transient_bytes(),
            gated_delta_inputs.transient_bytes(),
            gated_delta_output.transient_bytes(),
            gated_rms_norm_output.transient_bytes(),
        ]
        .into_iter()
        .try_fold(0_u64, |total, bytes| {
            let bytes = u64::try_from(bytes).map_err(|_| {
                EngineError::MemoryBudget("CUDA prefill workspace allocation exceeds u64".into())
            })?;
            checked_add(total, bytes, "CUDA prefill workspace allocation")
        })?;
        let planned = checked_add(
            checked_add(
                budget.total_bytes,
                projection_budget.total_bytes,
                "CUDA attention/projection workspace bytes",
            )?,
            linear_budget.total_bytes,
            "CUDA complete prefill workspace bytes",
        )?;
        if actual != planned {
            return Err(EngineError::InvalidState(format!(
                "CUDA prefill workspace allocated {actual} bytes, planned {planned}"
            )));
        }
        Ok(Self {
            max_chunk_tokens,
            budget,
            projection_budget,
            linear_budget,
            embedding,
            hidden_norm,
            key_norm,
            rope,
            query_gate,
            paged_gqa_output,
            projection_activation,
            projection_outputs,
            causal_conv_output,
            gated_delta_inputs,
            gated_delta_output,
            gated_rms_norm_output,
        })
    }

    pub fn max_chunk_tokens(&self) -> usize {
        self.max_chunk_tokens
    }

    pub fn budget(&self) -> CudaPrefillWorkspaceBudget {
        self.budget
    }

    pub fn projection_budget(&self) -> CudaPrefillProjectionWorkspacePlan {
        self.projection_budget
    }

    pub fn linear_budget(&self) -> CudaPrefillLinearWorkspaceBudget {
        self.linear_budget
    }

    pub fn transient_bytes(&self) -> u64 {
        self.budget.total_bytes
            + self.projection_budget.total_bytes
            + self.linear_budget.total_bytes
    }

    pub fn embedding(&self) -> &PreparedCudaBatchedEmbeddingWorkspace {
        &self.embedding
    }

    pub fn hidden_norm(&self) -> &PreparedCudaBatchedRmsNormWorkspace {
        &self.hidden_norm
    }

    pub fn key_norm(&self) -> &PreparedCudaBatchedRmsNormWorkspace {
        &self.key_norm
    }

    pub fn rope(&self) -> &PreparedCudaBatchedRopeWorkspace {
        &self.rope
    }

    pub fn query_gate(&self) -> &PreparedCudaBatchedQueryGateOutput {
        &self.query_gate
    }

    pub fn paged_gqa_output(&self) -> &PreparedCudaPagedGqaPrefillOutput {
        &self.paged_gqa_output
    }

    pub fn projection_activation(&self) -> &PreparedCudaBatchedA8Workspace {
        &self.projection_activation
    }

    pub fn projection_outputs(&self) -> &PreparedCudaBatchedA8OutputArena {
        &self.projection_outputs
    }

    pub fn causal_conv_output(&self) -> &PreparedCudaCausalConvScanOutput {
        &self.causal_conv_output
    }

    pub fn gated_delta_inputs(&self) -> &PreparedCudaGatedDeltaScanInputs {
        &self.gated_delta_inputs
    }

    pub fn gated_delta_output(&self) -> &PreparedCudaGatedDeltaScanOutput {
        &self.gated_delta_output
    }

    pub fn gated_rms_norm_output(&self) -> &PreparedCudaBatchedGatedRmsNormOutput {
        &self.gated_rms_norm_output
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CudaSpeculativeBase {
    target_tokens: usize,
    mtp_tokens: usize,
}

pub struct PreparedCudaNormGraph {
    regular: BTreeMap<String, PreparedCudaRmsNorm>,
    residual: BTreeMap<String, PreparedCudaResidualRmsNorm>,
    model_bytes: u64,
    graph_bytes: u64,
}

impl PreparedCudaNormGraph {
    pub fn regular_count(&self) -> usize {
        self.regular.len()
    }

    pub fn residual_count(&self) -> usize {
        self.residual.len()
    }

    pub fn regular(&self, key: &str) -> Result<&PreparedCudaRmsNorm> {
        self.regular
            .get(key)
            .ok_or_else(|| EngineError::InvalidState(format!("prepared CUDA norm {key} not found")))
    }

    pub fn residual(&self, key: &str) -> Result<&PreparedCudaResidualRmsNorm> {
        self.residual.get(key).ok_or_else(|| {
            EngineError::InvalidState(format!("prepared CUDA residual norm {key} not found"))
        })
    }

    pub fn model_bytes(&self) -> u64 {
        self.model_bytes
    }

    pub fn graph_bytes(&self) -> u64 {
        self.graph_bytes
    }
}

pub struct PreparedCudaFullAttentionLayer {
    key: String,
    query_gate: PreparedCudaQueryGate,
    key_norm: PreparedCudaRmsNorm,
    key_rope: PreparedCudaPartialRope,
    kv: PreparedCudaPagedGqa,
    model_bytes: u64,
    graph_bytes: u64,
    session_bytes: u64,
}

impl PreparedCudaFullAttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        key: &str,
        q_norm_weight_f16_le: &[u8],
        k_norm_weight_f16_le: &[u8],
        maximum_context_tokens: usize,
    ) -> Result<Self> {
        if config != &Qwen38Config::default() {
            return Err(EngineError::Shape(
                "CUDA full-attention preparation requires the frozen Qwen3.8-27B topology".into(),
            ));
        }
        let query_gate = runtime.prepare_query_gate_norm_rope_f32(
            CudaQueryGateConfig::QWEN38_27B,
            q_norm_weight_f16_le,
        )?;
        let key_norm = runtime.prepare_qwen_rms_norm_f16(
            CudaRmsNormConfig {
                rows: config.num_key_value_heads,
                columns: config.head_dim,
                epsilon: config.rms_norm_epsilon,
            },
            k_norm_weight_f16_le,
        )?;
        let key_rope = runtime.prepare_partial_rope_f32(CudaPartialRopeConfig {
            heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rotary_dim: config.rotary_dim,
            theta: config.rope_theta,
        })?;
        let kv = runtime.prepare_paged_q2q4_gqa(CudaPagedGqaConfig {
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            maximum_tokens: maximum_context_tokens,
            page_tokens: DEFAULT_KV_PAGE_TOKENS,
            sink_tokens: DEFAULT_KV_SINK_TOKENS,
            recent_tokens: DEFAULT_KV_RECENT_TOKENS,
        })?;
        let model_bytes = sum_usize(
            [query_gate.model_bytes(), key_norm.model_bytes()],
            "CUDA full-attention model bytes",
        )?;
        let graph_bytes = sum_usize(
            [
                query_gate.transient_bytes(),
                key_norm.transient_bytes(),
                key_rope.transient_bytes(),
                kv.transient_bytes(),
            ],
            "CUDA full-attention graph bytes",
        )?;
        let session_bytes = u64::try_from(kv.packed_device_bytes())
            .map_err(|_| EngineError::MemoryBudget("CUDA packed KV bytes exceed u64".into()))?;
        Ok(Self {
            key: key.to_owned(),
            query_gate,
            key_norm,
            key_rope,
            kv,
            model_bytes,
            graph_bytes,
            session_bytes,
        })
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn query_gate(&self) -> &PreparedCudaQueryGate {
        &self.query_gate
    }

    pub fn key_norm(&self) -> &PreparedCudaRmsNorm {
        &self.key_norm
    }

    pub fn key_rope(&self) -> &PreparedCudaPartialRope {
        &self.key_rope
    }

    pub fn kv_mut(&mut self) -> &mut PreparedCudaPagedGqa {
        &mut self.kv
    }

    pub fn model_bytes(&self) -> u64 {
        self.model_bytes
    }

    pub fn graph_bytes(&self) -> u64 {
        self.graph_bytes
    }

    pub fn session_bytes(&self) -> u64 {
        self.session_bytes
    }

    pub fn reset(&mut self) -> Result<()> {
        self.kv.reset()
    }

    fn begin_speculative(&mut self) -> Result<()> {
        self.kv.begin_speculative()
    }

    fn restore_speculative(&mut self) -> Result<()> {
        self.kv.restore_speculative()
    }

    fn commit_speculative(&mut self) -> Result<()> {
        self.kv.commit_speculative()
    }

    pub fn dispatch_device<'a>(
        &'a mut self,
        runtime: &CudaCandidateRuntime,
        query_gate_input: CudaDeviceF32View<'_>,
        key_input: CudaDeviceF32View<'_>,
        value_input: CudaDeviceF32View<'_>,
        position: u64,
    ) -> Result<(CudaDeviceF32View<'a>, CudaDeviceF32View<'a>)> {
        let Self {
            query_gate,
            key_norm,
            key_rope,
            kv,
            ..
        } = self;
        query_gate.write_position(position)?;
        key_rope.write_position(position)?;
        let (query, gate) =
            runtime.dispatch_query_gate_norm_rope_device(query_gate, query_gate_input)?;
        let key = runtime.dispatch_qwen_rms_norm_f16_device(key_norm, key_input)?;
        let key = runtime.dispatch_partial_rope_f32_device(key_rope, key)?;
        let attention =
            runtime.append_and_dispatch_paged_q2q4_gqa_device(kv, query, key, value_input)?;
        Ok((attention, gate))
    }

    /// Advances one complete token-major prompt chunk through Qwen's exact
    /// full-attention frontend and canonical mixed Q2/Q4 KV cache. All output
    /// storage is borrowed from the one graph-owned prefill pool.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_prefill_device<'a>(
        &mut self,
        runtime: &CudaCandidateRuntime,
        key_norm_workspace: &'a PreparedCudaBatchedRmsNormWorkspace,
        rope_workspace: &'a PreparedCudaBatchedRopeWorkspace,
        query_gate_output: &'a PreparedCudaBatchedQueryGateOutput,
        attention_output: &'a PreparedCudaPagedGqaPrefillOutput,
        query_gate_input: CudaDeviceF32View<'_>,
        key_input: CudaDeviceF32View<'_>,
        value_input: CudaDeviceF32View<'_>,
        start_position: usize,
        tokens: usize,
    ) -> Result<(CudaDeviceF32View<'a>, CudaDeviceF32View<'a>)> {
        self.dispatch_prefill_with_positions_device(
            runtime,
            key_norm_workspace,
            rope_workspace,
            query_gate_output,
            attention_output,
            query_gate_input,
            key_input,
            value_input,
            start_position,
            start_position,
            tokens,
        )
    }

    /// Variant used by MTP prefill, whose first cached token corresponds to
    /// absolute RoPE position one. Target prefill passes identical cache and
    /// RoPE starts through [`Self::dispatch_prefill_device`].
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_prefill_with_positions_device<'a>(
        &mut self,
        runtime: &CudaCandidateRuntime,
        key_norm_workspace: &'a PreparedCudaBatchedRmsNormWorkspace,
        rope_workspace: &'a PreparedCudaBatchedRopeWorkspace,
        query_gate_output: &'a PreparedCudaBatchedQueryGateOutput,
        attention_output: &'a PreparedCudaPagedGqaPrefillOutput,
        query_gate_input: CudaDeviceF32View<'_>,
        key_input: CudaDeviceF32View<'_>,
        value_input: CudaDeviceF32View<'_>,
        cache_start_token: usize,
        rope_start_position: usize,
        tokens: usize,
    ) -> Result<(CudaDeviceF32View<'a>, CudaDeviceF32View<'a>)> {
        if self.kv.tokens() != cache_start_token {
            return Err(EngineError::InvalidState(format!(
                "CUDA full-attention prefill cache starts at {cache_start_token}, but its KV cache has {} tokens",
                self.kv.tokens()
            )));
        }
        let Self {
            query_gate,
            key_norm,
            kv,
            ..
        } = self;
        runtime.write_batched_rope_positions(rope_workspace, rope_start_position as u64, tokens)?;
        let (query, gate) = runtime.dispatch_batched_query_gate_norm_rope_with_table_device(
            query_gate,
            rope_workspace,
            query_gate_output,
            query_gate_input,
            tokens,
        )?;
        let config = Qwen38Config::default();
        let key_rows = tokens
            .checked_mul(config.num_key_value_heads)
            .ok_or_else(|| EngineError::Shape("CUDA batched key rows overflow".into()))?;
        let key = runtime.dispatch_batched_qwen_rms_norm_f16_device(
            key_norm,
            key_norm_workspace,
            key_input,
            key_rows,
        )?;
        let key = runtime.dispatch_batched_partial_rope_with_table_f32_device(
            rope_workspace,
            CudaPartialRopeConfig {
                heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                rotary_dim: config.rotary_dim,
                theta: config.rope_theta,
            },
            key,
            tokens,
        )?;
        let attention = runtime.append_and_dispatch_paged_q2q4_gqa_prefill_device(
            kv,
            attention_output,
            query,
            key,
            value_input,
            tokens,
        )?;
        Ok((attention, gate))
    }
}

pub struct PreparedCudaLinearMixerLayer {
    layer: usize,
    convolution: PreparedCudaCausalConv,
    inputs: PreparedCudaGatedDeltaInputs,
    recurrence: PreparedCudaGatedDelta,
    norm: PreparedCudaGatedRmsNorm,
    model_bytes: u64,
    graph_bytes: u64,
    session_bytes: u64,
    speculative_checkpoint_bytes: u64,
}

impl PreparedCudaLinearMixerLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        runtime: &CudaCandidateRuntime,
        layer: usize,
        convolution_weight_f16_le: &[u8],
        a_log_f32_le: &[u8],
        dt_bias_f32_le: &[u8],
        norm_weight_f16_le: &[u8],
    ) -> Result<Self> {
        let convolution = runtime
            .prepare_causal_conv_f16(CudaCausalConvConfig::QWEN38_27B, convolution_weight_f16_le)?;
        let inputs = runtime.prepare_gated_delta_inputs_f32_le(a_log_f32_le, dt_bias_f32_le)?;
        let recurrence = runtime.prepare_gated_delta_f16(CudaGatedDeltaConfig::QWEN38_27B)?;
        let norm = runtime
            .prepare_gated_rms_norm_f16(CudaGatedRmsNormConfig::QWEN38_27B, norm_weight_f16_le)?;
        let model_bytes = sum_usize(
            [
                convolution.model_bytes(),
                inputs.model_bytes(),
                norm.model_bytes(),
            ],
            "CUDA linear mixer model bytes",
        )?;
        let graph_bytes = sum_usize(
            [
                convolution.transient_bytes(),
                inputs.transient_bytes(),
                recurrence.transient_bytes(),
                norm.transient_bytes(),
            ],
            "CUDA linear mixer graph bytes",
        )?;
        let speculative_checkpoint_bytes = sum_usize(
            [
                convolution.speculative_checkpoint_bytes(),
                recurrence.speculative_checkpoint_bytes(),
            ],
            "CUDA linear checkpoint bytes",
        )?;
        let session_bytes = sum_usize(
            [
                convolution.resident_state_bytes(),
                recurrence.resident_state_bytes(),
                convolution.speculative_checkpoint_bytes(),
                recurrence.speculative_checkpoint_bytes(),
            ],
            "CUDA linear mixer session bytes",
        )?;
        Ok(Self {
            layer,
            convolution,
            inputs,
            recurrence,
            norm,
            model_bytes,
            graph_bytes,
            session_bytes,
            speculative_checkpoint_bytes,
        })
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn model_bytes(&self) -> u64 {
        self.model_bytes
    }

    pub fn graph_bytes(&self) -> u64 {
        self.graph_bytes
    }

    pub fn session_bytes(&self) -> u64 {
        self.session_bytes
    }

    pub fn speculative_checkpoint_bytes(&self) -> u64 {
        self.speculative_checkpoint_bytes
    }

    fn begin_speculative(&mut self) -> Result<()> {
        self.convolution.begin_speculative()?;
        if let Err(error) = self.recurrence.begin_speculative() {
            self.convolution.commit_speculative()?;
            return Err(error);
        }
        Ok(())
    }

    fn restore_speculative(&mut self) -> Result<()> {
        self.convolution.restore_speculative()?;
        self.recurrence.restore_speculative()
    }

    fn commit_speculative(&mut self) -> Result<()> {
        self.convolution.commit_speculative()?;
        self.recurrence.commit_speculative()
    }

    pub fn convolution_mut(&mut self) -> &mut PreparedCudaCausalConv {
        &mut self.convolution
    }

    pub fn inputs_mut(&mut self) -> &mut PreparedCudaGatedDeltaInputs {
        &mut self.inputs
    }

    pub fn recurrence_mut(&mut self) -> &mut PreparedCudaGatedDelta {
        &mut self.recurrence
    }

    pub fn norm(&self) -> &PreparedCudaGatedRmsNorm {
        &self.norm
    }

    pub fn dispatch_device<'a>(
        &'a mut self,
        runtime: &CudaCandidateRuntime,
        mixed_qkv: CudaDeviceF32View<'_>,
        gate: CudaDeviceF32View<'_>,
        raw_a: CudaDeviceF32View<'_>,
        raw_b: CudaDeviceF32View<'_>,
    ) -> Result<CudaDeviceF32View<'a>> {
        let Self {
            convolution,
            inputs,
            recurrence,
            norm,
            ..
        } = self;
        let convolved = runtime.dispatch_causal_conv_f16_device(convolution, mixed_qkv)?;
        let value_width = recurrence
            .config()
            .heads
            .checked_mul(recurrence.config().value_dim)
            .ok_or_else(|| EngineError::Shape("CUDA linear value width overflows".into()))?;
        let value_offset = convolved.values().checked_sub(value_width).ok_or_else(|| {
            EngineError::Shape("CUDA convolved QKV is narrower than its value tail".into())
        })?;
        let value = convolved.slice(value_offset, value_width)?;
        let prepared =
            runtime.dispatch_gated_delta_inputs_device(inputs, convolved, raw_a, raw_b)?;
        let mixed = runtime.dispatch_gated_delta_f16_device(
            recurrence,
            prepared.query,
            prepared.key,
            value,
            prepared.log_decay,
            prepared.beta,
        )?;
        runtime.dispatch_gated_rms_norm_f16_device(norm, mixed, gate)
    }

    /// Advances one complete token-major prompt chunk through the exact
    /// Qwen3.8 linear-attention mixer. The caller supplies graph-owned,
    /// reusable workspaces so this path performs no per-chunk allocation and
    /// never stages the recurrent tensors through host memory.
    ///
    /// This is deliberately a launch-only layer primitive: the enclosing
    /// graph submission owns the single chunk commit barrier and fail-closed
    /// reset policy. Standalone callers must reset the layer after an error
    /// because either recurrent state may have advanced.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_prefill_device<'a>(
        &mut self,
        runtime: &CudaCandidateRuntime,
        convolution_output: &'a PreparedCudaCausalConvScanOutput,
        input_workspace: &'a PreparedCudaGatedDeltaScanInputs,
        recurrence_output: &'a PreparedCudaGatedDeltaScanOutput,
        norm_output: &'a PreparedCudaBatchedGatedRmsNormOutput,
        mixed_qkv: CudaDeviceF32View<'_>,
        gate: CudaDeviceF32View<'_>,
        raw_a: CudaDeviceF32View<'_>,
        raw_b: CudaDeviceF32View<'_>,
        tokens: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        let Self {
            convolution,
            inputs,
            recurrence,
            norm,
            ..
        } = self;
        let convolved = runtime.dispatch_causal_conv_f16_scan_device(
            convolution,
            convolution_output,
            mixed_qkv,
            tokens,
        )?;
        let prepared = runtime.dispatch_gated_delta_scan_inputs_device(
            inputs,
            input_workspace,
            convolved,
            raw_a,
            raw_b,
            tokens,
        )?;
        let mixed = runtime.dispatch_gated_delta_f16_scan_device(
            recurrence,
            recurrence_output,
            prepared.query,
            prepared.key,
            prepared.value,
            prepared.log_decay,
            prepared.beta,
            tokens,
        )?;
        runtime.dispatch_batched_gated_rms_norm_f16_device(norm, norm_output, mixed, gate, tokens)
    }
}

impl PreparedCudaProjectionGraph {
    pub fn prepare(
        runtime: &CudaCandidateRuntime,
        artifact: &ModelArtifact,
        config: &Qwen38Config,
        maximum_context_tokens: usize,
        mtp_draft_token_ids: Option<&[u32]>,
    ) -> Result<Self> {
        if maximum_context_tokens < DEFAULT_KV_SINK_TOKENS + DEFAULT_KV_RECENT_TOKENS
            || maximum_context_tokens > config.max_position_embeddings
        {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA context {maximum_context_tokens} must be between {} and {} tokens",
                DEFAULT_KV_SINK_TOKENS + DEFAULT_KV_RECENT_TOKENS,
                config.max_position_embeddings
            )));
        }
        validate_tensor_contract(artifact.manifest(), config)?;
        let plan = CudaProjectionPlan::qwen38(config)?;
        let decode_schedule = CudaDecodeSchedule::qwen38(config)?;
        let decode_bindings = CudaDecodeBindingPlan::qwen38(&decode_schedule, &plan, config)?;
        let prefill_schedule = CudaPrefillSchedule::qwen38(config, CUDA_PREFILL_CHUNK_TOKENS)?;
        let prefill_bindings = CudaPrefillBindingPlan::qwen38(&prefill_schedule, &plan, config)?;
        let mut activations = BTreeMap::new();
        let mut projections = BTreeMap::new();
        let mut linear_mixers = BTreeMap::new();
        let mut full_attention = BTreeMap::new();
        let mut model_bytes = 0_u64;
        let mut graph_bytes = 0_u64;
        let mut session_bytes = 0_u64;
        let mut speculative_checkpoint_bytes = 0_u64;

        let embedding_view = artifact.recovered_matrix(EMBEDDING_MATRIX)?;
        let embedding = runtime.prepare_embedding_recovered(embedding_view)?;
        model_bytes = checked_add(
            model_bytes,
            u64::try_from(embedding.model_bytes()).map_err(|_| {
                EngineError::MemoryBudget("CUDA embedding model bytes exceed u64".into())
            })?,
            "CUDA model bytes",
        )?;
        graph_bytes = checked_add(
            graph_bytes,
            u64::try_from(embedding.graph_bytes()).map_err(|_| {
                EngineError::MemoryBudget("CUDA embedding graph bytes exceed u64".into())
            })?,
            "CUDA graph bytes",
        )?;

        for group in plan.groups() {
            let first_name = group.projection_names.first().ok_or_else(|| {
                EngineError::InvalidState(format!("CUDA projection group {} is empty", group.key))
            })?;
            let first = artifact.recovered_matrix(first_name)?;
            let first_s_in = packed_f16(first.s_in.as_recovery_scales()?)?;
            for name in group.projection_names.iter().skip(1) {
                let candidate = artifact.recovered_matrix(name)?;
                let candidate_s_in = packed_f16(candidate.s_in.as_recovery_scales()?)?;
                if candidate.matrix.columns != first.matrix.columns || candidate_s_in != first_s_in
                {
                    return Err(EngineError::InvalidArtifact(format!(
                        "CUDA activation group {} has non-identical s_in at {name}",
                        group.key
                    )));
                }
            }
            let activation = runtime.prepare_shared_a8_activation_recovered(first)?;
            let activation_model_bytes = scale_bytes(first.matrix.columns, "CUDA s_in")?;
            let activation_graph_bytes = u64::try_from(activation.resident_bytes())
                .map_err(|_| EngineError::MemoryBudget("CUDA activation bytes exceed u64".into()))?
                .checked_sub(activation_model_bytes)
                .ok_or_else(|| {
                    EngineError::InvalidState(
                        "CUDA activation residency is below its recovery scale bytes".into(),
                    )
                })?;
            model_bytes = checked_add(model_bytes, activation_model_bytes, "CUDA model bytes")?;
            graph_bytes = checked_add(graph_bytes, activation_graph_bytes, "CUDA graph bytes")?;
            if activations.insert(group.key.clone(), activation).is_some() {
                return Err(EngineError::InvalidState(format!(
                    "duplicate CUDA activation group {}",
                    group.key
                )));
            }

            for name in &group.projection_names {
                let recovered = artifact.recovered_matrix(name)?;
                let projection = runtime
                    .prepare_shared_a8_projection_recovered(recovered, Activation::Identity)?;
                let projection_model_bytes = u64::try_from(recovered.matrix.weights.len())
                    .map_err(|_| {
                        EngineError::MemoryBudget("CUDA projection weights exceed u64".into())
                    })?
                    .checked_add(scale_bytes(recovered.matrix.rows, "CUDA s_out")?)
                    .ok_or_else(|| {
                        EngineError::MemoryBudget("CUDA projection model bytes overflow".into())
                    })?;
                let projection_resident =
                    u64::try_from(projection.resident_bytes()).map_err(|_| {
                        EngineError::MemoryBudget("CUDA projection residency exceeds u64".into())
                    })?;
                let projection_graph_bytes = projection_resident
                    .checked_sub(projection_model_bytes)
                    .ok_or_else(|| {
                        EngineError::InvalidState(
                            "CUDA projection residency is below immutable model bytes".into(),
                        )
                    })?;
                model_bytes = checked_add(model_bytes, projection_model_bytes, "CUDA model bytes")?;
                graph_bytes = checked_add(graph_bytes, projection_graph_bytes, "CUDA graph bytes")?;
                if projections.insert(name.clone(), projection).is_some() {
                    return Err(EngineError::InvalidState(format!(
                        "duplicate prepared CUDA projection {name}"
                    )));
                }
            }
        }

        if activations.len() != plan.group_count() || projections.len() != plan.projection_count() {
            return Err(EngineError::InvalidState(format!(
                "prepared CUDA graph has {}/{} activation groups and {}/{} projections",
                activations.len(),
                plan.group_count(),
                projections.len(),
                plan.projection_count()
            )));
        }

        for layer in 0..config.num_hidden_layers {
            if config.layer_kind(layer) != Some(LayerKind::LinearAttention) {
                continue;
            }
            let prefix = format!("model.language_model.layers.{layer}.linear_attn");
            let mixer = PreparedCudaLinearMixerLayer::prepare(
                runtime,
                layer,
                artifact_f16(artifact, &format!("{prefix}.conv1d.weight"))?,
                artifact_f32(artifact, &format!("{prefix}.A_log"))?,
                artifact_f32(artifact, &format!("{prefix}.dt_bias"))?,
                artifact_f16(artifact, &format!("{prefix}.norm.weight"))?,
            )?;
            speculative_checkpoint_bytes = checked_add(
                speculative_checkpoint_bytes,
                mixer.speculative_checkpoint_bytes,
                "CUDA speculative checkpoint bytes",
            )?;
            model_bytes = checked_add(model_bytes, mixer.model_bytes, "CUDA model bytes")?;
            graph_bytes = checked_add(graph_bytes, mixer.graph_bytes, "CUDA graph bytes")?;
            session_bytes = checked_add(session_bytes, mixer.session_bytes, "CUDA session bytes")?;
            if linear_mixers.insert(layer, mixer).is_some() {
                return Err(EngineError::InvalidState(format!(
                    "duplicate CUDA linear mixer layer {layer}"
                )));
            }
        }
        if linear_mixers.len() != config.linear_attention_layers() {
            return Err(EngineError::InvalidState(format!(
                "prepared CUDA graph has {} linear mixers, expected {}",
                linear_mixers.len(),
                config.linear_attention_layers()
            )));
        }

        for layer in 0..config.num_hidden_layers {
            if config.layer_kind(layer) != Some(LayerKind::FullAttention) {
                continue;
            }
            let key = format!("target:{layer}");
            let prefix = format!("model.language_model.layers.{layer}.self_attn");
            let prepared = prepare_full_attention(
                runtime,
                artifact,
                config,
                &key,
                &prefix,
                maximum_context_tokens,
            )?;
            model_bytes = checked_add(model_bytes, prepared.model_bytes, "CUDA model bytes")?;
            graph_bytes = checked_add(graph_bytes, prepared.graph_bytes, "CUDA graph bytes")?;
            session_bytes =
                checked_add(session_bytes, prepared.session_bytes, "CUDA session bytes")?;
            if full_attention.insert(key.clone(), prepared).is_some() {
                return Err(EngineError::InvalidState(format!(
                    "duplicate CUDA full-attention layer {key}"
                )));
            }
        }
        let mtp_key = "mtp:0".to_owned();
        let mtp = prepare_full_attention(
            runtime,
            artifact,
            config,
            &mtp_key,
            "mtp.layers.0.self_attn",
            maximum_context_tokens,
        )?;
        model_bytes = checked_add(model_bytes, mtp.model_bytes, "CUDA model bytes")?;
        graph_bytes = checked_add(graph_bytes, mtp.graph_bytes, "CUDA graph bytes")?;
        session_bytes = checked_add(session_bytes, mtp.session_bytes, "CUDA session bytes")?;
        full_attention.insert(mtp_key, mtp);
        let expected_full_attention = config
            .full_attention_layers()
            .checked_add(config.mtp_num_hidden_layers)
            .ok_or_else(|| EngineError::Shape("CUDA full-attention count overflows".into()))?;
        if full_attention.len() != expected_full_attention {
            return Err(EngineError::InvalidState(format!(
                "prepared CUDA graph has {} full-attention states, expected {expected_full_attention}",
                full_attention.len()
            )));
        }
        let norms = prepare_norms(runtime, artifact, config)?;
        model_bytes = checked_add(model_bytes, norms.model_bytes, "CUDA model bytes")?;
        graph_bytes = checked_add(graph_bytes, norms.graph_bytes, "CUDA graph bytes")?;
        let prefill_workspaces = PreparedCudaPrefillWorkspaces::prepare(
            runtime,
            config,
            prefill_bindings.max_chunk_tokens(),
            maximum_context_tokens,
            prefill_bindings.projection_workspace(),
            &embedding,
        )?;
        graph_bytes = checked_add(
            graph_bytes,
            prefill_workspaces.transient_bytes(),
            "CUDA graph bytes",
        )?;
        let mtp_concat = runtime.prepare_f32_concat(config.hidden_size, config.hidden_size)?;
        graph_bytes = checked_add(
            graph_bytes,
            u64::try_from(mtp_concat.transient_bytes())
                .map_err(|_| EngineError::MemoryBudget("CUDA MTP concat exceeds u64".into()))?,
            "CUDA graph bytes",
        )?;
        let target_hidden_checkpoint = runtime.prepare_f32_checkpoint(config.hidden_size)?;
        session_bytes = checked_add(
            session_bytes,
            u64::try_from(target_hidden_checkpoint.resident_bytes()).map_err(|_| {
                EngineError::MemoryBudget("CUDA target-hidden checkpoint exceeds u64".into())
            })?,
            "CUDA session bytes",
        )?;
        speculative_checkpoint_bytes = checked_add(
            speculative_checkpoint_bytes,
            u64::try_from(target_hidden_checkpoint.resident_bytes()).map_err(|_| {
                EngineError::MemoryBudget("CUDA target-hidden checkpoint exceeds u64".into())
            })?,
            "CUDA speculative checkpoint bytes",
        )?;
        let mtp_draft_projection = mtp_draft_token_ids
            .map(|row_ids| {
                let lm_head = projections.get("lm_head.weight").ok_or_else(|| {
                    EngineError::InvalidState("CUDA LM head is not resident".into())
                })?;
                runtime.prepare_gathered_a8_projection(lm_head, row_ids)
            })
            .transpose()?;
        if let Some(gathered) = &mtp_draft_projection {
            graph_bytes = checked_add(
                graph_bytes,
                u64::try_from(gathered.resident_bytes()).map_err(|_| {
                    EngineError::MemoryBudget("CUDA gathered LM-head bytes exceed u64".into())
                })?,
                "CUDA graph bytes",
            )?;
        }
        let graph = Self {
            artifact: artifact.clone(),
            plan,
            decode_bindings,
            prefill_bindings,
            embedding,
            activations,
            projections,
            mtp_draft_projection,
            linear_mixers,
            full_attention,
            norms,
            prefill_workspaces,
            mtp_concat,
            target_hidden_checkpoint,
            model_bytes,
            graph_bytes,
            session_bytes,
            speculative_checkpoint_bytes,
            target_tokens: 0,
            poisoned: false,
            mtp_tokens: 0,
            mtp_poisoned: false,
            speculative_base: None,
        };
        graph.validate_bound_resources()?;
        Ok(graph)
    }

    pub fn artifact_manifest_sha256(&self) -> &str {
        self.artifact.manifest_sha256()
    }

    pub fn plan(&self) -> &CudaProjectionPlan {
        &self.plan
    }

    pub fn decode_bindings(&self) -> &CudaDecodeBindingPlan {
        &self.decode_bindings
    }

    pub fn prefill_bindings(&self) -> &CudaPrefillBindingPlan {
        &self.prefill_bindings
    }

    pub fn prefill_workspaces(&self) -> &PreparedCudaPrefillWorkspaces {
        &self.prefill_workspaces
    }

    pub fn embedding(&self) -> &PreparedCudaEmbedding {
        &self.embedding
    }

    pub fn embedding_s_out(&self, row: usize) -> Result<f32> {
        let recovered = self.artifact.recovered_matrix(EMBEDDING_MATRIX)?;
        if row >= recovered.matrix.rows {
            return Err(EngineError::Shape(format!(
                "CUDA embedding row {row} exceeds {}",
                recovered.matrix.rows
            )));
        }
        recovered.s_out.value(row)
    }

    pub fn activation(&self, key: &str) -> Result<&PreparedCudaA8Activation> {
        self.activations.get(key).ok_or_else(|| {
            EngineError::InvalidState(format!("prepared CUDA activation {key} not found"))
        })
    }

    pub fn projection(&self, name: &str) -> Result<&PreparedCudaA8Projection> {
        self.projections.get(name).ok_or_else(|| {
            EngineError::InvalidState(format!("prepared CUDA projection {name} not found"))
        })
    }

    /// Executes one frozen prompt-chunk projection fan-out through the graph-
    /// owned maximum-width A8 and four-slot output arenas. The returned views
    /// borrow the graph, preventing a later arena overwrite while they remain
    /// live.
    pub fn dispatch_prefill_projection_fanout_device<'a, const N: usize>(
        &'a self,
        runtime: &CudaCandidateRuntime,
        input: CudaDeviceF32View<'_>,
        tokens: usize,
        names: [String; N],
    ) -> Result<[CudaDeviceF32View<'a>; N]> {
        let first = names.first().ok_or_else(|| {
            EngineError::InvalidState("CUDA prefill projection fan-out cannot be empty".into())
        })?;
        let group = self.plan.group_for_projection(first)?;
        let activation = self.activation(group)?;
        let mut prepared = Vec::with_capacity(N);
        for name in &names {
            if self.plan.group_for_projection(name)? != group {
                return Err(EngineError::InvalidState(format!(
                    "CUDA prefill projection fan-out mixes activation groups at {name}"
                )));
            }
            prepared.push((
                self.projection(name)?,
                prefill_projection_output_slot(name)?,
            ));
        }
        runtime
            .dispatch_batched_a8_arena_fanout_device(
                activation,
                self.prefill_workspaces.projection_activation(),
                self.prefill_workspaces.projection_outputs(),
                input,
                tokens,
                &prepared,
            )?
            .try_into()
            .map_err(|_| EngineError::InvalidState("CUDA prefill fan-out count changed".into()))
    }

    /// Closes the full-attention chunk edge from paged GQA plus query gate to
    /// the resident output projection without materializing a gated tensor.
    pub fn dispatch_prefill_attention_gate_projection_device<'a>(
        &'a self,
        runtime: &CudaCandidateRuntime,
        attention: CudaDeviceF32View<'_>,
        gate: CudaDeviceF32View<'_>,
        tokens: usize,
        name: &str,
    ) -> Result<CudaDeviceF32View<'a>> {
        let group = self.plan.group_for_projection(name)?;
        let activation = self.activation(group)?;
        let projection = self.projection(name)?;
        let slot = prefill_projection_output_slot(name)?;
        runtime
            .dispatch_batched_a8_arena_sigmoid_gate_fanout_device(
                activation,
                self.prefill_workspaces.projection_activation(),
                self.prefill_workspaces.projection_outputs(),
                attention,
                gate,
                tokens,
                &[(projection, slot)],
            )?
            .pop()
            .ok_or_else(|| {
                EngineError::InvalidState("CUDA prefill attention projection has no output".into())
            })
    }

    /// Closes the FFN chunk edge from gate/up projections to the resident down
    /// projection through the same graph-owned arenas.
    pub fn dispatch_prefill_swiglu_projection_device<'a>(
        &'a self,
        runtime: &CudaCandidateRuntime,
        gate: CudaDeviceF32View<'_>,
        up: CudaDeviceF32View<'_>,
        tokens: usize,
        name: &str,
    ) -> Result<CudaDeviceF32View<'a>> {
        let group = self.plan.group_for_projection(name)?;
        let activation = self.activation(group)?;
        let projection = self.projection(name)?;
        let slot = prefill_projection_output_slot(name)?;
        runtime
            .dispatch_batched_a8_arena_swiglu_fanout_device(
                activation,
                self.prefill_workspaces.projection_activation(),
                self.prefill_workspaces.projection_outputs(),
                gate,
                up,
                tokens,
                &[(projection, slot)],
            )?
            .pop()
            .ok_or_else(|| {
                EngineError::InvalidState("CUDA prefill SwiGLU projection has no output".into())
            })
    }

    pub fn mtp_draft_rows(&self) -> Option<usize> {
        self.mtp_draft_projection
            .as_ref()
            .map(PreparedCudaGatheredA8Projection::rows)
    }

    pub fn linear_mixer_count(&self) -> usize {
        self.linear_mixers.len()
    }

    pub fn linear_mixer(&self, layer: usize) -> Result<&PreparedCudaLinearMixerLayer> {
        self.linear_mixers.get(&layer).ok_or_else(|| {
            EngineError::InvalidState(format!("prepared CUDA linear mixer {layer} not found"))
        })
    }

    pub fn linear_mixer_mut(&mut self, layer: usize) -> Result<&mut PreparedCudaLinearMixerLayer> {
        self.linear_mixers.get_mut(&layer).ok_or_else(|| {
            EngineError::InvalidState(format!("prepared CUDA linear mixer {layer} not found"))
        })
    }

    pub fn full_attention_count(&self) -> usize {
        self.full_attention.len()
    }

    pub fn full_attention(&self, key: &str) -> Result<&PreparedCudaFullAttentionLayer> {
        self.full_attention.get(key).ok_or_else(|| {
            EngineError::InvalidState(format!("prepared CUDA full attention {key} not found"))
        })
    }

    pub fn full_attention_mut(&mut self, key: &str) -> Result<&mut PreparedCudaFullAttentionLayer> {
        self.full_attention.get_mut(key).ok_or_else(|| {
            EngineError::InvalidState(format!("prepared CUDA full attention {key} not found"))
        })
    }

    pub fn norms(&self) -> &PreparedCudaNormGraph {
        &self.norms
    }

    pub fn model_bytes(&self) -> u64 {
        self.model_bytes
    }

    pub fn graph_bytes(&self) -> u64 {
        self.graph_bytes
    }

    pub fn session_bytes(&self) -> u64 {
        self.session_bytes
    }

    pub fn speculative_checkpoint_bytes(&self) -> u64 {
        self.speculative_checkpoint_bytes
    }

    pub fn target_tokens(&self) -> usize {
        self.target_tokens
    }

    pub fn speculative_branch_active(&self) -> bool {
        self.speculative_base.is_some()
    }

    /// Preserve exactly the mutable state required to verify a chained MTP
    /// block. Linear recurrence/convolution state is copied once on device;
    /// paged KV retains its boundary slot and snapshots metadata only. The
    /// caller must either commit the complete branch or restore and replay the
    /// causally accepted prefix.
    pub fn begin_speculative_branch(&mut self, runtime: &CudaCandidateRuntime) -> Result<()> {
        if self.poisoned || self.mtp_poisoned || self.speculative_base.is_some() {
            return Err(EngineError::InvalidState(
                "CUDA speculative checkpoint requires a healthy graph without a pending branch"
                    .into(),
            ));
        }
        if self.target_tokens == 0 || self.target_tokens != self.mtp_tokens.saturating_add(1) {
            return Err(EngineError::InvalidState(format!(
                "CUDA speculative checkpoint requires target state one token ahead of MTP, observed {}/{}",
                self.target_tokens, self.mtp_tokens
            )));
        }
        let final_target_norm = format!(
            "target:{}:post_ffn:final",
            Qwen38Config::default().num_hidden_layers - 1
        );
        let final_hidden = self
            .norms
            .residual(&final_target_norm)?
            .normalized_output()?;
        runtime.snapshot_f32_device(&mut self.target_hidden_checkpoint, final_hidden)?;

        let mut begun_mixers = Vec::new();
        for (layer, mixer) in &mut self.linear_mixers {
            if let Err(error) = mixer.begin_speculative() {
                for begun in begun_mixers {
                    self.linear_mixers
                        .get_mut(&begun)
                        .expect("recorded CUDA mixer")
                        .commit_speculative()?;
                }
                self.target_hidden_checkpoint.commit()?;
                return Err(error);
            }
            begun_mixers.push(*layer);
        }

        let mut begun_attention = Vec::new();
        for (key, attention) in &mut self.full_attention {
            if let Err(error) = attention.begin_speculative() {
                for begun in begun_attention {
                    self.full_attention
                        .get_mut(&begun)
                        .expect("recorded CUDA attention")
                        .commit_speculative()?;
                }
                for mixer in self.linear_mixers.values_mut() {
                    mixer.commit_speculative()?;
                }
                self.target_hidden_checkpoint.commit()?;
                return Err(error);
            }
            begun_attention.push(key.clone());
        }
        self.speculative_base = Some(CudaSpeculativeBase {
            target_tokens: self.target_tokens,
            mtp_tokens: self.mtp_tokens,
        });
        Ok(())
    }

    /// Restore the pre-branch target/MTP state. The executor then replays only
    /// the accepted causal prefix through the ordinary committed token path.
    pub fn restore_speculative_branch(&mut self, runtime: &CudaCandidateRuntime) -> Result<()> {
        let base = self.speculative_base.take().ok_or_else(|| {
            EngineError::InvalidState("CUDA graph has no pending speculative branch".into())
        })?;
        let restored = (|| {
            for mixer in self.linear_mixers.values_mut() {
                mixer.restore_speculative()?;
            }
            for attention in self.full_attention.values_mut() {
                attention.restore_speculative()?;
            }
            let final_target_norm = format!(
                "target:{}:post_ffn:final",
                Qwen38Config::default().num_hidden_layers - 1
            );
            let destination = self
                .norms
                .residual(&final_target_norm)?
                .normalized_output()?;
            runtime.restore_f32_device(&mut self.target_hidden_checkpoint, destination)
        })();
        if let Err(error) = restored {
            self.poisoned = true;
            self.mtp_poisoned = true;
            return Err(error);
        }
        self.target_tokens = base.target_tokens;
        self.mtp_tokens = base.mtp_tokens;
        self.poisoned = false;
        self.mtp_poisoned = false;
        Ok(())
    }

    /// Keep a completely accepted speculative branch and discard only its
    /// bounded replay checkpoint.
    pub fn commit_speculative_branch(&mut self) -> Result<()> {
        if self.target_tokens != self.mtp_tokens.saturating_add(1) {
            return Err(EngineError::InvalidState(format!(
                "CUDA speculative commit requires target exactly one token ahead, observed {}/{}",
                self.target_tokens, self.mtp_tokens
            )));
        }
        self.speculative_base.take().ok_or_else(|| {
            EngineError::InvalidState("CUDA graph has no pending speculative branch".into())
        })?;
        let committed = (|| {
            for mixer in self.linear_mixers.values_mut() {
                mixer.commit_speculative()?;
            }
            for attention in self.full_attention.values_mut() {
                attention.commit_speculative()?;
            }
            self.target_hidden_checkpoint.commit()
        })();
        if let Err(error) = committed {
            self.poisoned = true;
            self.mtp_poisoned = true;
            return Err(error);
        }
        Ok(())
    }

    pub fn reset_session(&mut self) -> Result<()> {
        for mixer in self.linear_mixers.values_mut() {
            mixer.convolution.reset()?;
            mixer.recurrence.reset()?;
        }
        for attention in self.full_attention.values_mut() {
            attention.reset()?;
        }
        self.target_hidden_checkpoint.clear()?;
        self.target_tokens = 0;
        self.poisoned = false;
        self.mtp_tokens = 0;
        self.mtp_poisoned = false;
        self.speculative_base = None;
        Ok(())
    }

    /// Executes one target-only prompt chunk through the frozen 645-step
    /// program. MTP is the sole explicitly disabled schedule step.
    pub fn dispatch_target_prefill_chunk_without_mtp_device<'a>(
        &'a mut self,
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        token_ids: &[u32],
        start_position: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        self.dispatch_target_prefill_chunk_device_impl(
            runtime,
            config,
            token_ids,
            start_position,
            false,
            true,
        )?
        .ok_or_else(|| {
            EngineError::InvalidState("CUDA final prefill omitted LM-head logits".into())
        })
    }

    /// Advances a non-final target-only prompt chunk without reading the
    /// complete LM head. All target session state commits exactly as in the
    /// logits-producing path.
    pub fn dispatch_target_prefill_state_without_mtp_device(
        &mut self,
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        token_ids: &[u32],
        start_position: usize,
    ) -> Result<()> {
        let logits = self.dispatch_target_prefill_chunk_device_impl(
            runtime,
            config,
            token_ids,
            start_position,
            false,
            false,
        )?;
        if logits.is_some() {
            return Err(EngineError::InvalidState(
                "CUDA state-only prefill unexpectedly produced logits".into(),
            ));
        }
        Ok(())
    }

    /// Executes target and the causally shifted one-layer MTP prompt state in
    /// one chunk transaction. Target remains exactly one token ahead of MTP.
    pub fn dispatch_target_prefill_chunk_with_mtp_device<'a>(
        &'a mut self,
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        token_ids: &[u32],
        start_position: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        self.dispatch_target_prefill_chunk_device_impl(
            runtime,
            config,
            token_ids,
            start_position,
            true,
            true,
        )?
        .ok_or_else(|| {
            EngineError::InvalidState("CUDA final MTP prefill omitted LM-head logits".into())
        })
    }

    /// Advances a non-final target+MTP prompt chunk while omitting the target
    /// LM head. The retained hidden boundary and both causal caches still
    /// commit atomically.
    pub fn dispatch_target_prefill_state_with_mtp_device(
        &mut self,
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        token_ids: &[u32],
        start_position: usize,
    ) -> Result<()> {
        let logits = self.dispatch_target_prefill_chunk_device_impl(
            runtime,
            config,
            token_ids,
            start_position,
            true,
            false,
        )?;
        if logits.is_some() {
            return Err(EngineError::InvalidState(
                "CUDA state-only MTP prefill unexpectedly produced logits".into(),
            ));
        }
        Ok(())
    }

    fn dispatch_target_prefill_chunk_device_impl<'a>(
        &'a mut self,
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        token_ids: &[u32],
        start_position: usize,
        mtp_enabled: bool,
        emit_logits: bool,
    ) -> Result<Option<CudaDeviceF32View<'a>>> {
        if config != &Qwen38Config::default() {
            return Err(EngineError::Shape(
                "CUDA target prefill requires the frozen Qwen3.8-27B topology".into(),
            ));
        }
        if self.poisoned || (mtp_enabled && self.mtp_poisoned) || self.speculative_base.is_some() {
            return Err(EngineError::InvalidState(
                "CUDA target prefill requires a healthy graph without a speculative branch".into(),
            ));
        }
        if token_ids.is_empty()
            || token_ids.len() > self.prefill_workspaces.max_chunk_tokens()
            || token_ids
                .iter()
                .any(|token| (*token as usize) >= self.embedding.rows())
        {
            return Err(EngineError::Shape(
                "CUDA target prefill token chunk is empty, oversized, or out of vocabulary".into(),
            ));
        }
        let chunk = CudaPrefillChunk {
            start_position,
            token_count: token_ids.len(),
        };
        let mtp_alignment = mtp_enabled
            .then(|| CudaMtpPrefillAlignment::qwen38(chunk, self.target_tokens, self.mtp_tokens))
            .transpose()?;
        let Self {
            plan,
            prefill_bindings,
            embedding,
            activations,
            projections,
            linear_mixers,
            full_attention,
            norms,
            prefill_workspaces,
            target_tokens,
            poisoned,
            mtp_tokens,
            mtp_poisoned,
            ..
        } = self;
        let mut cursor = prefill_bindings.execution_cursor(
            chunk,
            *target_tokens,
            config.max_position_embeddings,
        )?;
        *poisoned = true;
        if mtp_enabled {
            *mtp_poisoned = true;
        }
        let tokens = token_ids.len();
        let dispatch = runtime.run_token_submission(
            "target prefill chunk commit synchronization",
            || -> Result<Option<CudaDeviceF32View<'a>>> {
                let embedded = runtime.dispatch_embedding_rows_device(
                    embedding,
                    prefill_workspaces.embedding(),
                    token_ids,
                )?;
                advance_prefill(&mut cursor, None, CudaPrefillOperation::EmbeddingBatch)?;
                let mut residual = embedded;
                let mut normalized = runtime.dispatch_batched_qwen_rms_norm_f16_device(
                    norms.regular("target:initial")?,
                    prefill_workspaces.hidden_norm(),
                    residual,
                    tokens,
                )?;
                advance_prefill(&mut cursor, Some(0), CudaPrefillOperation::RmsNormBatch)?;

                for layer in 0..config.num_hidden_layers {
                    let prefix = format!("model.language_model.layers.{layer}");
                    let mixer_output = match config.layer_kind(layer).expect("frozen layer") {
                        LayerKind::FullAttention => {
                            let [query_gate, key, value] =
                                dispatch_prefill_projection_fanout_parts(
                                    runtime,
                                    plan,
                                    activations,
                                    projections,
                                    prefill_workspaces,
                                    normalized,
                                    tokens,
                                    [
                                        format!("{prefix}.self_attn.q_proj.weight"),
                                        format!("{prefix}.self_attn.k_proj.weight"),
                                        format!("{prefix}.self_attn.v_proj.weight"),
                                    ],
                                )?;
                            advance_prefill(
                                &mut cursor,
                                Some(layer),
                                CudaPrefillOperation::FullAttentionFanoutBatch,
                            )?;
                            let attention = full_attention
                                .get_mut(&format!("target:{layer}"))
                                .ok_or_else(|| {
                                    EngineError::InvalidState(format!(
                                        "CUDA full-attention layer {layer} is not resident"
                                    ))
                                })?;
                            let (attention_output, gate) = attention.dispatch_prefill_device(
                                runtime,
                                prefill_workspaces.key_norm(),
                                prefill_workspaces.rope(),
                                prefill_workspaces.query_gate(),
                                prefill_workspaces.paged_gqa_output(),
                                query_gate,
                                key,
                                value,
                                start_position,
                                tokens,
                            )?;
                            for operation in [
                                CudaPrefillOperation::QueryGateNormRopeBatch,
                                CudaPrefillOperation::KeyRopeBatch,
                                CudaPrefillOperation::PagedKvAppendBatch,
                                CudaPrefillOperation::PagedGqaCausalScan,
                            ] {
                                advance_prefill(&mut cursor, Some(layer), operation)?;
                            }
                            let output = dispatch_prefill_fused_projection_parts(
                                runtime,
                                plan,
                                activations,
                                projections,
                                prefill_workspaces,
                                attention_output,
                                gate,
                                tokens,
                                &format!("{prefix}.self_attn.o_proj.weight"),
                                PrefillFusedEdge::AttentionGate,
                            )?;
                            advance_prefill(
                                &mut cursor,
                                Some(layer),
                                CudaPrefillOperation::AttentionGateOutputProjectionBatch,
                            )?;
                            output
                        }
                        LayerKind::LinearAttention => {
                            let [mixed_qkv, gate, raw_a, raw_b] =
                                dispatch_prefill_projection_fanout_parts(
                                    runtime,
                                    plan,
                                    activations,
                                    projections,
                                    prefill_workspaces,
                                    normalized,
                                    tokens,
                                    [
                                        format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                                        format!("{prefix}.linear_attn.in_proj_z.weight"),
                                        format!("{prefix}.linear_attn.in_proj_a.weight"),
                                        format!("{prefix}.linear_attn.in_proj_b.weight"),
                                    ],
                                )?;
                            advance_prefill(
                                &mut cursor,
                                Some(layer),
                                CudaPrefillOperation::LinearFanoutBatch,
                            )?;
                            let mixer = linear_mixers.get_mut(&layer).ok_or_else(|| {
                                EngineError::InvalidState(format!(
                                    "CUDA linear-attention layer {layer} is not resident"
                                ))
                            })?;
                            let mixed = mixer.dispatch_prefill_device(
                                runtime,
                                prefill_workspaces.causal_conv_output(),
                                prefill_workspaces.gated_delta_inputs(),
                                prefill_workspaces.gated_delta_output(),
                                prefill_workspaces.gated_rms_norm_output(),
                                mixed_qkv,
                                gate,
                                raw_a,
                                raw_b,
                                tokens,
                            )?;
                            for operation in [
                                CudaPrefillOperation::CausalConvolutionScan,
                                CudaPrefillOperation::GatedDeltaPrepareBatch,
                                CudaPrefillOperation::GatedDeltaCausalScan,
                                CudaPrefillOperation::GatedRmsNormBatch,
                            ] {
                                advance_prefill(&mut cursor, Some(layer), operation)?;
                            }
                            let [output] = dispatch_prefill_projection_fanout_parts(
                                runtime,
                                plan,
                                activations,
                                projections,
                                prefill_workspaces,
                                mixed,
                                tokens,
                                [format!("{prefix}.linear_attn.out_proj.weight")],
                            )?;
                            advance_prefill(
                                &mut cursor,
                                Some(layer),
                                CudaPrefillOperation::LinearOutputProjectionBatch,
                            )?;
                            output
                        }
                    };

                    (residual, normalized) = runtime
                        .dispatch_batched_residual_rms_norm_f16_device(
                            norms.residual(&format!("target:{layer}:post_attention"))?,
                            prefill_workspaces.hidden_norm(),
                            residual,
                            mixer_output,
                            tokens,
                        )?;
                    advance_prefill(
                        &mut cursor,
                        Some(layer),
                        CudaPrefillOperation::ResidualRmsNormBatch,
                    )?;

                    let [gate, up] = dispatch_prefill_projection_fanout_parts(
                        runtime,
                        plan,
                        activations,
                        projections,
                        prefill_workspaces,
                        normalized,
                        tokens,
                        [
                            format!("{prefix}.mlp.gate_proj.weight"),
                            format!("{prefix}.mlp.up_proj.weight"),
                        ],
                    )?;
                    advance_prefill(
                        &mut cursor,
                        Some(layer),
                        CudaPrefillOperation::FfnGateUpFanoutBatch,
                    )?;
                    let down = dispatch_prefill_fused_projection_parts(
                        runtime,
                        plan,
                        activations,
                        projections,
                        prefill_workspaces,
                        gate,
                        up,
                        tokens,
                        &format!("{prefix}.mlp.down_proj.weight"),
                        PrefillFusedEdge::SwiGlu,
                    )?;
                    advance_prefill(
                        &mut cursor,
                        Some(layer),
                        CudaPrefillOperation::SwiGluDownProjectionBatch,
                    )?;
                    let post_ffn = if layer + 1 == config.num_hidden_layers {
                        format!("target:{layer}:post_ffn:final")
                    } else {
                        format!("target:{layer}:post_ffn:layer_{}", layer + 1)
                    };
                    (residual, normalized) = runtime
                        .dispatch_batched_residual_rms_norm_f16_device(
                            norms.residual(&post_ffn)?,
                            prefill_workspaces.hidden_norm(),
                            residual,
                            down,
                            tokens,
                        )?;
                    advance_prefill(
                        &mut cursor,
                        Some(layer),
                        CudaPrefillOperation::ResidualRmsNormBatch,
                    )?;
                }

                let logits = if emit_logits {
                    let last_row = normalized.slice(
                        (tokens - 1)
                            .checked_mul(config.hidden_size)
                            .ok_or_else(|| {
                                EngineError::Shape("CUDA final prompt row offset overflows".into())
                            })?,
                        config.hidden_size,
                    )?;
                    let logits = dispatch_projection_device(
                        runtime,
                        plan,
                        activations,
                        projections,
                        last_row,
                        "lm_head.weight",
                    )?;
                    advance_prefill(&mut cursor, None, CudaPrefillOperation::LastTokenLmHead)?;
                    Some(logits)
                } else {
                    cursor.skip_intermediate_lm_head()?;
                    None
                };
                if let Some(alignment) = mtp_alignment {
                    dispatch_mtp_prefill_parts(
                        runtime,
                        config,
                        plan,
                        activations,
                        projections,
                        full_attention,
                        norms,
                        prefill_workspaces,
                        embedded,
                        normalized,
                        alignment,
                    )?;
                    cursor.advance_mtp(alignment)?;
                } else {
                    cursor.skip_disabled_mtp()?;
                }
                Ok(logits)
            },
        );
        let logits = dispatch?;
        advance_prefill(&mut cursor, None, CudaPrefillOperation::ChunkBarrier)?;
        if let Some(alignment) = mtp_alignment {
            (*target_tokens, *mtp_tokens) = cursor.finish_with_mtp(alignment)?;
            *mtp_poisoned = false;
        } else {
            *target_tokens = cursor.finish()?;
        }
        *poisoned = false;
        Ok(logits)
    }

    pub fn dispatch_target_token_device<'a>(
        &'a mut self,
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        token_id: usize,
        position: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        if config != &Qwen38Config::default() {
            return Err(EngineError::Shape(
                "CUDA target dispatch requires the frozen Qwen3.8-27B topology".into(),
            ));
        }
        if self.poisoned {
            return Err(EngineError::InvalidState(
                "CUDA target session is poisoned; reset is required".into(),
            ));
        }
        if position != self.target_tokens {
            return Err(EngineError::InvalidState(format!(
                "CUDA target position {position} does not match committed token count {}",
                self.target_tokens
            )));
        }
        if token_id >= self.embedding.rows() {
            return Err(EngineError::Shape(format!(
                "CUDA token {token_id} exceeds embedding vocabulary {}",
                self.embedding.rows()
            )));
        }
        let embedding_scale = self.embedding_s_out(token_id)?;

        let Self {
            plan,
            embedding,
            activations,
            projections,
            linear_mixers,
            full_attention,
            norms,
            target_tokens,
            poisoned,
            ..
        } = self;
        *poisoned = true;
        let logits = runtime.run_token_submission("target token commit synchronization", || {
            let embedded =
                runtime.dispatch_embedding_row_device(embedding, token_id, embedding_scale)?;
            let mut residual = embedded;
            let mut normalized = runtime
                .dispatch_qwen_rms_norm_f16_device(norms.regular("target:initial")?, residual)?;

            for layer in 0..config.num_hidden_layers {
                let prefix = format!("model.language_model.layers.{layer}");
                let mixer_output = match config.layer_kind(layer).expect("frozen layer") {
                    LayerKind::FullAttention => {
                        let [query_gate, key, value] = dispatch_projection_fanout_device(
                            runtime,
                            plan,
                            activations,
                            projections,
                            normalized,
                            [
                                format!("{prefix}.self_attn.q_proj.weight"),
                                format!("{prefix}.self_attn.k_proj.weight"),
                                format!("{prefix}.self_attn.v_proj.weight"),
                            ],
                        )?;
                        let attention = full_attention
                            .get_mut(&format!("target:{layer}"))
                            .ok_or_else(|| {
                                EngineError::InvalidState(format!(
                                    "CUDA full-attention layer {layer} is not resident"
                                ))
                            })?;
                        let (attention_output, gate) = attention.dispatch_device(
                            runtime,
                            query_gate,
                            key,
                            value,
                            position as u64,
                        )?;
                        dispatch_sigmoid_gate_projection_device(
                            runtime,
                            plan,
                            activations,
                            projections,
                            attention_output,
                            gate,
                            &format!("{prefix}.self_attn.o_proj.weight"),
                        )?
                    }
                    LayerKind::LinearAttention => {
                        let [mixed_qkv, gate, raw_a, raw_b] = dispatch_projection_fanout_device(
                            runtime,
                            plan,
                            activations,
                            projections,
                            normalized,
                            [
                                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                                format!("{prefix}.linear_attn.in_proj_z.weight"),
                                format!("{prefix}.linear_attn.in_proj_a.weight"),
                                format!("{prefix}.linear_attn.in_proj_b.weight"),
                            ],
                        )?;
                        let mixer = linear_mixers.get_mut(&layer).ok_or_else(|| {
                            EngineError::InvalidState(format!(
                                "CUDA linear-attention layer {layer} is not resident"
                            ))
                        })?;
                        let mixed =
                            mixer.dispatch_device(runtime, mixed_qkv, gate, raw_a, raw_b)?;
                        dispatch_projection_device(
                            runtime,
                            plan,
                            activations,
                            projections,
                            mixed,
                            &format!("{prefix}.linear_attn.out_proj.weight"),
                        )?
                    }
                };

                let post_attention_key = format!("target:{layer}:post_attention");
                let post_attention = norms.residual(&post_attention_key)?;
                (residual, normalized) = runtime.dispatch_residual_rms_norm_f16_device(
                    post_attention,
                    residual,
                    mixer_output,
                )?;

                let [gate, up] = dispatch_projection_fanout_device(
                    runtime,
                    plan,
                    activations,
                    projections,
                    normalized,
                    [
                        format!("{prefix}.mlp.gate_proj.weight"),
                        format!("{prefix}.mlp.up_proj.weight"),
                    ],
                )?;
                let down = dispatch_swiglu_projection_device(
                    runtime,
                    plan,
                    activations,
                    projections,
                    gate,
                    up,
                    &format!("{prefix}.mlp.down_proj.weight"),
                )?;
                let post_ffn_key = if layer + 1 == config.num_hidden_layers {
                    format!("target:{layer}:post_ffn:final")
                } else {
                    format!("target:{layer}:post_ffn:layer_{}", layer + 1)
                };
                (residual, normalized) = runtime.dispatch_residual_rms_norm_f16_device(
                    norms.residual(&post_ffn_key)?,
                    residual,
                    down,
                )?;
            }

            dispatch_projection_device(
                runtime,
                plan,
                activations,
                projections,
                normalized,
                "lm_head.weight",
            )
        })?;
        *target_tokens = target_tokens
            .checked_add(1)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA target token count overflows".into()))?;
        *poisoned = false;
        Ok(logits)
    }

    pub fn mtp_tokens(&self) -> usize {
        self.mtp_tokens
    }

    /// Borrow the most recent complete target distribution without a host
    /// readback. The LM-head prepared output is overwritten only by the next
    /// target projection; gathered MTP proposals own a separate buffer.
    pub fn target_logits_device(&self) -> Result<CudaDeviceF32View<'_>> {
        if self.poisoned || self.target_tokens == 0 {
            return Err(EngineError::InvalidState(
                "CUDA target logits require one healthy committed target token".into(),
            ));
        }
        self.projections
            .get("lm_head.weight")
            .ok_or_else(|| EngineError::InvalidState("CUDA LM head is not resident".into()))?
            .device_output()
    }

    /// Executes the native one-layer MTP graph after the target model has
    /// selected `next_token_id`. The returned logits draft the following
    /// token; acceptance still requires a subsequent target-model step.
    pub fn dispatch_mtp_draft_device<'a>(
        &'a mut self,
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        next_token_id: usize,
        absolute_position: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        self.dispatch_mtp_draft_device_impl(
            runtime,
            config,
            next_token_id,
            absolute_position,
            CudaMtpHeadMode::Full,
        )?
        .ok_or_else(|| EngineError::InvalidState("CUDA full MTP draft omitted logits".into()))
    }

    pub fn dispatch_mtp_restricted_draft_device<'a>(
        &'a mut self,
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        next_token_id: usize,
        absolute_position: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        self.dispatch_mtp_draft_device_impl(
            runtime,
            config,
            next_token_id,
            absolute_position,
            CudaMtpHeadMode::Restricted,
        )?
        .ok_or_else(|| EngineError::InvalidState("CUDA restricted MTP draft omitted logits".into()))
    }

    /// Advance the native MTP state for an accepted token without reading
    /// either the complete or gathered LM head. This closes the final causal
    /// edge of a fully accepted speculative block.
    pub fn dispatch_mtp_advance_device(
        &mut self,
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        next_token_id: usize,
        absolute_position: usize,
    ) -> Result<()> {
        let logits = self.dispatch_mtp_draft_device_impl(
            runtime,
            config,
            next_token_id,
            absolute_position,
            CudaMtpHeadMode::StateOnly,
        )?;
        if logits.is_some() {
            return Err(EngineError::InvalidState(
                "CUDA state-only MTP advance unexpectedly produced logits".into(),
            ));
        }
        Ok(())
    }

    fn dispatch_mtp_draft_device_impl<'a>(
        &'a mut self,
        runtime: &CudaCandidateRuntime,
        config: &Qwen38Config,
        next_token_id: usize,
        absolute_position: usize,
        head_mode: CudaMtpHeadMode,
    ) -> Result<Option<CudaDeviceF32View<'a>>> {
        if config != &Qwen38Config::default() {
            return Err(EngineError::Shape(
                "CUDA MTP dispatch requires the frozen Qwen3.8-27B topology".into(),
            ));
        }
        if self.poisoned || self.mtp_poisoned {
            return Err(EngineError::InvalidState(
                "CUDA target or MTP session is poisoned; reset is required".into(),
            ));
        }
        if self.target_tokens != absolute_position || absolute_position != self.mtp_tokens + 1 {
            return Err(EngineError::InvalidState(format!(
                "CUDA MTP position {absolute_position} differs from target/MTP state {}/{}",
                self.target_tokens, self.mtp_tokens
            )));
        }
        if next_token_id >= self.embedding.rows() {
            return Err(EngineError::Shape(format!(
                "CUDA MTP token {next_token_id} exceeds embedding vocabulary {}",
                self.embedding.rows()
            )));
        }
        let embedding_scale = self.embedding_s_out(next_token_id)?;
        let final_target_norm = format!("target:{}:post_ffn:final", config.num_hidden_layers - 1);

        let Self {
            plan,
            embedding,
            activations,
            projections,
            full_attention,
            norms,
            mtp_concat,
            mtp_draft_projection,
            mtp_tokens,
            mtp_poisoned,
            ..
        } = self;
        *mtp_poisoned = true;
        let logits = runtime.run_token_submission("MTP token commit synchronization", || {
            let embedded =
                runtime.dispatch_embedding_row_device(embedding, next_token_id, embedding_scale)?;
            let embedded = runtime
                .dispatch_qwen_rms_norm_f16_device(norms.regular("mtp:pre_embedding")?, embedded)?;
            let previous_hidden = norms.residual(&final_target_norm)?.normalized_output()?;
            let previous_hidden = runtime.dispatch_qwen_rms_norm_f16_device(
                norms.regular("mtp:pre_hidden")?,
                previous_hidden,
            )?;
            let projection_input =
                runtime.dispatch_f32_concat_device(mtp_concat, embedded, previous_hidden)?;
            let mut residual = dispatch_projection_device(
                runtime,
                plan,
                activations,
                projections,
                projection_input,
                "mtp.fc.weight",
            )?;
            let mut normalized =
                runtime.dispatch_qwen_rms_norm_f16_device(norms.regular("mtp:input")?, residual)?;
            let [query_gate, key, value] = dispatch_projection_fanout_device(
                runtime,
                plan,
                activations,
                projections,
                normalized,
                [
                    "mtp.layers.0.self_attn.q_proj.weight".to_owned(),
                    "mtp.layers.0.self_attn.k_proj.weight".to_owned(),
                    "mtp.layers.0.self_attn.v_proj.weight".to_owned(),
                ],
            )?;
            let attention = full_attention.get_mut("mtp:0").ok_or_else(|| {
                EngineError::InvalidState("CUDA MTP full-attention state is not resident".into())
            })?;
            let (attention_output, gate) = attention.dispatch_device(
                runtime,
                query_gate,
                key,
                value,
                absolute_position as u64,
            )?;
            let attention_output = dispatch_sigmoid_gate_projection_device(
                runtime,
                plan,
                activations,
                projections,
                attention_output,
                gate,
                "mtp.layers.0.self_attn.o_proj.weight",
            )?;
            (residual, normalized) = runtime.dispatch_residual_rms_norm_f16_device(
                norms.residual("mtp:post_attention")?,
                residual,
                attention_output,
            )?;
            let [gate, up] = dispatch_projection_fanout_device(
                runtime,
                plan,
                activations,
                projections,
                normalized,
                [
                    "mtp.layers.0.mlp.gate_proj.weight".to_owned(),
                    "mtp.layers.0.mlp.up_proj.weight".to_owned(),
                ],
            )?;
            let down = dispatch_swiglu_projection_device(
                runtime,
                plan,
                activations,
                projections,
                gate,
                up,
                "mtp.layers.0.mlp.down_proj.weight",
            )?;
            let (_, final_hidden) = runtime.dispatch_residual_rms_norm_f16_device(
                norms.residual("mtp:final")?,
                residual,
                down,
            )?;
            match head_mode {
                CudaMtpHeadMode::Full => dispatch_projection_device(
                    runtime,
                    plan,
                    activations,
                    projections,
                    final_hidden,
                    "lm_head.weight",
                )
                .map(Some),
                CudaMtpHeadMode::Restricted => {
                    let group = plan.group_for_projection("lm_head.weight")?;
                    let activation = activations.get(group).ok_or_else(|| {
                        EngineError::InvalidState(
                            "CUDA LM-head activation group is not resident".into(),
                        )
                    })?;
                    let projection = projections.get("lm_head.weight").ok_or_else(|| {
                        EngineError::InvalidState("CUDA LM head is not resident".into())
                    })?;
                    let gathered = mtp_draft_projection.as_ref().ok_or_else(|| {
                        EngineError::InvalidState(
                            "CUDA restricted MTP head was not prepared at load".into(),
                        )
                    })?;
                    runtime
                        .dispatch_shared_a8_gathered_device(
                            activation,
                            final_hidden,
                            projection,
                            gathered,
                        )
                        .map(Some)
                }
                CudaMtpHeadMode::StateOnly => Ok(None),
            }
        })?;
        *mtp_tokens = mtp_tokens
            .checked_add(1)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA MTP token count overflows".into()))?;
        *mtp_poisoned = false;
        Ok(logits)
    }

    pub fn resident_bytes(&self) -> Result<u64> {
        checked_add(
            checked_add(self.model_bytes, self.graph_bytes, "CUDA resident bytes")?,
            self.session_bytes,
            "CUDA resident bytes",
        )
    }

    fn validate_bound_resources(&self) -> Result<()> {
        for resources in self
            .decode_bindings
            .steps()
            .iter()
            .map(|step| step.resources.as_slice())
            .chain(
                self.prefill_bindings
                    .steps()
                    .iter()
                    .map(|step| step.resources.as_slice()),
            )
        {
            for resource in resources {
                match resource {
                    CudaPreparedResource::Embedding | CudaPreparedResource::TokenBarrier => {}
                    CudaPreparedResource::Activation(key) => {
                        self.activation(key)?;
                    }
                    CudaPreparedResource::Projection(name) => {
                        self.projection(name)?;
                    }
                    CudaPreparedResource::LinearMixer(layer) => {
                        self.linear_mixer(*layer)?;
                    }
                    CudaPreparedResource::FullAttention(key) => {
                        self.full_attention(key)?;
                    }
                    CudaPreparedResource::RegularNorm(key) => {
                        self.norms.regular(key)?;
                    }
                    CudaPreparedResource::ResidualNorm(key) => {
                        self.norms.residual(key)?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn advance_prefill(
    cursor: &mut CudaPrefillExecutionCursor<'_>,
    layer: Option<usize>,
    operation: CudaPrefillOperation,
) -> Result<()> {
    let schedule_index = cursor
        .next_step()
        .ok_or_else(|| EngineError::InvalidState("CUDA prefill schedule ended early".into()))?
        .schedule_index;
    cursor.advance(schedule_index, layer, operation)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_mtp_prefill_parts(
    runtime: &CudaCandidateRuntime,
    config: &Qwen38Config,
    plan: &CudaProjectionPlan,
    activations: &BTreeMap<String, PreparedCudaA8Activation>,
    projections: &BTreeMap<String, PreparedCudaA8Projection>,
    full_attention: &mut BTreeMap<String, PreparedCudaFullAttentionLayer>,
    norms: &PreparedCudaNormGraph,
    workspaces: &PreparedCudaPrefillWorkspaces,
    embedded_target: CudaDeviceF32View<'_>,
    normalized_target: CudaDeviceF32View<'_>,
    alignment: CudaMtpPrefillAlignment,
) -> Result<()> {
    let hidden = config.hidden_size;
    let target_rows = alignment.chunk.token_count;
    let final_target_norm = format!("target:{}:post_ffn:final", config.num_hidden_layers - 1);
    let retained_target_hidden = norms.residual(&final_target_norm)?.normalized_output()?;
    let latest_target_hidden = normalized_target.slice(
        target_rows
            .checked_sub(1)
            .and_then(|row| row.checked_mul(hidden))
            .ok_or_else(|| EngineError::Shape("CUDA target boundary row overflows".into()))?,
        hidden,
    )?;

    if alignment.rows == 0 {
        runtime.copy_f32_rows_device(
            latest_target_hidden,
            hidden,
            retained_target_hidden,
            hidden,
            0,
            1,
            hidden,
        )?;
        return Ok(());
    }

    let previous_hidden =
        workspaces
            .projection_outputs()
            .device_output(0, hidden, alignment.rows)?;
    if alignment.previous_chunk_hidden_rows == 1 {
        runtime.copy_f32_rows_device(
            retained_target_hidden,
            hidden,
            previous_hidden,
            hidden,
            0,
            1,
            hidden,
        )?;
    }
    if alignment.current_chunk_hidden_rows > 0 {
        let destination = previous_hidden.slice(
            alignment
                .previous_chunk_hidden_rows
                .checked_mul(hidden)
                .ok_or_else(|| {
                    EngineError::Shape("CUDA MTP hidden destination offset overflows".into())
                })?,
            alignment
                .current_chunk_hidden_rows
                .checked_mul(hidden)
                .ok_or_else(|| {
                    EngineError::Shape("CUDA MTP hidden destination length overflows".into())
                })?,
        )?;
        runtime.copy_f32_rows_device(
            normalized_target,
            hidden,
            destination,
            hidden,
            0,
            alignment.current_chunk_hidden_rows,
            hidden,
        )?;
    }
    runtime.copy_f32_rows_device(
        latest_target_hidden,
        hidden,
        retained_target_hidden,
        hidden,
        0,
        1,
        hidden,
    )?;

    let embedding_values = alignment
        .rows
        .checked_mul(hidden)
        .ok_or_else(|| EngineError::Shape("CUDA MTP embedding slice overflows".into()))?;
    let mtp_embedding = embedded_target.slice(
        alignment
            .input_token_offset
            .checked_mul(hidden)
            .ok_or_else(|| EngineError::Shape("CUDA MTP embedding offset overflows".into()))?,
        embedding_values,
    )?;
    let normalized_embedding = runtime.dispatch_batched_qwen_rms_norm_f16_device(
        norms.regular("mtp:pre_embedding")?,
        workspaces.hidden_norm(),
        mtp_embedding,
        alignment.rows,
    )?;
    let concat_width = hidden
        .checked_mul(2)
        .ok_or_else(|| EngineError::Shape("CUDA MTP concat width overflows".into()))?;
    let projection_input =
        workspaces
            .projection_outputs()
            .device_output(1, concat_width, alignment.rows)?;
    runtime.copy_f32_rows_device(
        normalized_embedding,
        hidden,
        projection_input,
        concat_width,
        0,
        alignment.rows,
        hidden,
    )?;
    let normalized_previous = runtime.dispatch_batched_qwen_rms_norm_f16_device(
        norms.regular("mtp:pre_hidden")?,
        workspaces.hidden_norm(),
        previous_hidden,
        alignment.rows,
    )?;
    runtime.copy_f32_rows_device(
        normalized_previous,
        hidden,
        projection_input,
        concat_width,
        hidden,
        alignment.rows,
        hidden,
    )?;

    let [mut residual] = dispatch_prefill_projection_fanout_parts(
        runtime,
        plan,
        activations,
        projections,
        workspaces,
        projection_input,
        alignment.rows,
        ["mtp.fc.weight".to_owned()],
    )?;
    let retained_residual = workspaces.embedding().device_output(alignment.rows)?;
    runtime.copy_f32_rows_device(
        residual,
        hidden,
        retained_residual,
        hidden,
        0,
        alignment.rows,
        hidden,
    )?;
    residual = retained_residual;
    let mut normalized = runtime.dispatch_batched_qwen_rms_norm_f16_device(
        norms.regular("mtp:input")?,
        workspaces.hidden_norm(),
        residual,
        alignment.rows,
    )?;
    let [query_gate, key, value] = dispatch_prefill_projection_fanout_parts(
        runtime,
        plan,
        activations,
        projections,
        workspaces,
        normalized,
        alignment.rows,
        [
            "mtp.layers.0.self_attn.q_proj.weight".to_owned(),
            "mtp.layers.0.self_attn.k_proj.weight".to_owned(),
            "mtp.layers.0.self_attn.v_proj.weight".to_owned(),
        ],
    )?;
    let attention = full_attention.get_mut("mtp:0").ok_or_else(|| {
        EngineError::InvalidState("CUDA MTP full-attention state is not resident".into())
    })?;
    let (attention_output, gate) = attention.dispatch_prefill_with_positions_device(
        runtime,
        workspaces.key_norm(),
        workspaces.rope(),
        workspaces.query_gate(),
        workspaces.paged_gqa_output(),
        query_gate,
        key,
        value,
        alignment.cache_start_token,
        alignment.rope_start_position,
        alignment.rows,
    )?;
    let attention_output = dispatch_prefill_fused_projection_parts(
        runtime,
        plan,
        activations,
        projections,
        workspaces,
        attention_output,
        gate,
        alignment.rows,
        "mtp.layers.0.self_attn.o_proj.weight",
        PrefillFusedEdge::AttentionGate,
    )?;
    (residual, normalized) = runtime.dispatch_batched_residual_rms_norm_f16_device(
        norms.residual("mtp:post_attention")?,
        workspaces.hidden_norm(),
        residual,
        attention_output,
        alignment.rows,
    )?;
    let [gate, up] = dispatch_prefill_projection_fanout_parts(
        runtime,
        plan,
        activations,
        projections,
        workspaces,
        normalized,
        alignment.rows,
        [
            "mtp.layers.0.mlp.gate_proj.weight".to_owned(),
            "mtp.layers.0.mlp.up_proj.weight".to_owned(),
        ],
    )?;
    let down = dispatch_prefill_fused_projection_parts(
        runtime,
        plan,
        activations,
        projections,
        workspaces,
        gate,
        up,
        alignment.rows,
        "mtp.layers.0.mlp.down_proj.weight",
        PrefillFusedEdge::SwiGlu,
    )?;
    let _ = runtime.dispatch_batched_residual_rms_norm_f16_device(
        norms.residual("mtp:final")?,
        workspaces.hidden_norm(),
        residual,
        down,
        alignment.rows,
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
enum CudaMtpHeadMode {
    Full,
    Restricted,
    StateOnly,
}

#[derive(Clone, Copy)]
enum PrefillFusedEdge {
    AttentionGate,
    SwiGlu,
}

#[allow(clippy::too_many_arguments)]
fn dispatch_prefill_projection_fanout_parts<'a, const N: usize>(
    runtime: &CudaCandidateRuntime,
    plan: &CudaProjectionPlan,
    activations: &BTreeMap<String, PreparedCudaA8Activation>,
    projections: &'a BTreeMap<String, PreparedCudaA8Projection>,
    workspaces: &'a PreparedCudaPrefillWorkspaces,
    input: CudaDeviceF32View<'_>,
    tokens: usize,
    names: [String; N],
) -> Result<[CudaDeviceF32View<'a>; N]> {
    let first = names.first().ok_or_else(|| {
        EngineError::InvalidState("CUDA prefill projection fan-out cannot be empty".into())
    })?;
    let group = plan.group_for_projection(first)?;
    let activation = activations.get(group).ok_or_else(|| {
        EngineError::InvalidState(format!("CUDA activation group {group} is not resident"))
    })?;
    let mut prepared = Vec::with_capacity(N);
    for name in &names {
        if plan.group_for_projection(name)? != group {
            return Err(EngineError::InvalidState(format!(
                "CUDA prefill projection fan-out mixes activation groups at {name}"
            )));
        }
        let projection = projections.get(name).ok_or_else(|| {
            EngineError::InvalidState(format!("CUDA projection {name} is not resident"))
        })?;
        prepared.push((projection, prefill_projection_output_slot(name)?));
    }
    runtime
        .dispatch_batched_a8_arena_fanout_device(
            activation,
            workspaces.projection_activation(),
            workspaces.projection_outputs(),
            input,
            tokens,
            &prepared,
        )?
        .try_into()
        .map_err(|_| EngineError::InvalidState("CUDA prefill fan-out count changed".into()))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_prefill_fused_projection_parts<'a>(
    runtime: &CudaCandidateRuntime,
    plan: &CudaProjectionPlan,
    activations: &BTreeMap<String, PreparedCudaA8Activation>,
    projections: &'a BTreeMap<String, PreparedCudaA8Projection>,
    workspaces: &'a PreparedCudaPrefillWorkspaces,
    left: CudaDeviceF32View<'_>,
    right: CudaDeviceF32View<'_>,
    tokens: usize,
    name: &str,
    edge: PrefillFusedEdge,
) -> Result<CudaDeviceF32View<'a>> {
    let group = plan.group_for_projection(name)?;
    let activation = activations.get(group).ok_or_else(|| {
        EngineError::InvalidState(format!("CUDA activation group {group} is not resident"))
    })?;
    let projection = projections.get(name).ok_or_else(|| {
        EngineError::InvalidState(format!("CUDA projection {name} is not resident"))
    })?;
    let slot = prefill_projection_output_slot(name)?;
    let mut outputs = match edge {
        PrefillFusedEdge::AttentionGate => runtime
            .dispatch_batched_a8_arena_sigmoid_gate_fanout_device(
                activation,
                workspaces.projection_activation(),
                workspaces.projection_outputs(),
                left,
                right,
                tokens,
                &[(projection, slot)],
            )?,
        PrefillFusedEdge::SwiGlu => runtime.dispatch_batched_a8_arena_swiglu_fanout_device(
            activation,
            workspaces.projection_activation(),
            workspaces.projection_outputs(),
            left,
            right,
            tokens,
            &[(projection, slot)],
        )?,
    };
    outputs.pop().ok_or_else(|| {
        EngineError::InvalidState("CUDA prefill fused projection has no output".into())
    })
}

fn dispatch_projection_fanout_device<'a, const N: usize>(
    runtime: &CudaCandidateRuntime,
    plan: &CudaProjectionPlan,
    activations: &BTreeMap<String, PreparedCudaA8Activation>,
    projections: &'a BTreeMap<String, PreparedCudaA8Projection>,
    input: CudaDeviceF32View<'_>,
    names: [String; N],
) -> Result<[CudaDeviceF32View<'a>; N]> {
    let first = names.first().ok_or_else(|| {
        EngineError::InvalidState("CUDA projection fan-out cannot be empty".into())
    })?;
    let group = plan.group_for_projection(first)?;
    let activation = activations.get(group).ok_or_else(|| {
        EngineError::InvalidState(format!("CUDA activation group {group} is not resident"))
    })?;
    let mut prepared = Vec::with_capacity(N);
    for name in &names {
        if plan.group_for_projection(name)? != group {
            return Err(EngineError::InvalidState(format!(
                "CUDA projection fan-out mixes activation groups at {name}"
            )));
        }
        prepared.push(projections.get(name).ok_or_else(|| {
            EngineError::InvalidState(format!("CUDA projection {name} is not resident"))
        })?);
    }
    runtime
        .dispatch_shared_a8_fanout_device(activation, input, &prepared)?
        .try_into()
        .map_err(|_| EngineError::InvalidState("CUDA fan-out output count changed".into()))
}

fn dispatch_projection_device<'a>(
    runtime: &CudaCandidateRuntime,
    plan: &CudaProjectionPlan,
    activations: &BTreeMap<String, PreparedCudaA8Activation>,
    projections: &'a BTreeMap<String, PreparedCudaA8Projection>,
    input: CudaDeviceF32View<'_>,
    name: &str,
) -> Result<CudaDeviceF32View<'a>> {
    let [output] = dispatch_projection_fanout_device(
        runtime,
        plan,
        activations,
        projections,
        input,
        [name.to_owned()],
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_sigmoid_gate_projection_device<'a>(
    runtime: &CudaCandidateRuntime,
    plan: &CudaProjectionPlan,
    activations: &BTreeMap<String, PreparedCudaA8Activation>,
    projections: &'a BTreeMap<String, PreparedCudaA8Projection>,
    attention: CudaDeviceF32View<'_>,
    gate: CudaDeviceF32View<'_>,
    name: &str,
) -> Result<CudaDeviceF32View<'a>> {
    let group = plan.group_for_projection(name)?;
    let activation = activations.get(group).ok_or_else(|| {
        EngineError::InvalidState(format!("CUDA activation group {group} is not resident"))
    })?;
    let projection = projections.get(name).ok_or_else(|| {
        EngineError::InvalidState(format!("CUDA projection {name} is not resident"))
    })?;
    let mut outputs = runtime.dispatch_shared_a8_sigmoid_gate_fanout_device(
        activation,
        attention,
        gate,
        &[projection],
    )?;
    outputs
        .pop()
        .ok_or_else(|| EngineError::InvalidState("CUDA gate projection has no output".into()))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_swiglu_projection_device<'a>(
    runtime: &CudaCandidateRuntime,
    plan: &CudaProjectionPlan,
    activations: &BTreeMap<String, PreparedCudaA8Activation>,
    projections: &'a BTreeMap<String, PreparedCudaA8Projection>,
    gate: CudaDeviceF32View<'_>,
    up: CudaDeviceF32View<'_>,
    name: &str,
) -> Result<CudaDeviceF32View<'a>> {
    let group = plan.group_for_projection(name)?;
    let activation = activations.get(group).ok_or_else(|| {
        EngineError::InvalidState(format!("CUDA activation group {group} is not resident"))
    })?;
    let projection = projections.get(name).ok_or_else(|| {
        EngineError::InvalidState(format!("CUDA projection {name} is not resident"))
    })?;
    let mut outputs =
        runtime.dispatch_shared_a8_swiglu_fanout_device(activation, gate, up, &[projection])?;
    outputs
        .pop()
        .ok_or_else(|| EngineError::InvalidState("CUDA SwiGLU projection has no output".into()))
}

fn prepare_norms(
    runtime: &CudaCandidateRuntime,
    artifact: &ModelArtifact,
    config: &Qwen38Config,
) -> Result<PreparedCudaNormGraph> {
    let mut regular = BTreeMap::new();
    let mut residual = BTreeMap::new();
    let mut model_bytes = 0_u64;
    let mut graph_bytes = 0_u64;
    let norm_config = CudaRmsNormConfig {
        rows: 1,
        columns: config.hidden_size,
        epsilon: config.rms_norm_epsilon,
    };

    add_regular_norm(
        runtime,
        artifact,
        norm_config,
        "target:initial",
        "model.language_model.layers.0.input_layernorm.weight",
        &mut regular,
        &mut model_bytes,
        &mut graph_bytes,
    )?;
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{layer}");
        add_residual_norm(
            runtime,
            artifact,
            norm_config,
            &format!("target:{layer}:post_attention"),
            &format!("{prefix}.post_attention_layernorm.weight"),
            &mut residual,
            &mut model_bytes,
            &mut graph_bytes,
        )?;
        let (key, tensor) = if layer + 1 == config.num_hidden_layers {
            (
                format!("target:{layer}:post_ffn:final"),
                "model.language_model.norm.weight".to_owned(),
            )
        } else {
            (
                format!("target:{layer}:post_ffn:layer_{}", layer + 1),
                format!(
                    "model.language_model.layers.{}.input_layernorm.weight",
                    layer + 1
                ),
            )
        };
        add_residual_norm(
            runtime,
            artifact,
            norm_config,
            &key,
            &tensor,
            &mut residual,
            &mut model_bytes,
            &mut graph_bytes,
        )?;
    }
    for (key, tensor) in [
        ("mtp:pre_embedding", "mtp.pre_fc_norm_embedding.weight"),
        ("mtp:pre_hidden", "mtp.pre_fc_norm_hidden.weight"),
        ("mtp:input", "mtp.layers.0.input_layernorm.weight"),
    ] {
        add_regular_norm(
            runtime,
            artifact,
            norm_config,
            key,
            tensor,
            &mut regular,
            &mut model_bytes,
            &mut graph_bytes,
        )?;
    }
    for (key, tensor) in [
        (
            "mtp:post_attention",
            "mtp.layers.0.post_attention_layernorm.weight",
        ),
        ("mtp:final", "mtp.norm.weight"),
    ] {
        add_residual_norm(
            runtime,
            artifact,
            norm_config,
            key,
            tensor,
            &mut residual,
            &mut model_bytes,
            &mut graph_bytes,
        )?;
    }
    if regular.len() != 4 || residual.len() != config.num_hidden_layers * 2 + 2 {
        return Err(EngineError::InvalidState(format!(
            "CUDA norm graph has {} regular and {} residual operators",
            regular.len(),
            residual.len()
        )));
    }
    Ok(PreparedCudaNormGraph {
        regular,
        residual,
        model_bytes,
        graph_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn add_regular_norm(
    runtime: &CudaCandidateRuntime,
    artifact: &ModelArtifact,
    config: CudaRmsNormConfig,
    key: &str,
    tensor: &str,
    norms: &mut BTreeMap<String, PreparedCudaRmsNorm>,
    model_bytes: &mut u64,
    graph_bytes: &mut u64,
) -> Result<()> {
    let prepared = runtime.prepare_qwen_rms_norm_f16(config, artifact_f16(artifact, tensor)?)?;
    *model_bytes = checked_add(
        *model_bytes,
        u64::try_from(prepared.model_bytes())
            .map_err(|_| EngineError::MemoryBudget("CUDA norm model bytes exceed u64".into()))?,
        "CUDA norm model bytes",
    )?;
    *graph_bytes = checked_add(
        *graph_bytes,
        u64::try_from(prepared.transient_bytes())
            .map_err(|_| EngineError::MemoryBudget("CUDA norm graph bytes exceed u64".into()))?,
        "CUDA norm graph bytes",
    )?;
    if norms.insert(key.to_owned(), prepared).is_some() {
        return Err(EngineError::InvalidState(format!(
            "duplicate CUDA norm {key}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_residual_norm(
    runtime: &CudaCandidateRuntime,
    artifact: &ModelArtifact,
    config: CudaRmsNormConfig,
    key: &str,
    tensor: &str,
    norms: &mut BTreeMap<String, PreparedCudaResidualRmsNorm>,
    model_bytes: &mut u64,
    graph_bytes: &mut u64,
) -> Result<()> {
    let prepared =
        runtime.prepare_residual_rms_norm_f16(config, artifact_f16(artifact, tensor)?)?;
    *model_bytes = checked_add(
        *model_bytes,
        u64::try_from(prepared.model_bytes()).map_err(|_| {
            EngineError::MemoryBudget("CUDA residual norm model bytes exceed u64".into())
        })?,
        "CUDA norm model bytes",
    )?;
    *graph_bytes = checked_add(
        *graph_bytes,
        u64::try_from(prepared.transient_bytes()).map_err(|_| {
            EngineError::MemoryBudget("CUDA residual norm graph bytes exceed u64".into())
        })?,
        "CUDA norm graph bytes",
    )?;
    if norms.insert(key.to_owned(), prepared).is_some() {
        return Err(EngineError::InvalidState(format!(
            "duplicate CUDA residual norm {key}"
        )));
    }
    Ok(())
}

fn prepare_full_attention(
    runtime: &CudaCandidateRuntime,
    artifact: &ModelArtifact,
    config: &Qwen38Config,
    key: &str,
    prefix: &str,
    maximum_context_tokens: usize,
) -> Result<PreparedCudaFullAttentionLayer> {
    PreparedCudaFullAttentionLayer::prepare(
        runtime,
        config,
        key,
        artifact_f16(artifact, &format!("{prefix}.q_norm.weight"))?,
        artifact_f16(artifact, &format!("{prefix}.k_norm.weight"))?,
        maximum_context_tokens,
    )
}

fn artifact_f16<'a>(artifact: &'a ModelArtifact, name: &str) -> Result<&'a [u8]> {
    match artifact.float_tensor(name)? {
        FloatTensorView::F16Le(bytes) => Ok(bytes),
        FloatTensorView::F32Le(_) => Err(EngineError::UnsupportedDType(format!(
            "CUDA tensor {name} must remain packed FP16"
        ))),
    }
}

fn artifact_f32<'a>(artifact: &'a ModelArtifact, name: &str) -> Result<&'a [u8]> {
    match artifact.float_tensor(name)? {
        FloatTensorView::F32Le(bytes) => Ok(bytes),
        FloatTensorView::F16Le(_) => Err(EngineError::UnsupportedDType(format!(
            "CUDA tensor {name} must remain packed F32"
        ))),
    }
}

fn scale_bytes(values: usize, label: &str) -> Result<u64> {
    values
        .checked_mul(std::mem::size_of::<half::f16>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| EngineError::MemoryBudget(format!("{label} overflow")))
}

fn packed_f16(scales: ScaleSlice<'_>) -> Result<&[u8]> {
    match scales {
        ScaleSlice::F16Le(bytes) => Ok(bytes),
        ScaleSlice::F32(_) => Err(EngineError::InvalidArtifact(
            "CUDA artifact graph requires packed FP16 recovery scales".into(),
        )),
    }
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| EngineError::MemoryBudget(format!("{label} overflow")))
}

fn checked_product(values: &[u64], label: &str) -> Result<u64> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| EngineError::MemoryBudget(format!("{label} overflow")))
    })
}

fn prefill_projection_output_slot(name: &str) -> Result<usize> {
    let slot = if name.ends_with(".gate_proj.weight")
        || name.ends_with(".q_proj.weight")
        || name.ends_with(".in_proj_qkv.weight")
        || name.ends_with(".o_proj.weight")
        || name.ends_with(".out_proj.weight")
        || name.ends_with(".down_proj.weight")
        || name == "mtp.fc.weight"
    {
        0
    } else if name.ends_with(".up_proj.weight")
        || name.ends_with(".k_proj.weight")
        || name.ends_with(".in_proj_z.weight")
    {
        1
    } else if name.ends_with(".v_proj.weight") || name.ends_with(".in_proj_a.weight") {
        2
    } else if name.ends_with(".in_proj_b.weight") {
        3
    } else {
        return Err(EngineError::InvalidState(format!(
            "CUDA prefill projection {name} has no frozen output slot"
        )));
    };
    Ok(slot)
}

fn sum_usize<const N: usize>(values: [usize; N], label: &str) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        let value = u64::try_from(value)
            .map_err(|_| EngineError::MemoryBudget(format!("{label} exceeds u64")))?;
        checked_add(total, value, label)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_prefill_frontend_and_attention_workspace_is_one_fixed_pool() {
        let budget = CudaPrefillWorkspaceBudget::qwen38(&Qwen38Config::default(), 512).unwrap();
        assert_eq!(budget.embedding_row_id_bytes, 2_048);
        assert_eq!(budget.embedding_output_bytes, 10_485_760);
        assert_eq!(budget.hidden_norm_bytes, 20_971_520);
        assert_eq!(budget.key_norm_bytes, 4_194_304);
        assert_eq!(budget.rope_table_bytes, 131_072);
        assert_eq!(budget.query_gate_bytes, 25_165_824);
        assert_eq!(budget.paged_gqa_output_bytes, 12_582_912);
        assert_eq!(budget.total_bytes, 73_533_440);
    }

    #[test]
    fn prefill_attention_workspace_scales_with_chunk_not_layer_count() {
        let config = Qwen38Config::default();
        let one = CudaPrefillWorkspaceBudget::qwen38(&config, 1).unwrap();
        let full = CudaPrefillWorkspaceBudget::qwen38(&config, 512).unwrap();
        assert_eq!(full.total_bytes, one.total_bytes * 512);
        assert_eq!(full.total_bytes, 73_533_440);
        assert!(CudaPrefillWorkspaceBudget::qwen38(&config, 0).is_err());
        assert!(CudaPrefillWorkspaceBudget::qwen38(&config, 65_536).is_err());
    }

    #[test]
    fn frozen_prefill_projection_arena_covers_all_chunk_matrices_once() {
        let config = Qwen38Config::default();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let plan = CudaPrefillProjectionWorkspacePlan::qwen38(&config, &projections, 512).unwrap();
        assert_eq!(plan.activation_columns, 17_408);
        assert_eq!(plan.output_slot_rows, [17_408, 17_408, 1_024, 48]);
        assert_eq!(plan.chunk_projection_count, 504);
        assert_eq!(plan.last_token_lm_head_rows, 248_320);
        assert_eq!(plan.activation_code_bytes, 8_912_896);
        assert_eq!(plan.activation_scale_bytes, 557_056);
        assert_eq!(plan.output_arena_bytes, 73_498_624);
        assert_eq!(plan.total_bytes, 82_968_576);
    }

    #[test]
    fn projection_arena_scales_with_chunk_not_projection_count() {
        let config = Qwen38Config::default();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let one = CudaPrefillProjectionWorkspacePlan::qwen38(&config, &projections, 1).unwrap();
        let full = CudaPrefillProjectionWorkspacePlan::qwen38(&config, &projections, 512).unwrap();
        assert_eq!(full.total_bytes, one.total_bytes * 512);
        assert_eq!(
            full.chunk_projection_count,
            projections.projection_count() - 1
        );
    }

    #[test]
    fn frozen_linear_prefill_workspace_is_shared_across_all_forty_eight_layers() {
        let config = Qwen38Config::default();
        let budget = CudaPrefillLinearWorkspaceBudget::qwen38(&config, 512).unwrap();
        assert_eq!(budget.causal_conv_output_bytes, 20_971_520);
        assert_eq!(budget.gated_delta_input_bytes, 37_945_344);
        assert_eq!(budget.gated_delta_output_bytes, 12_582_912);
        assert_eq!(budget.gated_rms_norm_output_bytes, 12_582_912);
        assert_eq!(budget.total_bytes, 84_082_688);
        let attention = CudaPrefillWorkspaceBudget::qwen38(&config, 512).unwrap();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let projection =
            CudaPrefillProjectionWorkspacePlan::qwen38(&config, &projections, 512).unwrap();
        assert_eq!(
            budget.total_bytes + attention.total_bytes + projection.total_bytes,
            240_584_704
        );
    }

    #[test]
    fn frozen_plan_covers_every_non_embedding_projection_once() {
        let plan = CudaProjectionPlan::qwen38(&Qwen38Config::default()).unwrap();
        assert_eq!(plan.projection_count(), 505);
        assert_eq!(plan.group_count(), 262);
        assert!(!plan.projection_groups.contains_key(EMBEDDING_MATRIX));
        assert_eq!(
            plan.groups()
                .iter()
                .map(|group| group.projection_names.len())
                .sum::<usize>(),
            plan.projection_count()
        );
    }

    #[test]
    fn qwen_fanouts_share_and_independent_edges_do_not() {
        let plan = CudaProjectionPlan::qwen38(&Qwen38Config::default()).unwrap();
        let prefix = "model.language_model.layers.3";
        let q = format!("{prefix}.self_attn.q_proj.weight");
        let k = format!("{prefix}.self_attn.k_proj.weight");
        let v = format!("{prefix}.self_attn.v_proj.weight");
        assert_eq!(
            plan.group_for_projection(&q).unwrap(),
            plan.group_for_projection(&k).unwrap()
        );
        assert_eq!(
            plan.group_for_projection(&q).unwrap(),
            plan.group_for_projection(&v).unwrap()
        );
        let down = format!("{prefix}.mlp.down_proj.weight");
        let output = format!("{prefix}.self_attn.o_proj.weight");
        assert_ne!(
            plan.group_for_projection(&down).unwrap(),
            plan.group_for_projection(&output).unwrap()
        );
        assert!(plan.group_for_projection(EMBEDDING_MATRIX).is_err());
    }

    #[test]
    fn mtp_target_projection_ownership_is_explicit() {
        let plan = CudaProjectionPlan::qwen38(&Qwen38Config::default()).unwrap();
        let mtp_q = "mtp.layers.0.self_attn.q_proj.weight";
        let mtp_k = "mtp.layers.0.self_attn.k_proj.weight";
        assert_eq!(
            plan.group_for_projection(mtp_q).unwrap(),
            plan.group_for_projection(mtp_k).unwrap()
        );
        assert!(plan
            .group_for_projection("mtp.fc.weight")
            .unwrap()
            .starts_with("independent:"));
        assert!(plan
            .group_for_projection("lm_head.weight")
            .unwrap()
            .starts_with("independent:"));
    }

    #[test]
    fn frozen_decode_bindings_cover_every_resident_operator() {
        let config = Qwen38Config::default();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let schedule = CudaDecodeSchedule::qwen38(&config).unwrap();
        let bindings = CudaDecodeBindingPlan::qwen38(&schedule, &projections, &config).unwrap();
        assert_eq!(bindings.steps().len(), 645);
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::Projection(_))
            }),
            505
        );
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::Activation(_))
            }),
            262
        );
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::LinearMixer(_))
            }),
            48
        );
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::FullAttention(_))
            }),
            17
        );
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::RegularNorm(_))
            }),
            4
        );
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::ResidualNorm(_))
            }),
            130
        );
    }

    #[test]
    fn frozen_prefill_bindings_cover_every_resident_operator() {
        let config = Qwen38Config::default();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let schedule = CudaPrefillSchedule::qwen38(&config, 512).unwrap();
        let bindings = CudaPrefillBindingPlan::qwen38(&schedule, &projections, &config).unwrap();
        assert_eq!(bindings.steps().len(), 645);
        assert_eq!(bindings.max_chunk_tokens(), 512);
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::Projection(_))
            }),
            505
        );
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::Activation(_))
            }),
            262
        );
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::LinearMixer(_))
            }),
            48
        );
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::FullAttention(_))
            }),
            17
        );
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::RegularNorm(_))
            }),
            4
        );
        assert_eq!(
            bindings.resource_count(|resource| {
                matches!(resource, CudaPreparedResource::ResidualNorm(_))
            }),
            130
        );
    }

    #[test]
    fn prefill_execution_cursor_commits_only_after_all_bound_steps() {
        let config = Qwen38Config::default();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let schedule = CudaPrefillSchedule::qwen38(&config, 512).unwrap();
        let bindings = CudaPrefillBindingPlan::qwen38(&schedule, &projections, &config).unwrap();
        let chunk = CudaPrefillChunk {
            start_position: 512,
            token_count: 137,
        };
        let mut cursor = bindings.execution_cursor(chunk, 512, 131_072).unwrap();
        assert_eq!(cursor.chunk(), chunk);
        while let Some(step) = cursor.next_step() {
            let (schedule_index, layer, operation) =
                (step.schedule_index, step.layer, step.operation);
            cursor.advance(schedule_index, layer, operation).unwrap();
        }
        assert_eq!(cursor.finish().unwrap(), 649);
    }

    #[test]
    fn prefill_execution_cursor_rejects_wrong_order_and_partial_commit() {
        let config = Qwen38Config::default();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let schedule = CudaPrefillSchedule::qwen38(&config, 512).unwrap();
        let bindings = CudaPrefillBindingPlan::qwen38(&schedule, &projections, &config).unwrap();
        let chunk = CudaPrefillChunk {
            start_position: 0,
            token_count: 512,
        };
        let mut cursor = bindings.execution_cursor(chunk, 0, 512).unwrap();
        let first = cursor.next_step().unwrap();
        assert!(cursor
            .advance(first.schedule_index + 1, first.layer, first.operation)
            .is_err());
        assert_eq!(cursor.next_step().unwrap().schedule_index, 0);
        cursor
            .advance(first.schedule_index, first.layer, first.operation)
            .unwrap();
        assert!(cursor.finish().is_err());

        assert!(bindings.execution_cursor(chunk, 1, 512).is_err());
        assert!(bindings
            .execution_cursor(
                CudaPrefillChunk {
                    start_position: 0,
                    token_count: 513,
                },
                0,
                1_024,
            )
            .is_err());
        assert!(bindings
            .execution_cursor(
                CudaPrefillChunk {
                    start_position: 500,
                    token_count: 20,
                },
                500,
                512,
            )
            .is_err());
    }

    #[test]
    fn target_only_prefill_can_skip_exactly_the_optional_mtp_step() {
        let config = Qwen38Config::default();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let schedule = CudaPrefillSchedule::qwen38(&config, 512).unwrap();
        let bindings = CudaPrefillBindingPlan::qwen38(&schedule, &projections, &config).unwrap();
        let mut cursor = bindings
            .execution_cursor(
                CudaPrefillChunk {
                    start_position: 0,
                    token_count: 8,
                },
                0,
                128,
            )
            .unwrap();
        assert!(cursor.skip_disabled_mtp().is_err());
        while cursor
            .next_step()
            .is_some_and(|step| step.operation != CudaPrefillOperation::MtpPrefillCausalScan)
        {
            let step = cursor.next_step().unwrap().clone();
            cursor
                .advance(step.schedule_index, step.layer, step.operation)
                .unwrap();
        }
        cursor.skip_disabled_mtp().unwrap();
        assert_eq!(
            cursor.next_step().unwrap().operation,
            CudaPrefillOperation::ChunkBarrier
        );
        assert!(cursor.skip_disabled_mtp().is_err());
    }

    #[test]
    fn intermediate_prefill_can_skip_only_its_lm_head_read() {
        let config = Qwen38Config::default();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let schedule = CudaPrefillSchedule::qwen38(&config, 512).unwrap();
        let bindings = CudaPrefillBindingPlan::qwen38(&schedule, &projections, &config).unwrap();
        let mut cursor = bindings
            .execution_cursor(
                CudaPrefillChunk {
                    start_position: 512,
                    token_count: 8,
                },
                512,
                1_024,
            )
            .unwrap();
        assert!(cursor.skip_intermediate_lm_head().is_err());
        while cursor
            .next_step()
            .is_some_and(|step| step.operation != CudaPrefillOperation::LastTokenLmHead)
        {
            let step = cursor.next_step().unwrap().clone();
            cursor
                .advance(step.schedule_index, step.layer, step.operation)
                .unwrap();
        }
        cursor.skip_intermediate_lm_head().unwrap();
        assert_eq!(
            cursor.next_step().unwrap().operation,
            CudaPrefillOperation::MtpPrefillCausalScan
        );
        assert!(cursor.skip_intermediate_lm_head().is_err());
        cursor.skip_disabled_mtp().unwrap();
        let barrier = cursor.next_step().unwrap().clone();
        cursor
            .advance(barrier.schedule_index, barrier.layer, barrier.operation)
            .unwrap();
        assert_eq!(cursor.finish().unwrap(), 520);
    }

    #[test]
    fn mtp_prefill_cursor_commits_aligned_target_and_mtp_counters() {
        let config = Qwen38Config::default();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let schedule = CudaPrefillSchedule::qwen38(&config, 512).unwrap();
        let bindings = CudaPrefillBindingPlan::qwen38(&schedule, &projections, &config).unwrap();
        let chunk = CudaPrefillChunk {
            start_position: 512,
            token_count: 8,
        };
        let alignment = CudaMtpPrefillAlignment::qwen38(chunk, 512, 511).unwrap();
        let mut cursor = bindings.execution_cursor(chunk, 512, 1_024).unwrap();
        while cursor
            .next_step()
            .is_some_and(|step| step.operation != CudaPrefillOperation::MtpPrefillCausalScan)
        {
            let step = cursor.next_step().unwrap().clone();
            cursor
                .advance(step.schedule_index, step.layer, step.operation)
                .unwrap();
        }
        let wrong_alignment = CudaMtpPrefillAlignment::qwen38(
            CudaPrefillChunk {
                start_position: 0,
                token_count: 8,
            },
            0,
            0,
        )
        .unwrap();
        assert!(cursor.advance_mtp(wrong_alignment).is_err());
        cursor.advance_mtp(alignment).unwrap();
        let barrier = cursor.next_step().unwrap().clone();
        cursor
            .advance(barrier.schedule_index, barrier.layer, barrier.operation)
            .unwrap();
        assert_eq!(cursor.finish_with_mtp(alignment).unwrap(), (520, 519));
    }

    #[test]
    fn prefill_binding_rejects_residual_norm_without_its_exact_producer() {
        let config = Qwen38Config::default();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let mut schedule = CudaPrefillSchedule::qwen38(&config, 512).unwrap();
        let first_residual = schedule
            .steps
            .iter()
            .position(|step| step.operation == CudaPrefillOperation::ResidualRmsNormBatch)
            .unwrap();
        schedule.steps[first_residual - 1].operation = CudaPrefillOperation::KeyRopeBatch;
        assert!(CudaPrefillBindingPlan::qwen38(&schedule, &projections, &config).is_err());
    }

    #[test]
    fn binding_rejects_schedule_that_hides_the_mtp_program() {
        let config = Qwen38Config::default();
        let projections = CudaProjectionPlan::qwen38(&config).unwrap();
        let mut schedule = CudaDecodeSchedule::qwen38(&config).unwrap();
        let mtp = schedule
            .steps
            .iter_mut()
            .find(|step| step.operation == CudaDecodeOperation::MtpDraftAndTargetVerify)
            .unwrap();
        mtp.operation = CudaDecodeOperation::LmHead;
        assert!(CudaDecodeBindingPlan::qwen38(&schedule, &projections, &config).is_err());
    }
}
