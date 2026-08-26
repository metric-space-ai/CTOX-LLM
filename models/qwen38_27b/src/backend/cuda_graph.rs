//! Artifact-backed CUDA projection graph for the frozen Qwen3.8-27B model.
//!
//! This module is deliberately model-specific. It resolves every target and
//! MTP projection into the exact shared-correction group used by the recovery
//! pipeline, then prepares immutable packed weights and recovery scales
//! directly from the CTOXQ mapping. It does not implement a generic graph
//! runtime and it does not admit embedding-row lookup until that dedicated
//! production kernel is wired.

use std::collections::{BTreeMap, BTreeSet};

use crate::backend::cuda_runtime::{
    CudaCandidateRuntime, CudaCausalConvConfig, CudaGatedDeltaConfig, CudaGatedRmsNormConfig,
    CudaPagedGqaConfig, CudaPartialRopeConfig, CudaQueryGateConfig, CudaRmsNormConfig,
    PreparedCudaA8Activation, PreparedCudaA8Projection, PreparedCudaCausalConv,
    PreparedCudaEmbedding, PreparedCudaGatedDelta, PreparedCudaGatedDeltaInputs,
    PreparedCudaGatedRmsNorm, PreparedCudaPagedGqa, PreparedCudaPartialRope, PreparedCudaQueryGate,
    PreparedCudaResidualRmsNorm, PreparedCudaRmsNorm,
};
use crate::backend::{Activation, ScaleSlice};
use crate::config::LayerKind;
use crate::fanout::qwen38_fanout_groups;
use crate::kv_cache::{DEFAULT_KV_PAGE_TOKENS, DEFAULT_KV_RECENT_TOKENS, DEFAULT_KV_SINK_TOKENS};
use crate::loader::{FloatTensorView, ModelArtifact};
use crate::tensor_contract::{expected_tensor_contract, validate_tensor_contract, TensorClass};
use crate::{EngineError, Qwen38Config, Result};

const EMBEDDING_MATRIX: &str = "model.language_model.embed_tokens.weight";

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

/// Resident CUDA projection state. `artifact` intentionally keeps the
/// immutable mapping alive, while all accelerator allocations are owned by
/// the prepared objects' private driver context.
pub struct PreparedCudaProjectionGraph {
    artifact: ModelArtifact,
    plan: CudaProjectionPlan,
    embedding: PreparedCudaEmbedding,
    activations: BTreeMap<String, PreparedCudaA8Activation>,
    projections: BTreeMap<String, PreparedCudaA8Projection>,
    linear_mixers: BTreeMap<usize, PreparedCudaLinearMixerLayer>,
    full_attention: BTreeMap<String, PreparedCudaFullAttentionLayer>,
    norms: PreparedCudaNormGraph,
    model_bytes: u64,
    graph_bytes: u64,
    session_bytes: u64,
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
}

impl PreparedCudaLinearMixerLayer {
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
}

impl PreparedCudaProjectionGraph {
    pub fn prepare(
        runtime: &CudaCandidateRuntime,
        artifact: &ModelArtifact,
        config: &Qwen38Config,
        maximum_context_tokens: usize,
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
        let mut activations = BTreeMap::new();
        let mut projections = BTreeMap::new();
        let mut linear_mixers = BTreeMap::new();
        let mut full_attention = BTreeMap::new();
        let mut model_bytes = 0_u64;
        let mut graph_bytes = 0_u64;
        let mut session_bytes = 0_u64;

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
            let convolution = runtime.prepare_causal_conv_f16(
                CudaCausalConvConfig::QWEN38_27B,
                artifact_f16(artifact, &format!("{prefix}.conv1d.weight"))?,
            )?;
            let inputs = runtime.prepare_gated_delta_inputs_f32_le(
                artifact_f32(artifact, &format!("{prefix}.A_log"))?,
                artifact_f32(artifact, &format!("{prefix}.dt_bias"))?,
            )?;
            let recurrence = runtime.prepare_gated_delta_f16(CudaGatedDeltaConfig::QWEN38_27B)?;
            let norm = runtime.prepare_gated_rms_norm_f16(
                CudaGatedRmsNormConfig::QWEN38_27B,
                artifact_f16(artifact, &format!("{prefix}.norm.weight"))?,
            )?;
            let mixer_model_bytes = sum_usize(
                [
                    convolution.model_bytes(),
                    inputs.model_bytes(),
                    norm.model_bytes(),
                ],
                "CUDA linear mixer model bytes",
            )?;
            let mixer_graph_bytes = sum_usize(
                [
                    convolution.transient_bytes(),
                    inputs.transient_bytes(),
                    recurrence.transient_bytes(),
                    norm.transient_bytes(),
                ],
                "CUDA linear mixer graph bytes",
            )?;
            let mixer_session_bytes = sum_usize(
                [
                    convolution.resident_state_bytes(),
                    recurrence.resident_state_bytes(),
                ],
                "CUDA linear mixer session bytes",
            )?;
            model_bytes = checked_add(model_bytes, mixer_model_bytes, "CUDA model bytes")?;
            graph_bytes = checked_add(graph_bytes, mixer_graph_bytes, "CUDA graph bytes")?;
            session_bytes = checked_add(session_bytes, mixer_session_bytes, "CUDA session bytes")?;
            let mixer = PreparedCudaLinearMixerLayer {
                layer,
                convolution,
                inputs,
                recurrence,
                norm,
                model_bytes: mixer_model_bytes,
                graph_bytes: mixer_graph_bytes,
                session_bytes: mixer_session_bytes,
            };
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
        Ok(Self {
            artifact: artifact.clone(),
            plan,
            embedding,
            activations,
            projections,
            linear_mixers,
            full_attention,
            norms,
            model_bytes,
            graph_bytes,
            session_bytes,
        })
    }

    pub fn artifact_manifest_sha256(&self) -> &str {
        self.artifact.manifest_sha256()
    }

    pub fn plan(&self) -> &CudaProjectionPlan {
        &self.plan
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

    pub fn resident_bytes(&self) -> Result<u64> {
        checked_add(
            checked_add(self.model_bytes, self.graph_bytes, "CUDA resident bytes")?,
            self.session_bytes,
            "CUDA resident bytes",
        )
    }
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
    let query_gate = runtime.prepare_query_gate_norm_rope_f32(
        CudaQueryGateConfig::QWEN38_27B,
        artifact_f16(artifact, &format!("{prefix}.q_norm.weight"))?,
    )?;
    let key_norm = runtime.prepare_qwen_rms_norm_f16(
        CudaRmsNormConfig {
            rows: config.num_key_value_heads,
            columns: config.head_dim,
            epsilon: config.rms_norm_epsilon,
        },
        artifact_f16(artifact, &format!("{prefix}.k_norm.weight"))?,
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
    Ok(PreparedCudaFullAttentionLayer {
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
}
