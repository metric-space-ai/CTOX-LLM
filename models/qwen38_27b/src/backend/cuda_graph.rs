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
    CudaCandidateRuntime, CudaCausalConvConfig, CudaDeviceF32View, CudaGatedDeltaConfig,
    CudaGatedRmsNormConfig, CudaPagedGqaConfig, CudaPartialRopeConfig, CudaQueryGateConfig,
    CudaRmsNormConfig, PreparedCudaA8Activation, PreparedCudaA8Projection, PreparedCudaCausalConv,
    PreparedCudaEmbedding, PreparedCudaF32Checkpoint, PreparedCudaF32Concat,
    PreparedCudaGatedDelta, PreparedCudaGatedDeltaInputs, PreparedCudaGatedRmsNorm,
    PreparedCudaPagedGqa, PreparedCudaPartialRope, PreparedCudaQueryGate,
    PreparedCudaResidualRmsNorm, PreparedCudaRmsNorm,
};
use crate::backend::cuda_schedule::{
    CudaDecodeOperation, CudaDecodeSchedule, CudaDecodeStep, CudaNormBinding,
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
    embedding: PreparedCudaEmbedding,
    activations: BTreeMap<String, PreparedCudaA8Activation>,
    projections: BTreeMap<String, PreparedCudaA8Projection>,
    linear_mixers: BTreeMap<usize, PreparedCudaLinearMixerLayer>,
    full_attention: BTreeMap<String, PreparedCudaFullAttentionLayer>,
    norms: PreparedCudaNormGraph,
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
        let decode_schedule = CudaDecodeSchedule::qwen38(config)?;
        let decode_bindings = CudaDecodeBindingPlan::qwen38(&decode_schedule, &plan, config)?;
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
                    convolution.speculative_checkpoint_bytes(),
                    recurrence.speculative_checkpoint_bytes(),
                ],
                "CUDA linear mixer session bytes",
            )?;
            speculative_checkpoint_bytes = checked_add(
                speculative_checkpoint_bytes,
                sum_usize(
                    [
                        convolution.speculative_checkpoint_bytes(),
                        recurrence.speculative_checkpoint_bytes(),
                    ],
                    "CUDA linear checkpoint bytes",
                )?,
                "CUDA speculative checkpoint bytes",
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
        let graph = Self {
            artifact: artifact.clone(),
            plan,
            decode_bindings,
            embedding,
            activations,
            projections,
            linear_mixers,
            full_attention,
            norms,
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
            dispatch_projection_device(
                runtime,
                plan,
                activations,
                projections,
                final_hidden,
                "lm_head.weight",
            )
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
        for step in self.decode_bindings.steps() {
            for resource in &step.resources {
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
