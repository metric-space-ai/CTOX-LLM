//! Artifact-resource binding for the frozen Qwen3.8-27B Metal decode graph.
//!
//! The binding deliberately owns logical tensor names and recovery activation
//! groups, not a Metal allocation strategy. A future executor must prepare
//! these exact resources from the backend-neutral CTOXQ artifact. This keeps
//! Metal and CUDA on identical Q2/Q4 codes while still allowing different
//! physical buffer layouts.

use std::collections::{BTreeMap, BTreeSet};

use crate::backend::metal_schedule::{
    MetalDecodeOperation, MetalDecodeSchedule, MetalDecodeStep, MetalNormBinding,
};
use crate::config::LayerKind;
use crate::fanout::qwen38_fanout_groups;
use crate::tensor_contract::{expected_tensor_contract, TensorClass};
use crate::{EngineError, Qwen38Config, Result};

const EMBEDDING_MATRIX: &str = "model.language_model.embed_tokens.weight";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalProjectionGroupPlan {
    pub key: String,
    pub projection_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalProjectionPlan {
    groups: Vec<MetalProjectionGroupPlan>,
    projection_groups: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetalPreparedResource {
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
pub struct MetalBoundDecodeStep {
    pub schedule_index: usize,
    pub layer: Option<usize>,
    pub operation: MetalDecodeOperation,
    pub resources: Vec<MetalPreparedResource>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDecodeBindingPlan {
    steps: Vec<MetalBoundDecodeStep>,
}

impl MetalProjectionPlan {
    pub fn qwen38(config: &Qwen38Config) -> Result<Self> {
        if config != &Qwen38Config::default() {
            return Err(EngineError::Shape(
                "Metal projection plan requires the frozen Qwen3.8-27B topology".into(),
            ));
        }
        let contract = expected_tensor_contract(config);
        let quantized: BTreeSet<String> = contract
            .iter()
            .filter(|(_, spec)| spec.class == TensorClass::QuantizedMatrix)
            .map(|(name, _)| name.clone())
            .collect();
        if !quantized.contains(EMBEDDING_MATRIX) {
            return Err(EngineError::InvalidState(
                "Metal graph contract is missing the embedding matrix".into(),
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
                        "Metal activation group {key} references non-projection {name}"
                    )));
                }
                if projection_groups
                    .insert(name.clone(), key.clone())
                    .is_some()
                {
                    return Err(EngineError::InvalidState(format!(
                        "Metal projection {name} belongs to multiple activation groups"
                    )));
                }
            }
            groups.push(MetalProjectionGroupPlan {
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
            groups.push(MetalProjectionGroupPlan {
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

    pub fn groups(&self) -> &[MetalProjectionGroupPlan] {
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
                    "Metal projection plan does not contain projection {name}"
                ))
            })
    }

    fn validate(&self) -> Result<()> {
        if self.groups.is_empty() || self.projection_groups.is_empty() {
            return Err(EngineError::InvalidState(
                "Metal projection graph cannot be empty".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for group in &self.groups {
            if group.key.is_empty() || group.projection_names.is_empty() {
                return Err(EngineError::InvalidState(
                    "Metal projection group has no key or projections".into(),
                ));
            }
            for name in &group.projection_names {
                if !seen.insert(name.as_str()) {
                    return Err(EngineError::InvalidState(format!(
                        "Metal projection {name} occurs more than once"
                    )));
                }
                if self.projection_groups.get(name) != Some(&group.key) {
                    return Err(EngineError::InvalidState(format!(
                        "Metal projection {name} has inconsistent activation ownership"
                    )));
                }
            }
        }
        if seen.len() != self.projection_groups.len() || seen.contains(EMBEDDING_MATRIX) {
            return Err(EngineError::InvalidState(
                "Metal projection ownership is incomplete or includes embedding".into(),
            ));
        }
        Ok(())
    }
}

impl MetalDecodeBindingPlan {
    pub fn qwen38(
        schedule: &MetalDecodeSchedule,
        projections: &MetalProjectionPlan,
        config: &Qwen38Config,
    ) -> Result<Self> {
        schedule.validate()?;
        let steps = schedule
            .steps
            .iter()
            .enumerate()
            .map(|(index, step)| bind_decode_step(index, step, projections, config))
            .collect::<Result<Vec<_>>>()?;
        let plan = Self { steps };
        plan.validate_complete_ownership(projections, config)?;
        Ok(plan)
    }

    pub fn steps(&self) -> &[MetalBoundDecodeStep] {
        &self.steps
    }

    pub fn resource_count(&self, expected: fn(&MetalPreparedResource) -> bool) -> usize {
        self.steps
            .iter()
            .flat_map(|step| &step.resources)
            .filter(|resource| expected(resource))
            .collect::<BTreeSet<_>>()
            .len()
    }

    fn validate_complete_ownership(
        &self,
        projections: &MetalProjectionPlan,
        config: &Qwen38Config,
    ) -> Result<()> {
        if self.steps.len() != 645 {
            return Err(EngineError::InvalidState(format!(
                "Metal decode binding has {} steps, expected 645",
                self.steps.len()
            )));
        }
        let resources: BTreeSet<_> = self
            .steps
            .iter()
            .flat_map(|step| step.resources.iter().cloned())
            .collect();

        let bound_projections: BTreeSet<_> = resources
            .iter()
            .filter_map(|resource| match resource {
                MetalPreparedResource::Projection(name) => Some(name.clone()),
                _ => None,
            })
            .collect();
        let expected_projections = projections.projection_groups.keys().cloned().collect();
        compare_resource_set("projection", &bound_projections, &expected_projections)?;

        let bound_activations: BTreeSet<_> = resources
            .iter()
            .filter_map(|resource| match resource {
                MetalPreparedResource::Activation(key) => Some(key.clone()),
                _ => None,
            })
            .collect();
        let expected_activations = projections
            .groups
            .iter()
            .map(|group| group.key.clone())
            .collect();
        compare_resource_set("activation", &bound_activations, &expected_activations)?;

        let bound_linear: BTreeSet<_> = resources
            .iter()
            .filter_map(|resource| match resource {
                MetalPreparedResource::LinearMixer(layer) => Some(*layer),
                _ => None,
            })
            .collect();
        let expected_linear = (0..config.num_hidden_layers)
            .filter(|layer| config.layer_kind(*layer) == Some(LayerKind::LinearAttention))
            .collect();
        compare_resource_set("linear mixer", &bound_linear, &expected_linear)?;

        let bound_attention: BTreeSet<_> = resources
            .iter()
            .filter_map(|resource| match resource {
                MetalPreparedResource::FullAttention(key) => Some(key.clone()),
                _ => None,
            })
            .collect();
        let mut expected_attention: BTreeSet<_> = (0..config.num_hidden_layers)
            .filter(|layer| config.layer_kind(*layer) == Some(LayerKind::FullAttention))
            .map(|layer| format!("target:{layer}"))
            .collect();
        expected_attention.insert("mtp:0".into());
        compare_resource_set("full attention", &bound_attention, &expected_attention)?;

        let bound_regular_norms: BTreeSet<_> = resources
            .iter()
            .filter_map(|resource| match resource {
                MetalPreparedResource::RegularNorm(key) => Some(key.clone()),
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
                MetalPreparedResource::ResidualNorm(key) => Some(key.clone()),
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

        if !resources.contains(&MetalPreparedResource::Embedding)
            || !resources.contains(&MetalPreparedResource::TokenBarrier)
        {
            return Err(EngineError::InvalidState(
                "Metal decode binding omits embedding or token barrier".into(),
            ));
        }
        Ok(())
    }
}

fn bind_decode_step(
    schedule_index: usize,
    step: &MetalDecodeStep,
    projections: &MetalProjectionPlan,
    config: &Qwen38Config,
) -> Result<MetalBoundDecodeStep> {
    if step.operation != MetalDecodeOperation::RmsNorm
        && step.operation != MetalDecodeOperation::ResidualRmsNorm
        && step.norm.is_some()
    {
        return Err(EngineError::InvalidState(format!(
            "Metal decode step {schedule_index} attaches a norm to {:?}",
            step.operation
        )));
    }
    let mut resources = Vec::new();
    match step.operation {
        MetalDecodeOperation::Embedding => {
            require_global_step(schedule_index, step)?;
            resources.push(MetalPreparedResource::Embedding);
        }
        MetalDecodeOperation::RmsNorm => {
            if step.layer != Some(0) || step.norm != Some(MetalNormBinding::LayerInput(0)) {
                return Err(EngineError::InvalidState(format!(
                    "Metal decode step {schedule_index} is not the frozen initial norm"
                )));
            }
            resources.push(MetalPreparedResource::RegularNorm("target:initial".into()));
        }
        MetalDecodeOperation::FullAttentionFanout => {
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
        MetalDecodeOperation::QueryGateNormRope
        | MetalDecodeOperation::KeyRope
        | MetalDecodeOperation::PagedKvAppend
        | MetalDecodeOperation::PagedGqa => {
            let layer = require_layer_kind(schedule_index, step, config, LayerKind::FullAttention)?;
            resources.push(MetalPreparedResource::FullAttention(format!(
                "target:{layer}"
            )));
        }
        MetalDecodeOperation::AttentionGateOutputProjection => {
            let layer = require_layer_kind(schedule_index, step, config, LayerKind::FullAttention)?;
            resources.push(MetalPreparedResource::FullAttention(format!(
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
        MetalDecodeOperation::LinearFanout => {
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
        MetalDecodeOperation::CausalConvolution
        | MetalDecodeOperation::GatedDeltaPrepare
        | MetalDecodeOperation::GatedDeltaRecurrent
        | MetalDecodeOperation::GatedRmsNorm => {
            let layer =
                require_layer_kind(schedule_index, step, config, LayerKind::LinearAttention)?;
            resources.push(MetalPreparedResource::LinearMixer(layer));
        }
        MetalDecodeOperation::LinearOutputProjection => {
            let layer =
                require_layer_kind(schedule_index, step, config, LayerKind::LinearAttention)?;
            resources.push(MetalPreparedResource::LinearMixer(layer));
            add_projection_resources(
                projections,
                [format!(
                    "model.language_model.layers.{layer}.linear_attn.out_proj.weight"
                )],
                &mut resources,
            )?;
        }
        MetalDecodeOperation::ResidualRmsNorm => {
            let layer = require_layer(schedule_index, step, config)?;
            let key = match step.norm {
                Some(MetalNormBinding::LayerPostAttention(bound)) if bound == layer => {
                    format!("target:{layer}:post_attention")
                }
                Some(MetalNormBinding::LayerInput(next)) if next == layer + 1 => {
                    format!("target:{layer}:post_ffn:layer_{next}")
                }
                Some(MetalNormBinding::Final) if layer + 1 == config.num_hidden_layers => {
                    format!("target:{layer}:post_ffn:final")
                }
                _ => {
                    return Err(EngineError::InvalidState(format!(
                        "Metal residual norm at step {schedule_index} has incompatible binding {:?}",
                        step.norm
                    )));
                }
            };
            resources.push(MetalPreparedResource::ResidualNorm(key));
        }
        MetalDecodeOperation::FfnGateUpFanout => {
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
        MetalDecodeOperation::SwiGluDownProjection => {
            let layer = require_layer(schedule_index, step, config)?;
            add_projection_resources(
                projections,
                [format!(
                    "model.language_model.layers.{layer}.mlp.down_proj.weight"
                )],
                &mut resources,
            )?;
        }
        MetalDecodeOperation::LmHead => {
            require_global_step(schedule_index, step)?;
            add_projection_resources(projections, ["lm_head.weight".into()], &mut resources)?;
        }
        MetalDecodeOperation::MtpDraftAndTargetVerify => {
            require_global_step(schedule_index, step)?;
            resources.extend([
                MetalPreparedResource::Embedding,
                MetalPreparedResource::FullAttention("mtp:0".into()),
                MetalPreparedResource::RegularNorm("mtp:pre_embedding".into()),
                MetalPreparedResource::RegularNorm("mtp:pre_hidden".into()),
                MetalPreparedResource::RegularNorm("mtp:input".into()),
                MetalPreparedResource::ResidualNorm("mtp:post_attention".into()),
                MetalPreparedResource::ResidualNorm("mtp:final".into()),
            ]);
            add_projection_resources(
                projections,
                [
                    "mtp.fc.weight".into(),
                    "mtp.layers.0.self_attn.q_proj.weight".into(),
                    "mtp.layers.0.self_attn.k_proj.weight".into(),
                    "mtp.layers.0.self_attn.v_proj.weight".into(),
                    "mtp.layers.0.self_attn.o_proj.weight".into(),
                    "mtp.layers.0.mlp.gate_proj.weight".into(),
                    "mtp.layers.0.mlp.up_proj.weight".into(),
                    "mtp.layers.0.mlp.down_proj.weight".into(),
                    "lm_head.weight".into(),
                ],
                &mut resources,
            )?;
        }
        MetalDecodeOperation::TokenCommandBufferCommit => {
            require_global_step(schedule_index, step)?;
            resources.push(MetalPreparedResource::TokenBarrier);
        }
    }
    resources.sort();
    resources.dedup();
    if resources.is_empty() {
        return Err(EngineError::InvalidState(format!(
            "Metal decode step {schedule_index} has no prepared resource"
        )));
    }
    Ok(MetalBoundDecodeStep {
        schedule_index,
        layer: step.layer,
        operation: step.operation,
        resources,
    })
}

fn add_projection_resources<const N: usize>(
    projections: &MetalProjectionPlan,
    names: [String; N],
    resources: &mut Vec<MetalPreparedResource>,
) -> Result<()> {
    for name in names {
        let group = projections.group_for_projection(&name)?;
        resources.push(MetalPreparedResource::Activation(group.to_owned()));
        resources.push(MetalPreparedResource::Projection(name));
    }
    Ok(())
}

fn require_global_step(schedule_index: usize, step: &MetalDecodeStep) -> Result<()> {
    if step.layer.is_some() {
        return Err(EngineError::InvalidState(format!(
            "Metal global decode step {schedule_index} carries a layer"
        )));
    }
    Ok(())
}

fn require_layer(
    schedule_index: usize,
    step: &MetalDecodeStep,
    config: &Qwen38Config,
) -> Result<usize> {
    let layer = step.layer.ok_or_else(|| {
        EngineError::InvalidState(format!("Metal decode step {schedule_index} has no layer"))
    })?;
    if layer >= config.num_hidden_layers {
        return Err(EngineError::InvalidState(format!(
            "Metal decode step {schedule_index} references layer {layer}"
        )));
    }
    Ok(layer)
}

fn require_layer_kind(
    schedule_index: usize,
    step: &MetalDecodeStep,
    config: &Qwen38Config,
    expected: LayerKind,
) -> Result<usize> {
    let layer = require_layer(schedule_index, step, config)?;
    if config.layer_kind(layer) != Some(expected) {
        return Err(EngineError::InvalidState(format!(
            "Metal decode step {schedule_index} binds {expected:?} resources to layer {layer}"
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
            "Metal decode binding owns {} {label} resources, expected {}",
            actual.len(),
            expected.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_plan_excludes_embedding_but_includes_shared_lm_head() {
        let plan = MetalProjectionPlan::qwen38(&Qwen38Config::default()).unwrap();
        assert_eq!(plan.projection_count(), 505);
        assert_eq!(plan.group_count(), 262);
        assert!(plan.group_for_projection("lm_head.weight").is_ok());
        assert!(plan.group_for_projection(EMBEDDING_MATRIX).is_err());
        assert_eq!(
            plan.groups()
                .iter()
                .map(|group| group.projection_names.len())
                .sum::<usize>(),
            plan.projection_count()
        );
    }

    #[test]
    fn complete_metal_binding_owns_every_model_resource() {
        let config = Qwen38Config::default();
        let projections = MetalProjectionPlan::qwen38(&config).unwrap();
        let schedule = MetalDecodeSchedule::qwen38(&config).unwrap();
        let bindings = MetalDecodeBindingPlan::qwen38(&schedule, &projections, &config).unwrap();
        assert_eq!(bindings.steps().len(), 645);
        assert_eq!(
            bindings.resource_count(|resource| matches!(
                resource,
                MetalPreparedResource::Projection(_)
            )),
            505
        );
        assert_eq!(
            bindings.resource_count(|resource| matches!(
                resource,
                MetalPreparedResource::Activation(_)
            )),
            262
        );
        assert_eq!(
            bindings.resource_count(|resource| matches!(
                resource,
                MetalPreparedResource::LinearMixer(_)
            )),
            48
        );
        assert_eq!(
            bindings.resource_count(|resource| matches!(
                resource,
                MetalPreparedResource::FullAttention(_)
            )),
            17
        );
        assert_eq!(
            bindings.resource_count(|resource| matches!(
                resource,
                MetalPreparedResource::RegularNorm(_)
            )),
            4
        );
        assert_eq!(
            bindings.resource_count(|resource| matches!(
                resource,
                MetalPreparedResource::ResidualNorm(_)
            )),
            130
        );
    }
}
