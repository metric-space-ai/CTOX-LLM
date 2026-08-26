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
    CudaCandidateRuntime, PreparedCudaA8Activation, PreparedCudaA8Projection,
};
use crate::backend::{Activation, ScaleSlice};
use crate::fanout::qwen38_fanout_groups;
use crate::loader::ModelArtifact;
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
    activations: BTreeMap<String, PreparedCudaA8Activation>,
    projections: BTreeMap<String, PreparedCudaA8Projection>,
    model_bytes: u64,
    graph_bytes: u64,
}

impl PreparedCudaProjectionGraph {
    pub fn prepare(
        runtime: &CudaCandidateRuntime,
        artifact: &ModelArtifact,
        config: &Qwen38Config,
    ) -> Result<Self> {
        validate_tensor_contract(artifact.manifest(), config)?;
        let plan = CudaProjectionPlan::qwen38(config)?;
        let mut activations = BTreeMap::new();
        let mut projections = BTreeMap::new();
        let mut model_bytes = 0_u64;
        let mut graph_bytes = 0_u64;

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
        Ok(Self {
            artifact: artifact.clone(),
            plan,
            activations,
            projections,
            model_bytes,
            graph_bytes,
        })
    }

    pub fn artifact_manifest_sha256(&self) -> &str {
        self.artifact.manifest_sha256()
    }

    pub fn plan(&self) -> &CudaProjectionPlan {
        &self.plan
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

    pub fn model_bytes(&self) -> u64 {
        self.model_bytes
    }

    pub fn graph_bytes(&self) -> u64 {
        self.graph_bytes
    }

    pub fn resident_bytes(&self) -> Result<u64> {
        checked_add(self.model_bytes, self.graph_bytes, "CUDA resident bytes")
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
