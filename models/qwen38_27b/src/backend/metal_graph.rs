//! Artifact-resource binding for the frozen Qwen3.8-27B Metal decode graph.
//!
//! The binding deliberately owns logical tensor names and recovery activation
//! groups, not a Metal allocation strategy. A future executor must prepare
//! these exact resources from the backend-neutral CTOXQ artifact. This keeps
//! Metal and CUDA on identical Q2/Q4 codes while still allowing different
//! physical buffer layouts.

use std::collections::{BTreeMap, BTreeSet};

use crate::backend::metal_schedule::{
    MetalBufferSlot, MetalDecodeOperation, MetalDecodeSchedule, MetalDecodeStep, MetalNormBinding,
};
use crate::config::LayerKind;
use crate::fanout::qwen38_fanout_groups;
use crate::tensor_contract::{expected_tensor_contract, TensorClass};
use crate::{EngineError, Qwen38Config, Result};

const EMBEDDING_MATRIX: &str = "model.language_model.embed_tokens.weight";
pub const METAL_MTP4_RECORDS: usize = 4;

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
    pub reads: Vec<MetalBufferSlot>,
    pub writes: Vec<MetalBufferSlot>,
    pub resources: Vec<MetalPreparedResource>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDecodeBindingPlan {
    steps: Vec<MetalBoundDecodeStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalDecodeBufferBinding {
    pub slot: MetalBufferSlot,
    pub values: usize,
    pub offset: usize,
    pub bytes: usize,
}

/// Static alias plan for every decode activation in one shared Metal buffer.
///
/// Persistent weights, KV pages, convolution/recurrent state, and tiny kernel
/// parameter blocks are not part of this arena. The plan covers exactly the
/// device-resident values named by [`MetalDecodeSchedule`]. Slots alias only
/// when none of their produced-value live intervals overlap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDecodeWorkspacePlan {
    bindings: BTreeMap<MetalBufferSlot, MetalDecodeBufferBinding>,
    total_bytes: usize,
    independent_bytes: usize,
    alignment: usize,
    mtp_draft_vocabulary_rows: usize,
}

/// Private activation slots used while expanding the single scheduled MTP
/// operation into its native frontend, transformer layer, and final norm.
/// Target logits and the final restricted draft remain in the decode arena;
/// this arena exists so neither live tensor can be aliased by MTP scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetalMtpBufferSlot {
    SelectedEmbedding,
    NormalizedEmbedding,
    NormalizedTargetHidden,
    Concatenated,
    HiddenA,
    Normalized,
    QueryGate,
    Query,
    Key,
    Value,
    AttentionGate,
    AttentionOutput,
    MixerOutput,
    ResidualHidden,
    ResidualNormalized,
    FfnGate,
    FfnUp,
    FfnDown,
    FinalHidden,
    FinalNormalized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalMtpBufferBinding {
    pub slot: MetalMtpBufferSlot,
    pub values: usize,
    pub offset: usize,
    pub bytes: usize,
}

/// Alias-safe workspace for the native one-token MTP subgraph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalMtpWorkspacePlan {
    bindings: BTreeMap<MetalMtpBufferSlot, MetalMtpBufferBinding>,
    total_bytes: usize,
    independent_bytes: usize,
    alignment: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LiveInterval {
    start: usize,
    end: usize,
}

/// Fail-closed progress tracker for one Metal token command buffer.
///
/// The future executor advances this cursor only after an operation has been
/// encoded successfully. Dropping an incomplete cursor cannot produce a new
/// committed token position; the executor must reject the matching
/// graph-wide speculative-state transaction before reusing the session.
#[derive(Debug)]
pub struct MetalDecodeExecutionCursor<'a> {
    plan: &'a MetalDecodeBindingPlan,
    token_position: usize,
    next_step: usize,
}

/// Four provisional one-token cursors sharing one physical Metal completion
/// barrier. Encoding a record never publishes its provisional position; only
/// consuming all four final barriers after the shared command buffer completed
/// can return the block's new scheduler-visible position.
#[derive(Debug)]
pub struct MetalMtp4ExecutionCursor<'a> {
    records: [MetalDecodeExecutionCursor<'a>; METAL_MTP4_RECORDS],
    start_position: usize,
    next_record: usize,
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

impl MetalDecodeWorkspacePlan {
    pub const ALIGNMENT: usize = 256;

    pub fn qwen38(
        schedule: &MetalDecodeSchedule,
        config: &Qwen38Config,
        mtp_draft_vocabulary_rows: usize,
    ) -> Result<Self> {
        schedule.validate()?;
        if config != &Qwen38Config::default() {
            return Err(EngineError::Shape(
                "Metal decode workspace requires the frozen Qwen3.8-27B topology".into(),
            ));
        }
        if mtp_draft_vocabulary_rows == 0 || mtp_draft_vocabulary_rows > config.vocab_size {
            return Err(EngineError::Shape(format!(
                "Metal MTP draft vocabulary has {mtp_draft_vocabulary_rows} rows, expected 1..={}",
                config.vocab_size
            )));
        }

        let widths = decode_slot_widths(config, mtp_draft_vocabulary_rows)?;
        let intervals = decode_live_intervals(schedule)?;
        let scheduled_slots: BTreeSet<_> = schedule
            .steps
            .iter()
            .flat_map(|step| step.reads.iter().chain(&step.writes).copied())
            .collect();
        if scheduled_slots != widths.keys().copied().collect()
            || scheduled_slots != intervals.keys().copied().collect()
        {
            return Err(EngineError::InvalidState(
                "Metal workspace widths or liveness omit a scheduled slot".into(),
            ));
        }

        let mut ordered = widths
            .iter()
            .map(|(&slot, &values)| {
                let bytes = values
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        EngineError::MemoryBudget(format!(
                            "Metal workspace byte count overflows for {slot:?}"
                        ))
                    })?;
                Ok((slot, values, bytes))
            })
            .collect::<Result<Vec<_>>>()?;
        ordered.sort_by_key(|(slot, _values, bytes)| (std::cmp::Reverse(*bytes), *slot));

        let mut bindings: BTreeMap<MetalBufferSlot, MetalDecodeBufferBinding> = BTreeMap::new();
        for (slot, values, bytes) in ordered {
            let mut offset = 0_usize;
            loop {
                offset = align_up(offset, Self::ALIGNMENT)?;
                let end = offset.checked_add(bytes).ok_or_else(|| {
                    EngineError::MemoryBudget("Metal workspace range overflows".into())
                })?;
                let mut conflict_end = None;
                for (&other_slot, other) in &bindings {
                    let other_end = other.offset.checked_add(other.bytes).ok_or_else(|| {
                        EngineError::MemoryBudget("Metal workspace binding end overflows".into())
                    })?;
                    if intervals_overlap(&intervals[&slot], &intervals[&other_slot])
                        && byte_ranges_overlap(offset, end, other.offset, other_end)
                    {
                        conflict_end = Some(conflict_end.unwrap_or(0_usize).max(other_end));
                    }
                }
                if let Some(next) = conflict_end {
                    offset = next;
                    continue;
                }
                bindings.insert(
                    slot,
                    MetalDecodeBufferBinding {
                        slot,
                        values,
                        offset,
                        bytes,
                    },
                );
                break;
            }
        }
        let total_bytes = bindings.values().try_fold(0_usize, |maximum, binding| {
            binding
                .offset
                .checked_add(binding.bytes)
                .map(|end| maximum.max(end))
                .ok_or_else(|| EngineError::MemoryBudget("Metal workspace end overflows".into()))
        })?;
        let total_bytes = align_up(total_bytes, Self::ALIGNMENT)?;
        let independent_bytes = bindings.values().try_fold(0_usize, |total, binding| {
            total.checked_add(binding.bytes).ok_or_else(|| {
                EngineError::MemoryBudget("Metal independent workspace bytes overflow".into())
            })
        })?;
        let plan = Self {
            bindings,
            total_bytes,
            independent_bytes,
            alignment: Self::ALIGNMENT,
            mtp_draft_vocabulary_rows,
        };
        plan.validate(&intervals)?;
        Ok(plan)
    }

    pub fn binding(&self, slot: MetalBufferSlot) -> Result<MetalDecodeBufferBinding> {
        self.bindings.get(&slot).copied().ok_or_else(|| {
            EngineError::InvalidState(format!("Metal workspace does not bind {slot:?}"))
        })
    }

    pub fn bindings(&self) -> impl Iterator<Item = MetalDecodeBufferBinding> + '_ {
        self.bindings.values().copied()
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn independent_bytes(&self) -> usize {
        self.independent_bytes
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn mtp_draft_vocabulary_rows(&self) -> usize {
        self.mtp_draft_vocabulary_rows
    }

    fn validate(&self, intervals: &BTreeMap<MetalBufferSlot, Vec<LiveInterval>>) -> Result<()> {
        if self.bindings.len() != 21
            || self.total_bytes == 0
            || !self.total_bytes.is_multiple_of(self.alignment)
            || self.total_bytes > self.independent_bytes
        {
            return Err(EngineError::MemoryBudget(
                "Metal shared decode workspace has an invalid geometry".into(),
            ));
        }
        let bindings: Vec<_> = self.bindings.values().collect();
        for (index, left) in bindings.iter().enumerate() {
            let left_end = left.offset.checked_add(left.bytes).ok_or_else(|| {
                EngineError::MemoryBudget("Metal workspace binding end overflows".into())
            })?;
            if left.bytes == 0
                || !left.offset.is_multiple_of(self.alignment)
                || left_end > self.total_bytes
            {
                return Err(EngineError::MemoryBudget(format!(
                    "Metal workspace binding {:?} is invalid",
                    left.slot
                )));
            }
            for right in &bindings[index + 1..] {
                let right_end = right.offset.checked_add(right.bytes).ok_or_else(|| {
                    EngineError::MemoryBudget("Metal workspace binding end overflows".into())
                })?;
                if intervals_overlap(&intervals[&left.slot], &intervals[&right.slot])
                    && byte_ranges_overlap(left.offset, left_end, right.offset, right_end)
                {
                    return Err(EngineError::InvalidState(format!(
                        "live Metal slots {:?} and {:?} alias",
                        left.slot, right.slot
                    )));
                }
            }
        }
        Ok(())
    }
}

impl MetalMtpWorkspacePlan {
    pub const ALIGNMENT: usize = MetalDecodeWorkspacePlan::ALIGNMENT;

    pub fn qwen38(config: &Qwen38Config) -> Result<Self> {
        if config != &Qwen38Config::default() {
            return Err(EngineError::Shape(
                "Metal MTP workspace requires the frozen Qwen3.8-27B topology".into(),
            ));
        }
        let widths = mtp_slot_widths(config)?;
        let intervals = mtp_live_intervals();
        if widths.keys().copied().collect::<BTreeSet<_>>()
            != intervals.keys().copied().collect::<BTreeSet<_>>()
        {
            return Err(EngineError::InvalidState(
                "Metal MTP workspace widths or liveness omit a scratch slot".into(),
            ));
        }

        let mut ordered = widths
            .iter()
            .map(|(&slot, &values)| {
                let bytes = values
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        EngineError::MemoryBudget(format!(
                            "Metal MTP workspace byte count overflows for {slot:?}"
                        ))
                    })?;
                Ok((slot, values, bytes))
            })
            .collect::<Result<Vec<_>>>()?;
        ordered.sort_by_key(|(slot, _values, bytes)| (std::cmp::Reverse(*bytes), *slot));

        let mut bindings = BTreeMap::new();
        for (slot, values, bytes) in ordered {
            let mut offset = 0_usize;
            loop {
                offset = align_up(offset, Self::ALIGNMENT)?;
                let end = offset.checked_add(bytes).ok_or_else(|| {
                    EngineError::MemoryBudget("Metal MTP workspace range overflows".into())
                })?;
                let mut conflict_end = None;
                for (&other_slot, other) in &bindings {
                    let other: &MetalMtpBufferBinding = other;
                    let other_end = other.offset.checked_add(other.bytes).ok_or_else(|| {
                        EngineError::MemoryBudget(
                            "Metal MTP workspace binding end overflows".into(),
                        )
                    })?;
                    if intervals_overlap(&intervals[&slot], &intervals[&other_slot])
                        && byte_ranges_overlap(offset, end, other.offset, other_end)
                    {
                        conflict_end = Some(conflict_end.unwrap_or(0_usize).max(other_end));
                    }
                }
                if let Some(next) = conflict_end {
                    offset = next;
                    continue;
                }
                bindings.insert(
                    slot,
                    MetalMtpBufferBinding {
                        slot,
                        values,
                        offset,
                        bytes,
                    },
                );
                break;
            }
        }
        let total_bytes = bindings.values().try_fold(0_usize, |maximum, binding| {
            binding
                .offset
                .checked_add(binding.bytes)
                .map(|end| maximum.max(end))
                .ok_or_else(|| {
                    EngineError::MemoryBudget("Metal MTP workspace end overflows".into())
                })
        })?;
        let total_bytes = align_up(total_bytes, Self::ALIGNMENT)?;
        let independent_bytes = bindings.values().try_fold(0_usize, |total, binding| {
            total.checked_add(binding.bytes).ok_or_else(|| {
                EngineError::MemoryBudget("Metal independent MTP workspace bytes overflow".into())
            })
        })?;
        let plan = Self {
            bindings,
            total_bytes,
            independent_bytes,
            alignment: Self::ALIGNMENT,
        };
        plan.validate(&intervals)?;
        Ok(plan)
    }

    pub fn binding(&self, slot: MetalMtpBufferSlot) -> Result<MetalMtpBufferBinding> {
        self.bindings.get(&slot).copied().ok_or_else(|| {
            EngineError::InvalidState(format!("Metal MTP workspace does not bind {slot:?}"))
        })
    }

    pub fn bindings(&self) -> impl Iterator<Item = MetalMtpBufferBinding> + '_ {
        self.bindings.values().copied()
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn independent_bytes(&self) -> usize {
        self.independent_bytes
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    fn validate(&self, intervals: &BTreeMap<MetalMtpBufferSlot, Vec<LiveInterval>>) -> Result<()> {
        if self.bindings.len() != 20
            || self.total_bytes == 0
            || !self.total_bytes.is_multiple_of(self.alignment)
            || self.total_bytes > self.independent_bytes
        {
            return Err(EngineError::MemoryBudget(
                "Metal shared MTP workspace has an invalid geometry".into(),
            ));
        }
        let bindings: Vec<_> = self.bindings.values().collect();
        for (index, left) in bindings.iter().enumerate() {
            let left_end = left.offset.checked_add(left.bytes).ok_or_else(|| {
                EngineError::MemoryBudget("Metal MTP workspace binding end overflows".into())
            })?;
            if left.bytes == 0
                || !left.offset.is_multiple_of(self.alignment)
                || left_end > self.total_bytes
            {
                return Err(EngineError::MemoryBudget(format!(
                    "Metal MTP workspace binding {:?} is invalid",
                    left.slot
                )));
            }
            for right in &bindings[index + 1..] {
                let right_end = right.offset.checked_add(right.bytes).ok_or_else(|| {
                    EngineError::MemoryBudget("Metal MTP workspace binding end overflows".into())
                })?;
                if intervals_overlap(&intervals[&left.slot], &intervals[&right.slot])
                    && byte_ranges_overlap(left.offset, left_end, right.offset, right_end)
                {
                    return Err(EngineError::InvalidState(format!(
                        "live Metal MTP slots {:?} and {:?} alias",
                        left.slot, right.slot
                    )));
                }
            }
        }
        Ok(())
    }
}

fn mtp_slot_widths(config: &Qwen38Config) -> Result<BTreeMap<MetalMtpBufferSlot, usize>> {
    let checked = |left: usize, right: usize, label: &str| {
        left.checked_mul(right)
            .ok_or_else(|| EngineError::MemoryBudget(format!("Metal MTP {label} overflows")))
    };
    let hidden = config.hidden_size;
    let query = checked(config.num_attention_heads, config.head_dim, "query width")?;
    let query_gate = checked(query, 2, "query/gate width")?;
    let key_value = checked(config.num_key_value_heads, config.head_dim, "K/V width")?;
    Ok(BTreeMap::from([
        (MetalMtpBufferSlot::SelectedEmbedding, hidden),
        (MetalMtpBufferSlot::NormalizedEmbedding, hidden),
        (MetalMtpBufferSlot::NormalizedTargetHidden, hidden),
        (
            MetalMtpBufferSlot::Concatenated,
            checked(hidden, 2, "concat width")?,
        ),
        (MetalMtpBufferSlot::HiddenA, hidden),
        (MetalMtpBufferSlot::Normalized, hidden),
        (MetalMtpBufferSlot::QueryGate, query_gate),
        (MetalMtpBufferSlot::Query, query),
        (MetalMtpBufferSlot::Key, key_value),
        (MetalMtpBufferSlot::Value, key_value),
        (MetalMtpBufferSlot::AttentionGate, query),
        (MetalMtpBufferSlot::AttentionOutput, query),
        (MetalMtpBufferSlot::MixerOutput, hidden),
        (MetalMtpBufferSlot::ResidualHidden, hidden),
        (MetalMtpBufferSlot::ResidualNormalized, hidden),
        (MetalMtpBufferSlot::FfnGate, config.intermediate_size),
        (MetalMtpBufferSlot::FfnUp, config.intermediate_size),
        (MetalMtpBufferSlot::FfnDown, hidden),
        (MetalMtpBufferSlot::FinalHidden, hidden),
        (MetalMtpBufferSlot::FinalNormalized, hidden),
    ]))
}

fn mtp_live_intervals() -> BTreeMap<MetalMtpBufferSlot, Vec<LiveInterval>> {
    use MetalMtpBufferSlot::*;
    BTreeMap::from([
        (SelectedEmbedding, vec![LiveInterval { start: 0, end: 1 }]),
        (NormalizedEmbedding, vec![LiveInterval { start: 1, end: 2 }]),
        (
            NormalizedTargetHidden,
            vec![LiveInterval { start: 1, end: 2 }],
        ),
        (Concatenated, vec![LiveInterval { start: 2, end: 3 }]),
        (HiddenA, vec![LiveInterval { start: 3, end: 9 }]),
        (Normalized, vec![LiveInterval { start: 4, end: 5 }]),
        (QueryGate, vec![LiveInterval { start: 5, end: 6 }]),
        (Query, vec![LiveInterval { start: 6, end: 7 }]),
        (Key, vec![LiveInterval { start: 5, end: 7 }]),
        (Value, vec![LiveInterval { start: 5, end: 7 }]),
        (AttentionGate, vec![LiveInterval { start: 6, end: 8 }]),
        (AttentionOutput, vec![LiveInterval { start: 7, end: 8 }]),
        (MixerOutput, vec![LiveInterval { start: 8, end: 9 }]),
        (ResidualHidden, vec![LiveInterval { start: 9, end: 12 }]),
        (ResidualNormalized, vec![LiveInterval { start: 9, end: 10 }]),
        (FfnGate, vec![LiveInterval { start: 10, end: 11 }]),
        (FfnUp, vec![LiveInterval { start: 10, end: 11 }]),
        (FfnDown, vec![LiveInterval { start: 11, end: 12 }]),
        (FinalHidden, vec![LiveInterval { start: 12, end: 13 }]),
        (FinalNormalized, vec![LiveInterval { start: 12, end: 13 }]),
    ])
}

fn decode_slot_widths(
    config: &Qwen38Config,
    mtp_draft_vocabulary_rows: usize,
) -> Result<BTreeMap<MetalBufferSlot, usize>> {
    let checked = |left: usize, right: usize, label: &str| {
        left.checked_mul(right)
            .ok_or_else(|| EngineError::MemoryBudget(format!("Metal {label} width overflows")))
    };
    let query = checked(config.num_attention_heads, config.head_dim, "query")?;
    let query_gate = checked(query, 2, "query/gate")?;
    let full_key_value = checked(config.num_key_value_heads, config.head_dim, "full K/V")?;
    let linear_key = checked(
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        "linear repeated key",
    )?;
    let linear_value = checked(
        config.linear_num_value_heads,
        config.linear_value_head_dim,
        "linear value",
    )?;
    let linear_native_key = checked(
        config.linear_num_key_heads,
        config.linear_key_head_dim,
        "linear native key",
    )?;
    let linear_qkv = checked(linear_native_key, 2, "linear Q/K")?
        .checked_add(linear_value)
        .ok_or_else(|| EngineError::MemoryBudget("Metal linear QKV width overflows".into()))?;
    Ok(BTreeMap::from([
        (MetalBufferSlot::HiddenA, config.hidden_size),
        (MetalBufferSlot::HiddenB, config.hidden_size),
        (MetalBufferSlot::Normalized, config.hidden_size),
        (MetalBufferSlot::QueryGate, query_gate),
        (MetalBufferSlot::Query, query.max(linear_key)),
        (MetalBufferSlot::Key, full_key_value.max(linear_key)),
        (MetalBufferSlot::Value, full_key_value.max(linear_value)),
        (MetalBufferSlot::AttentionGate, query),
        (MetalBufferSlot::AttentionOutput, query.max(linear_value)),
        (MetalBufferSlot::MixerOutput, config.hidden_size),
        (MetalBufferSlot::LinearQkv, linear_qkv),
        (MetalBufferSlot::LinearZ, linear_value),
        (MetalBufferSlot::LinearA, config.linear_num_value_heads),
        (MetalBufferSlot::LinearB, config.linear_num_value_heads),
        (MetalBufferSlot::LogDecay, config.linear_num_value_heads),
        (MetalBufferSlot::Beta, config.linear_num_value_heads),
        (MetalBufferSlot::FfnGate, config.intermediate_size),
        (MetalBufferSlot::FfnUp, config.intermediate_size),
        (MetalBufferSlot::FfnDown, config.hidden_size),
        (MetalBufferSlot::TargetLogits, config.vocab_size),
        (MetalBufferSlot::MtpDraft, mtp_draft_vocabulary_rows),
    ]))
}

fn decode_live_intervals(
    schedule: &MetalDecodeSchedule,
) -> Result<BTreeMap<MetalBufferSlot, Vec<LiveInterval>>> {
    let mut intervals: BTreeMap<MetalBufferSlot, Vec<LiveInterval>> = BTreeMap::new();
    let mut active: BTreeMap<MetalBufferSlot, usize> = BTreeMap::new();
    for (step_index, step) in schedule.steps.iter().enumerate() {
        let reads: BTreeSet<_> = step.reads.iter().copied().collect();
        for slot in &step.reads {
            let interval_index = active.get(slot).copied().ok_or_else(|| {
                EngineError::InvalidState(format!(
                    "Metal workspace sees read-before-write for {slot:?} at step {step_index}"
                ))
            })?;
            intervals.get_mut(slot).expect("active interval exists")[interval_index].end =
                step_index;
        }
        for slot in &step.writes {
            if reads.contains(slot) {
                continue;
            }
            let slot_intervals = intervals.entry(*slot).or_default();
            slot_intervals.push(LiveInterval {
                start: step_index,
                end: step_index,
            });
            active.insert(*slot, slot_intervals.len() - 1);
        }
    }
    Ok(intervals)
}

fn intervals_overlap(left: &[LiveInterval], right: &[LiveInterval]) -> bool {
    left.iter().any(|left| {
        right
            .iter()
            .any(|right| left.start <= right.end && right.start <= left.end)
    })
}

fn byte_ranges_overlap(
    left_start: usize,
    left_end: usize,
    right_start: usize,
    right_end: usize,
) -> bool {
    left_start < right_end && right_start < left_end
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(EngineError::MemoryBudget(
            "Metal workspace alignment must be a power of two".into(),
        ));
    }
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| EngineError::MemoryBudget("Metal workspace alignment overflows".into()))
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

    pub fn execution_cursor(
        &self,
        token_position: usize,
        committed_tokens: usize,
        admitted_context: usize,
    ) -> Result<MetalDecodeExecutionCursor<'_>> {
        if token_position != committed_tokens {
            return Err(EngineError::InvalidState(format!(
                "Metal decode position is {token_position}, but {committed_tokens} tokens are committed"
            )));
        }
        if token_position >= admitted_context {
            return Err(EngineError::MemoryBudget(format!(
                "Metal decode position {token_position} exceeds admitted context {admitted_context}"
            )));
        }
        if self.steps.last().is_none_or(|step| {
            step.operation != MetalDecodeOperation::TokenCommandBufferCommit
                || step.schedule_index + 1 != self.steps.len()
        }) {
            return Err(EngineError::InvalidState(
                "Metal decode binding has no sole final command-buffer commit".into(),
            ));
        }
        Ok(MetalDecodeExecutionCursor {
            plan: self,
            token_position,
            next_step: 0,
        })
    }

    pub fn mtp4_execution_cursor(
        &self,
        token_position: usize,
        committed_tokens: usize,
        admitted_context: usize,
    ) -> Result<MetalMtp4ExecutionCursor<'_>> {
        if token_position != committed_tokens {
            return Err(EngineError::InvalidState(format!(
                "Metal MTP4 position is {token_position}, but {committed_tokens} tokens are committed"
            )));
        }
        let block_end = token_position
            .checked_add(METAL_MTP4_RECORDS)
            .ok_or_else(|| EngineError::MemoryBudget("Metal MTP4 position overflows".into()))?;
        if block_end > admitted_context {
            return Err(EngineError::MemoryBudget(format!(
                "Metal MTP4 block {token_position}..{block_end} exceeds admitted context {admitted_context}"
            )));
        }
        let records: [MetalDecodeExecutionCursor<'_>; METAL_MTP4_RECORDS] = (0..METAL_MTP4_RECORDS)
            .map(|record| {
                let position = token_position + record;
                self.execution_cursor(position, position, admitted_context)
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| {
                EngineError::InvalidState("Metal MTP4 cursor lost a logical record".into())
            })?;
        Ok(MetalMtp4ExecutionCursor {
            records,
            start_position: token_position,
            next_record: 0,
        })
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

impl<'a> MetalDecodeExecutionCursor<'a> {
    pub fn token_position(&self) -> usize {
        self.token_position
    }

    pub fn next_step(&self) -> Option<&'a MetalBoundDecodeStep> {
        self.plan.steps.get(self.next_step)
    }

    /// Records the exact operation that the executor successfully encoded.
    /// An omitted, duplicated, reordered, or wrong-layer dispatch fails before
    /// a token position can become committed.
    pub fn advance(
        &mut self,
        schedule_index: usize,
        layer: Option<usize>,
        operation: MetalDecodeOperation,
    ) -> Result<()> {
        if operation == MetalDecodeOperation::TokenCommandBufferCommit {
            return Err(EngineError::InvalidState(
                "Metal final commit requires commit_after_completion".into(),
            ));
        }
        let expected = self.next_step().ok_or_else(|| {
            EngineError::InvalidState("Metal token command buffer is already complete".into())
        })?;
        if expected.schedule_index != schedule_index
            || expected.layer != layer
            || expected.operation != operation
        {
            return Err(EngineError::InvalidState(format!(
                "Metal dispatch ({schedule_index}, {layer:?}, {operation:?}) does not match bound step ({}, {:?}, {:?})",
                expected.schedule_index, expected.layer, expected.operation
            )));
        }
        self.next_step += 1;
        Ok(())
    }

    /// Records the sole final barrier only after the caller has committed and
    /// waited for successful command-buffer completion.
    pub fn commit_after_completion(&mut self, schedule_index: usize) -> Result<()> {
        let expected = self.next_step().ok_or_else(|| {
            EngineError::InvalidState("Metal token command buffer is already complete".into())
        })?;
        if expected.schedule_index != schedule_index
            || expected.layer.is_some()
            || expected.operation != MetalDecodeOperation::TokenCommandBufferCommit
        {
            return Err(EngineError::InvalidState(format!(
                "Metal completed command buffer at step {schedule_index} does not match final bound step ({}, {:?}, {:?})",
                expected.schedule_index, expected.layer, expected.operation
            )));
        }
        self.next_step += 1;
        Ok(())
    }

    /// Returns the new committed position only after the final command buffer
    /// has completed and every bound step has advanced in exact order.
    pub fn finish(self) -> Result<usize> {
        if self.next_step != self.plan.steps.len() {
            return Err(EngineError::InvalidState(format!(
                "Metal token completed {} of {} bound steps",
                self.next_step,
                self.plan.steps.len()
            )));
        }
        self.token_position
            .checked_add(1)
            .ok_or_else(|| EngineError::MemoryBudget("Metal token position overflows".into()))
    }
}

impl MetalMtp4ExecutionCursor<'_> {
    /// Records the complete logical operation sequence for exactly the next
    /// MTP4 record while deliberately leaving that record's final barrier
    /// pending. Records cannot be skipped or encoded out of order.
    pub fn record_complete_encoded_record(&mut self, record_index: usize) -> Result<usize> {
        if record_index != self.next_record || record_index >= METAL_MTP4_RECORDS {
            return Err(EngineError::InvalidState(format!(
                "Metal MTP4 encoded record {record_index}, expected {}",
                self.next_record
            )));
        }
        let cursor = &mut self.records[record_index];
        loop {
            let step = cursor.next_step().cloned().ok_or_else(|| {
                EngineError::InvalidState(format!(
                    "Metal MTP4 record {record_index} omitted its final barrier"
                ))
            })?;
            if step.operation == MetalDecodeOperation::TokenCommandBufferCommit {
                if step.layer.is_some() {
                    return Err(EngineError::InvalidState(format!(
                        "Metal MTP4 record {record_index} final barrier belongs to a layer"
                    )));
                }
                self.next_record += 1;
                return Ok(step.schedule_index);
            }
            cursor.advance(step.schedule_index, step.layer, step.operation)?;
        }
    }

    /// Consumes all four pending logical barriers after the one shared physical
    /// command buffer completed. This method is valid only for a fully accepted
    /// block; partial branches drop this cursor and replay accepted records via
    /// ordinary complete-token cursors.
    pub fn commit_after_shared_completion(
        self,
        barrier_indices: [usize; METAL_MTP4_RECORDS],
    ) -> Result<usize> {
        if self.next_record != METAL_MTP4_RECORDS {
            return Err(EngineError::InvalidState(format!(
                "Metal MTP4 encoded {} of {METAL_MTP4_RECORDS} records",
                self.next_record
            )));
        }
        let mut final_position = self.start_position;
        for (record, (mut cursor, barrier)) in
            self.records.into_iter().zip(barrier_indices).enumerate()
        {
            cursor.commit_after_completion(barrier)?;
            let finished = cursor.finish()?;
            let expected = self.start_position + record + 1;
            if finished != expected {
                return Err(EngineError::InvalidState(format!(
                    "Metal MTP4 record {record} finished at {finished}, expected {expected}"
                )));
            }
            final_position = finished;
        }
        Ok(final_position)
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
        reads: step.reads.clone(),
        writes: step.writes.clone(),
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

    #[test]
    fn decode_cursor_commits_only_after_all_bound_steps() {
        let config = Qwen38Config::default();
        let projections = MetalProjectionPlan::qwen38(&config).unwrap();
        let schedule = MetalDecodeSchedule::qwen38(&config).unwrap();
        let bindings = MetalDecodeBindingPlan::qwen38(&schedule, &projections, &config).unwrap();

        let partial = bindings.execution_cursor(17, 17, 128 * 1024).unwrap();
        assert_eq!(partial.token_position(), 17);
        assert!(partial.finish().is_err());

        let mut cursor = bindings.execution_cursor(17, 17, 128 * 1024).unwrap();
        let first = cursor.next_step().unwrap().clone();
        assert!(cursor
            .advance(first.schedule_index + 1, first.layer, first.operation)
            .is_err());
        assert_eq!(cursor.next_step().unwrap(), &first);
        while let Some(step) = cursor.next_step().cloned() {
            if step.operation == MetalDecodeOperation::TokenCommandBufferCommit {
                assert!(cursor
                    .advance(step.schedule_index, step.layer, step.operation)
                    .is_err());
                cursor.commit_after_completion(step.schedule_index).unwrap();
            } else {
                cursor
                    .advance(step.schedule_index, step.layer, step.operation)
                    .unwrap();
            }
        }
        assert_eq!(cursor.finish().unwrap(), 18);
    }

    #[test]
    fn decode_cursor_rejects_uncommitted_or_out_of_context_positions() {
        let config = Qwen38Config::default();
        let projections = MetalProjectionPlan::qwen38(&config).unwrap();
        let schedule = MetalDecodeSchedule::qwen38(&config).unwrap();
        let bindings = MetalDecodeBindingPlan::qwen38(&schedule, &projections, &config).unwrap();
        assert!(bindings.execution_cursor(9, 8, 128).is_err());
        assert!(bindings.execution_cursor(128, 128, 128).is_err());
    }

    #[test]
    fn mtp4_cursor_publishes_only_after_all_records_share_the_final_barrier() {
        let config = Qwen38Config::default();
        let projections = MetalProjectionPlan::qwen38(&config).unwrap();
        let schedule = MetalDecodeSchedule::qwen38(&config).unwrap();
        let bindings = MetalDecodeBindingPlan::qwen38(&schedule, &projections, &config).unwrap();

        assert!(bindings.mtp4_execution_cursor(37, 36, 128).is_err());
        assert!(bindings.mtp4_execution_cursor(37, 37, 40).is_err());

        let mut cursor = bindings.mtp4_execution_cursor(37, 37, 41).unwrap();
        assert!(cursor.record_complete_encoded_record(1).is_err());
        let mut barriers = [0_usize; METAL_MTP4_RECORDS];
        for (record, barrier) in barriers.iter_mut().enumerate() {
            *barrier = cursor.record_complete_encoded_record(record).unwrap();
            assert_eq!(*barrier, 644);
        }
        let wrong = [644, 644, 643, 644];
        let mut invalid = bindings.mtp4_execution_cursor(37, 37, 41).unwrap();
        for record in 0..METAL_MTP4_RECORDS {
            invalid.record_complete_encoded_record(record).unwrap();
        }
        assert!(invalid.commit_after_shared_completion(wrong).is_err());
        assert_eq!(cursor.commit_after_shared_completion(barriers).unwrap(), 41);
    }

    #[test]
    fn decode_workspace_aliases_only_dead_slots_and_keeps_logits_live() {
        let config = Qwen38Config::default();
        let schedule = MetalDecodeSchedule::qwen38(&config).unwrap();
        let workspace = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000).unwrap();
        assert_eq!(workspace.bindings().count(), 21);
        assert_eq!(workspace.alignment(), 256);
        assert_eq!(workspace.mtp_draft_vocabulary_rows(), 40_000);
        assert_eq!(workspace.total_bytes(), 1_173_760);
        assert_eq!(workspace.independent_bytes(), 1_633_280);
        assert_eq!(
            workspace
                .binding(MetalBufferSlot::TargetLogits)
                .unwrap()
                .bytes,
            config.vocab_size * std::mem::size_of::<f32>()
        );
        assert_eq!(
            workspace.binding(MetalBufferSlot::MtpDraft).unwrap().bytes,
            40_000 * std::mem::size_of::<f32>()
        );
        assert_ne!(
            workspace
                .binding(MetalBufferSlot::TargetLogits)
                .unwrap()
                .offset,
            workspace.binding(MetalBufferSlot::MtpDraft).unwrap().offset
        );
    }

    #[test]
    fn decode_workspace_rejects_invalid_mtp_vocabulary() {
        let config = Qwen38Config::default();
        let schedule = MetalDecodeSchedule::qwen38(&config).unwrap();
        assert!(MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 0).is_err());
        assert!(
            MetalDecodeWorkspacePlan::qwen38(&schedule, &config, config.vocab_size + 1).is_err()
        );
    }

    #[test]
    fn mtp_workspace_is_separate_compact_and_alias_safe() {
        let config = Qwen38Config::default();
        let workspace = MetalMtpWorkspacePlan::qwen38(&config).unwrap();
        assert_eq!(workspace.bindings().count(), 20);
        assert_eq!(workspace.alignment(), 256);
        assert_eq!(workspace.total_bytes(), 180_224);
        assert_eq!(workspace.independent_bytes(), 536_576);
        assert!(workspace.total_bytes() < workspace.independent_bytes() / 2);
        assert_eq!(
            workspace
                .binding(MetalMtpBufferSlot::Concatenated)
                .unwrap()
                .values,
            config.hidden_size * 2
        );
        assert_eq!(
            workspace
                .binding(MetalMtpBufferSlot::FfnGate)
                .unwrap()
                .values,
            config.intermediate_size
        );
        assert_ne!(
            workspace
                .binding(MetalMtpBufferSlot::FfnGate)
                .unwrap()
                .offset,
            workspace.binding(MetalMtpBufferSlot::FfnUp).unwrap().offset
        );
    }
}
