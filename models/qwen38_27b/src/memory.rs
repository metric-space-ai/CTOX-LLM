use serde::{Deserialize, Serialize};

use crate::kv_cache::DEFAULT_KV_PAGE_TOKENS;
use crate::{EngineError, Qwen38Config, Result};

pub const MIB: u64 = 1024 * 1024;
pub const GIB: u64 = 1024 * MIB;
pub const FOLD_WEIGHT_LIMIT_BYTES: u64 = 8_375_186_227; // 7.8 GiB
pub const FOLD_TARGET_BYTES: u64 = 10_415_295_693; // 9.7 GiB
pub const FOLD_HARD_LIMIT_BYTES: u64 = 10 * GIB;
pub const FOLD_RUNTIME_BUDGET_BYTES: u64 = 550 * MIB;
const KV_PAGE_METADATA_BYTES: u64 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearStateDType {
    F16,
    F32,
}

impl LinearStateDType {
    const fn bytes_per_value(self) -> u64 {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculativeStateStrategy {
    Disabled,
    /// Preserve one target-state checkpoint and replay the accepted prefix
    /// after a rejection. This trades compute for bounded residency.
    ReplayOnReject,
    /// Keep one target recurrent-state page per speculative position.
    AlignedPages,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RuntimeMemoryBudget {
    pub executable_code_and_rodata_bytes: u64,
    pub java_jni_ui_bytes: u64,
    pub tokenizer_sampler_graph_bytes: u64,
    pub native_heap_stacks_allocator_bytes: u64,
    pub accelerator_commands_descriptors_bytes: u64,
    pub kernel_workspaces_bytes: u64,
    pub admission_telemetry_bytes: u64,
}

impl RuntimeMemoryBudget {
    pub const fn fold_default() -> Self {
        Self {
            executable_code_and_rodata_bytes: 32 * MIB,
            java_jni_ui_bytes: 64 * MIB,
            tokenizer_sampler_graph_bytes: 64 * MIB,
            native_heap_stacks_allocator_bytes: 64 * MIB,
            accelerator_commands_descriptors_bytes: 96 * MIB,
            kernel_workspaces_bytes: 208 * MIB,
            admission_telemetry_bytes: 22 * MIB,
        }
    }

    pub const fn total_bytes(self) -> u64 {
        self.executable_code_and_rodata_bytes
            + self.java_jni_ui_bytes
            + self.tokenizer_sampler_graph_bytes
            + self.native_heap_stacks_allocator_bytes
            + self.accelerator_commands_descriptors_bytes
            + self.kernel_workspaces_bytes
            + self.admission_telemetry_bytes
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ResidentMemorySnapshot {
    /// Android process PSS, including Java/native/file-backed resident pages.
    pub process_pss_bytes: u64,
    /// Counted deliberately: the engine must not depend on compressed swap.
    pub process_swap_pss_bytes: u64,
    /// DMA/QNN/Vulkan memory not already attributed to process PSS.
    pub accelerator_unattributed_bytes: u64,
}

impl ResidentMemorySnapshot {
    pub fn accounted_bytes(self) -> Result<u64> {
        self.process_pss_bytes
            .checked_add(self.process_swap_pss_bytes)
            .and_then(|value| value.checked_add(self.accelerator_unattributed_bytes))
            .ok_or_else(|| EngineError::MemoryBudget("measured memory counter overflow".into()))
    }

    pub fn admit(self, additional_bytes: u64) -> Result<u64> {
        let projected = self
            .accounted_bytes()?
            .checked_add(additional_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("projected memory counter overflow".into()))?;
        if projected > FOLD_TARGET_BYTES {
            return Err(EngineError::MemoryBudget(format!(
                "admission projects {:.3} GiB, above 9.7 GiB operating target",
                projected as f64 / GIB as f64
            )));
        }
        Ok(projected)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FoldMemoryPlan {
    pub context_tokens: u64,
    pub weights_bytes: u64,
    pub kv_raw_q2_bytes: u64,
    pub kv_scale_bytes: u64,
    pub kv_q4_recent_and_sink_bytes: u64,
    pub mtp_kv_raw_q2_bytes: u64,
    pub mtp_kv_scale_bytes: u64,
    pub mtp_kv_q4_recent_and_sink_bytes: u64,
    pub mtp_kv_bytes: u64,
    pub kv_page_metadata_bytes: u64,
    pub kv_page_boundary_reserve_bytes: u64,
    pub kv_requantization_scratch_bytes: u64,
    pub linear_state_dtype: LinearStateDType,
    pub linear_recurrent_state_bytes: u64,
    pub linear_convolution_state_bytes: u64,
    pub linear_state_bytes: u64,
    pub speculative_draft_tokens: u32,
    pub speculative_state_strategy: SpeculativeStateStrategy,
    pub speculative_extra_linear_state_bytes: u64,
    pub runtime_budget: RuntimeMemoryBudget,
    pub runtime_bytes: u64,
    pub total_bytes: u64,
    pub target_bytes: u64,
    pub hard_limit_bytes: u64,
}

impl FoldMemoryPlan {
    pub fn for_context(
        config: &Qwen38Config,
        context_tokens: u64,
        weights_bytes: u64,
    ) -> Result<Self> {
        Self::for_execution(
            config,
            context_tokens,
            weights_bytes,
            LinearStateDType::F32,
            0,
            SpeculativeStateStrategy::Disabled,
        )
    }

    pub fn for_execution(
        config: &Qwen38Config,
        context_tokens: u64,
        weights_bytes: u64,
        linear_state_dtype: LinearStateDType,
        speculative_draft_tokens: u32,
        speculative_state_strategy: SpeculativeStateStrategy,
    ) -> Result<Self> {
        if context_tokens == 0 || context_tokens > config.max_position_embeddings as u64 {
            return Err(EngineError::MemoryBudget(format!(
                "context {context_tokens} is outside 1..={}",
                config.max_position_embeddings
            )));
        }
        if weights_bytes > FOLD_WEIGHT_LIMIT_BYTES {
            return Err(EngineError::MemoryBudget(format!(
                "text+MTP weights are {:.3} GiB, limit is 7.8 GiB",
                weights_bytes as f64 / GIB as f64
            )));
        }
        if (speculative_draft_tokens == 0)
            != (speculative_state_strategy == SpeculativeStateStrategy::Disabled)
        {
            return Err(EngineError::MemoryBudget(
                "speculative drafts and state strategy disagree".into(),
            ));
        }

        let kv_values = context_tokens
            * config.full_attention_layers() as u64
            * config.num_key_value_heads as u64
            * config.head_dim as u64
            * 2; // K and V
        let kv_raw_q2_bytes = kv_values.div_ceil(4);
        let kv_scale_bytes = kv_values.div_ceil(64) * 2;

        // 128 sink + 256 recent tokens remain Q4 instead of Q2. Count only
        // the delta because their base Q2 storage is already included above.
        let q4_tokens = context_tokens.min(384);
        let values_per_token = config.full_attention_layers() as u64
            * config.num_key_value_heads as u64
            * config.head_dim as u64
            * 2;
        let kv_q4_recent_and_sink_bytes = q4_tokens * values_per_token / 4;

        // The native MTP layer has its own full-attention state. Its cache is
        // not interchangeable with any target layer and therefore must be
        // resident whenever MTP is enabled.
        let resident_mtp_layers = if speculative_draft_tokens == 0 {
            0
        } else {
            config.mtp_num_hidden_layers as u64
        };
        let mtp_kv_values = context_tokens
            .checked_mul(resident_mtp_layers)
            .and_then(|values| values.checked_mul(config.num_key_value_heads as u64))
            .and_then(|values| values.checked_mul(config.head_dim as u64))
            .and_then(|values| values.checked_mul(2))
            .ok_or_else(|| EngineError::MemoryBudget("MTP KV values overflow".into()))?;
        let mtp_kv_raw_q2_bytes = mtp_kv_values.div_ceil(4);
        let mtp_kv_scale_bytes = mtp_kv_values.div_ceil(64) * 2;
        let mtp_values_per_token =
            resident_mtp_layers * config.num_key_value_heads as u64 * config.head_dim as u64 * 2;
        let mtp_kv_q4_recent_and_sink_bytes = q4_tokens * mtp_values_per_token / 4;
        let mtp_kv_bytes = mtp_kv_raw_q2_bytes
            .checked_add(mtp_kv_scale_bytes)
            .and_then(|bytes| bytes.checked_add(mtp_kv_q4_recent_and_sink_bytes))
            .ok_or_else(|| EngineError::MemoryBudget("MTP KV bytes overflow".into()))?;

        // Paged storage adds small but real allocations beyond packed tensor
        // payloads. Count 64 bytes of host/device metadata per 128-token page,
        // one possible Q4 boundary page, and one temporary Q2 page while an
        // old Q4 page is converted. The conversion is layer-sequential, so the
        // temporary page is counted once rather than once per layer.
        let resident_kv_layers = (config.full_attention_layers() as u64)
            .checked_add(resident_mtp_layers)
            .ok_or_else(|| EngineError::MemoryBudget("resident KV layers overflow".into()))?;
        let pages_per_layer = context_tokens.div_ceil(DEFAULT_KV_PAGE_TOKENS as u64);
        let kv_page_metadata_bytes = resident_kv_layers
            .checked_mul(pages_per_layer)
            .and_then(|pages| pages.checked_mul(KV_PAGE_METADATA_BYTES))
            .ok_or_else(|| EngineError::MemoryBudget("KV page metadata overflows".into()))?;
        let boundary_tokens = context_tokens
            .saturating_sub(q4_tokens)
            .min(DEFAULT_KV_PAGE_TOKENS.saturating_sub(1) as u64);
        let kv_page_boundary_reserve_bytes = boundary_tokens
            .checked_mul(
                (values_per_token / 4)
                    .checked_add(mtp_values_per_token / 4)
                    .ok_or_else(|| {
                        EngineError::MemoryBudget("KV boundary delta overflows".into())
                    })?,
            )
            .ok_or_else(|| EngineError::MemoryBudget("KV boundary reserve overflows".into()))?;
        let one_layer_values_per_token = (config.num_key_value_heads as u64)
            .checked_mul(config.head_dim as u64)
            .and_then(|values| values.checked_mul(2))
            .ok_or_else(|| EngineError::MemoryBudget("one-layer KV width overflows".into()))?;
        let one_layer_page_values = one_layer_values_per_token
            .checked_mul(DEFAULT_KV_PAGE_TOKENS as u64)
            .ok_or_else(|| EngineError::MemoryBudget("one-layer KV page overflows".into()))?;
        let kv_requantization_scratch_bytes = if resident_kv_layers == 0 {
            0
        } else {
            one_layer_page_values
                .div_ceil(4)
                .checked_add(
                    one_layer_page_values
                        .div_ceil(64)
                        .checked_mul(2)
                        .ok_or_else(|| {
                            EngineError::MemoryBudget("KV page scale bytes overflow".into())
                        })?,
                )
                .ok_or_else(|| {
                    EngineError::MemoryBudget("KV requantization scratch overflows".into())
                })?
        };

        let linear_layers = config.linear_attention_layers() as u64;
        let linear_recurrent_state_values = linear_layers
            .checked_mul(config.linear_num_value_heads as u64)
            .and_then(|values| values.checked_mul(config.linear_key_head_dim as u64))
            .and_then(|values| values.checked_mul(config.linear_value_head_dim as u64))
            .ok_or_else(|| EngineError::MemoryBudget("linear recurrent state overflows".into()))?;
        let linear_recurrent_state_bytes = linear_recurrent_state_values
            .checked_mul(linear_state_dtype.bytes_per_value())
            .ok_or_else(|| EngineError::MemoryBudget("linear recurrent bytes overflow".into()))?;
        let linear_convolution_channels = (config.linear_num_key_heads as u64)
            .checked_mul(config.linear_key_head_dim as u64)
            .and_then(|key_width| key_width.checked_mul(2))
            .and_then(|two_keys| {
                (config.linear_num_value_heads as u64)
                    .checked_mul(config.linear_value_head_dim as u64)
                    .and_then(|value_width| two_keys.checked_add(value_width))
            })
            .ok_or_else(|| {
                EngineError::MemoryBudget("linear convolution width overflows".into())
            })?;
        let linear_convolution_state_bytes = linear_layers
            .checked_mul(linear_convolution_channels)
            .and_then(|values| values.checked_mul(config.linear_conv_kernel_dim as u64))
            .and_then(|values| values.checked_mul(linear_state_dtype.bytes_per_value()))
            .ok_or_else(|| {
                EngineError::MemoryBudget("linear convolution state overflows".into())
            })?;
        let linear_state_bytes = linear_recurrent_state_bytes
            .checked_add(linear_convolution_state_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("linear state total overflows".into()))?;
        let speculative_state_copies = match speculative_state_strategy {
            SpeculativeStateStrategy::Disabled => 0,
            SpeculativeStateStrategy::ReplayOnReject => 1,
            SpeculativeStateStrategy::AlignedPages => speculative_draft_tokens as u64,
        };
        let speculative_extra_linear_state_bytes = linear_state_bytes
            .checked_mul(speculative_state_copies)
            .ok_or_else(|| EngineError::MemoryBudget("speculative state bytes overflow".into()))?;

        let runtime_budget = RuntimeMemoryBudget::fold_default();
        let runtime_bytes = runtime_budget.total_bytes();
        debug_assert_eq!(runtime_bytes, FOLD_RUNTIME_BUDGET_BYTES);
        let total_bytes = weights_bytes
            + kv_raw_q2_bytes
            + kv_scale_bytes
            + kv_q4_recent_and_sink_bytes
            + mtp_kv_bytes
            + kv_page_metadata_bytes
            + kv_page_boundary_reserve_bytes
            + kv_requantization_scratch_bytes
            + linear_state_bytes
            + speculative_extra_linear_state_bytes
            + runtime_bytes;

        Ok(Self {
            context_tokens,
            weights_bytes,
            kv_raw_q2_bytes,
            kv_scale_bytes,
            kv_q4_recent_and_sink_bytes,
            mtp_kv_raw_q2_bytes,
            mtp_kv_scale_bytes,
            mtp_kv_q4_recent_and_sink_bytes,
            mtp_kv_bytes,
            kv_page_metadata_bytes,
            kv_page_boundary_reserve_bytes,
            kv_requantization_scratch_bytes,
            linear_state_dtype,
            linear_recurrent_state_bytes,
            linear_convolution_state_bytes,
            linear_state_bytes,
            speculative_draft_tokens,
            speculative_state_strategy,
            speculative_extra_linear_state_bytes,
            runtime_budget,
            runtime_bytes,
            total_bytes,
            target_bytes: FOLD_TARGET_BYTES,
            hard_limit_bytes: FOLD_HARD_LIMIT_BYTES,
        })
    }

    pub fn verify(&self) -> Result<()> {
        if self.total_bytes > self.hard_limit_bytes {
            return Err(EngineError::MemoryBudget(format!(
                "plan is {:.3} GiB, above 10 GiB hard limit",
                self.total_bytes as f64 / GIB as f64
            )));
        }
        if self.total_bytes > self.target_bytes {
            return Err(EngineError::MemoryBudget(format!(
                "plan is {:.3} GiB, above 9.7 GiB target",
                self.total_bytes as f64 / GIB as f64
            )));
        }
        Ok(())
    }

    pub fn gib(bytes: u64) -> f64 {
        bytes as f64 / GIB as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q2_128k_plan_fits_target_at_weight_ceiling() {
        let plan =
            FoldMemoryPlan::for_context(&Qwen38Config::default(), 131_072, FOLD_WEIGHT_LIMIT_BYTES)
                .unwrap();
        assert_eq!(plan.kv_raw_q2_bytes, GIB);
        assert_eq!(plan.kv_scale_bytes, 128 * MIB);
        assert_eq!(plan.mtp_kv_bytes, 0);
        assert_eq!(plan.kv_page_metadata_bytes, MIB);
        assert_eq!(plan.kv_page_boundary_reserve_bytes, 127 * 8_192);
        assert_eq!(plan.kv_requantization_scratch_bytes, 72 * 1024);
        assert_eq!(plan.linear_recurrent_state_bytes, 144 * MIB);
        assert_eq!(plan.linear_convolution_state_bytes, 15 * MIB / 2);
        assert_eq!(plan.linear_state_bytes, 303 * MIB / 2);
        plan.verify().unwrap();
        assert!(plan.total_bytes < FOLD_TARGET_BYTES);
    }

    #[test]
    fn fold_mtp4_replay_fits_only_with_f16_state_at_weight_ceiling() {
        let config = Qwen38Config::default();
        let replay = FoldMemoryPlan::for_execution(
            &config,
            131_072,
            FOLD_WEIGHT_LIMIT_BYTES,
            LinearStateDType::F16,
            4,
            SpeculativeStateStrategy::ReplayOnReject,
        )
        .unwrap();
        assert_eq!(replay.linear_state_bytes, 303 * MIB / 4);
        assert_eq!(replay.mtp_kv_raw_q2_bytes, 64 * MIB);
        assert_eq!(replay.mtp_kv_scale_bytes, 8 * MIB);
        assert_eq!(replay.mtp_kv_q4_recent_and_sink_bytes, 192 * 1024);
        assert_eq!(replay.mtp_kv_bytes, 72 * MIB + 192 * 1024);
        assert_eq!(replay.kv_page_metadata_bytes, 1_114_112);
        assert_eq!(replay.kv_page_boundary_reserve_bytes, 1_105_408);
        assert_eq!(replay.kv_requantization_scratch_bytes, 72 * 1024);
        assert_eq!(
            replay.speculative_extra_linear_state_bytes,
            replay.linear_state_bytes
        );
        replay.verify().unwrap();

        let aligned = FoldMemoryPlan::for_execution(
            &config,
            131_072,
            FOLD_WEIGHT_LIMIT_BYTES,
            LinearStateDType::F16,
            4,
            SpeculativeStateStrategy::AlignedPages,
        )
        .unwrap();
        assert_eq!(
            aligned.speculative_extra_linear_state_bytes,
            aligned.linear_state_bytes * 4
        );
        assert!(aligned.verify().is_err());

        let f32_replay = FoldMemoryPlan::for_execution(
            &config,
            131_072,
            FOLD_WEIGHT_LIMIT_BYTES,
            LinearStateDType::F32,
            4,
            SpeculativeStateStrategy::ReplayOnReject,
        )
        .unwrap();
        assert!(f32_replay.verify().is_err());
    }

    #[test]
    fn speculative_strategy_and_draft_count_must_agree() {
        let config = Qwen38Config::default();
        assert!(FoldMemoryPlan::for_execution(
            &config,
            131_072,
            FOLD_WEIGHT_LIMIT_BYTES,
            LinearStateDType::F16,
            0,
            SpeculativeStateStrategy::ReplayOnReject,
        )
        .is_err());
        assert!(FoldMemoryPlan::for_execution(
            &config,
            131_072,
            FOLD_WEIGHT_LIMIT_BYTES,
            LinearStateDType::F16,
            4,
            SpeculativeStateStrategy::Disabled,
        )
        .is_err());
    }

    #[test]
    fn rejects_overweight_artifact() {
        assert!(FoldMemoryPlan::for_context(
            &Qwen38Config::default(),
            131_072,
            FOLD_WEIGHT_LIMIT_BYTES + 1
        )
        .is_err());
    }

    #[test]
    fn runtime_budget_includes_engine_and_accelerator_memory() {
        let runtime = RuntimeMemoryBudget::fold_default();
        assert_eq!(runtime.total_bytes(), 550 * MIB);
        assert!(runtime.executable_code_and_rodata_bytes > 0);
        assert!(runtime.kernel_workspaces_bytes > 0);
    }

    #[test]
    fn admission_counts_swap_and_external_accelerator_memory() {
        let snapshot = ResidentMemorySnapshot {
            process_pss_bytes: 9 * GIB,
            process_swap_pss_bytes: 64 * MIB,
            accelerator_unattributed_bytes: 128 * MIB,
        };
        assert!(snapshot.admit(128 * MIB).is_ok());
        assert!(snapshot.admit(GIB).is_err());
    }
}
