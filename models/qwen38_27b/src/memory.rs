use serde::{Deserialize, Serialize};

use crate::{EngineError, Qwen38Config, Result};

pub const MIB: u64 = 1024 * 1024;
pub const GIB: u64 = 1024 * MIB;
pub const FOLD_WEIGHT_LIMIT_BYTES: u64 = 8_375_186_227; // 7.8 GiB
pub const FOLD_TARGET_BYTES: u64 = 10_415_295_693; // 9.7 GiB
pub const FOLD_HARD_LIMIT_BYTES: u64 = 10 * GIB;
pub const FOLD_RUNTIME_BUDGET_BYTES: u64 = 550 * MIB;

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
    pub linear_state_bytes: u64,
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

        let linear_state_values = config.linear_attention_layers() as u64
            * config.linear_num_value_heads as u64
            * config.linear_key_head_dim as u64
            * config.linear_value_head_dim as u64;
        let linear_state_bytes = linear_state_values * 4;

        let runtime_budget = RuntimeMemoryBudget::fold_default();
        let runtime_bytes = runtime_budget.total_bytes();
        debug_assert_eq!(runtime_bytes, FOLD_RUNTIME_BUDGET_BYTES);
        let total_bytes = weights_bytes
            + kv_raw_q2_bytes
            + kv_scale_bytes
            + kv_q4_recent_and_sink_bytes
            + linear_state_bytes
            + runtime_bytes;

        Ok(Self {
            context_tokens,
            weights_bytes,
            kv_raw_q2_bytes,
            kv_scale_bytes,
            kv_q4_recent_and_sink_bytes,
            linear_state_bytes,
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
        assert_eq!(plan.linear_state_bytes, 144 * MIB);
        plan.verify().unwrap();
        assert!(plan.total_bytes < FOLD_TARGET_BYTES);
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
