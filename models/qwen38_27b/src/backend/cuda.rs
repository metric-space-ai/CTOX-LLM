//! CUDA backend contract and SM86 kernel ABI baseline.
//!
//! The driver-API descriptor below pins the binary/kernel ABI that a future
//! vendored-derived CUDA module must satisfy on SM86 (Ampere, compute
//! capability 8.6) before it may leave [`PromotionState::Contract`]. Ground
//! truth for the packed tile techniques comes from the pinned llama.cpp
//! reference set under `models/qwen38_27b/vendor/cuda/` (see `UPSTREAM.json`
//! for revision and SHA-256 digests); anchors below use the form
//! `// ref: path:line-range` relative to that directory.
//!
//! Promotion stays fail-closed: no production kernel is authored here, no
//! scalar fallback exists, and execution is rejected until a same-device
//! verifier run and benchmark evidence exist per `docs/PROMOTION_GATES.md`.

use crate::backend::{Backend, BackendKind, FusedMatVec, PromotionState, RecoveredRow};
use crate::format::TensorDType;
use crate::quant::{BLOCK_LEN, Q2_BLOCK_BYTES, Q4_BLOCK_BYTES};
use crate::{EngineError, Result};

/// SM86 is Ampere consumer silicon (e.g. RTX 30-series), compute capability
/// 8.6. Upstream gates the mma path on Ampere and stream-k on Volta+, both
/// satisfied here.
// ref: ggml/src/ggml-cuda/mma.cuh:923-1022
// ref: ggml/src/ggml-cuda/mmq.cu:121-122
pub const SM86_COMPUTE_CAPABILITY: (u32, u32) = (8, 6);

/// NVIDIA warp width; all upstream dp4a/mma tiles assume 32 lanes.
// ref: ggml/src/ggml-cuda/mmq.cuh:143-144
pub const SM86_WARP_SIZE: u32 = 32;

/// Device ABI is LP64: every device pointer occupies 8 bytes in the
/// cuLaunchKernel parameter buffer.
pub const DEVICE_PTR_BYTES: u32 = 8;

/// One kernel parameter slot in the driver-API parameter buffer, in launch
/// order. `offset` is the byte offset from the start of the packed buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelParam {
    pub name: &'static str,
    pub size_bytes: u32,
    pub offset_bytes: u32,
}

/// ABI contract for one fused Q2/Q4 matvec kernel: the exact driver-API
/// symbol, the packed weight layout it consumes, and its parameter buffer.
///
/// The kernel must fuse input scale (`s_in`), output scale (`s_out`), bias,
/// and the activation in one launch; there is no unfused production path.
/// Both recovery-scale pointers address the original packed FP16 CTOXQ data;
/// a conforming module widens values in registers and must not require an f32
/// scale expansion at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaKernelAbi {
    /// Exact `CUfunction` symbol the cubin/fatbin module must export.
    pub symbol: &'static str,
    /// Packed dtype this kernel consumes. Only Q2_B64 and Q4_B64 exist;
    /// there is deliberately no Q3 descriptor.
    pub dtype: TensorDType,
    /// Values per packed block (must equal `quant::BLOCK_LEN`).
    pub block_len: usize,
    /// Bytes per packed block, matching `quant` encode/decode.
    pub block_bytes: usize,
    /// Maximum resident threads per block for SM86 occupancy checks.
    pub max_threads_per_block: u32,
    pub params: &'static [KernelParam],
}

// Parameter buffer shared by both fused matvec kernels. Order and widths are
// fixed ABI; pointers are 8 bytes, scalars 4 bytes, no padding is permitted.
// ref: ggml/src/ggml-cuda/mmq.cuh:3955-4048 (launch-time argument staging)
const FUSED_MATVEC_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "weights",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "input",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "s_in",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "s_out",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "bias",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 32,
    },
    KernelParam {
        name: "output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 40,
    },
    KernelParam {
        name: "rows",
        size_bytes: 4,
        offset_bytes: 48,
    },
    KernelParam {
        name: "columns",
        size_bytes: 4,
        offset_bytes: 52,
    },
    KernelParam {
        name: "activation",
        size_bytes: 4,
        offset_bytes: 56,
    },
];

/// Total bytes of the fused-matvec parameter buffer.
pub const FUSED_MATVEC_PARAM_BYTES: u32 = 60;

/// Q2_B64 fused matvec: 64 codes of 2 bits plus one f16 scale per block.
/// The integer accumulation stage follows the upstream dp4a pattern.
// ref: ggml/src/ggml-cuda/vecdotq.cuh:18-32
// ref: ggml/src/ggml-cuda/vecdotq.cuh:115-137
// ref: ggml/src/ggml-cuda/dequantize.cuh:25-38
pub const Q2_B64_FUSED_MATVEC: CudaKernelAbi = CudaKernelAbi {
    symbol: "ctox_q2_b64_fused_matvec_sm86",
    dtype: TensorDType::Q2B64,
    block_len: BLOCK_LEN,
    block_bytes: Q2_BLOCK_BYTES,
    max_threads_per_block: 256,
    params: FUSED_MATVEC_PARAMS,
};

/// Q4_B64 fused matvec: 64 codes of 4 bits plus one f16 scale per block.
// ref: ggml/src/ggml-cuda/vecdotq.cuh:27-32
// ref: ggml/src/ggml-cuda/vecdotq.cuh:115-137
// ref: ggml/src/ggml-cuda/dequantize.cuh:25-38
pub const Q4_B64_FUSED_MATVEC: CudaKernelAbi = CudaKernelAbi {
    symbol: "ctox_q4_b64_fused_matvec_sm86",
    dtype: TensorDType::Q4B64,
    block_len: BLOCK_LEN,
    block_bytes: Q4_BLOCK_BYTES,
    max_threads_per_block: 256,
    params: FUSED_MATVEC_PARAMS,
};

/// Module-level ABI contract for the SM86 kernel image: the compute
/// capability the cubin must target and every kernel it must export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaModuleAbi {
    pub compute_capability: (u32, u32),
    pub kernels: &'static [CudaKernelAbi],
}

/// The single supported module ABI. Additional architectures require their
/// own descriptor and their own verifier/benchmark evidence.
pub const SM86_MODULE_ABI: CudaModuleAbi = CudaModuleAbi {
    compute_capability: SM86_COMPUTE_CAPABILITY,
    kernels: &[Q2_B64_FUSED_MATVEC, Q4_B64_FUSED_MATVEC],
};

impl CudaModuleAbi {
    /// Every `CUfunction` symbol a conforming module must export.
    pub fn expected_symbols(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.kernels.iter().map(|kernel| kernel.symbol)
    }

    /// Selects the kernel descriptor for a packed dtype, or fails closed for
    /// anything that is not Q2_B64/Q4_B64.
    pub fn descriptor_for(&self, dtype: TensorDType) -> Result<&'static CudaKernelAbi> {
        self.kernels
            .iter()
            .find(|kernel| kernel.dtype == dtype)
            .ok_or(EngineError::UnsupportedOperation {
                backend: "cuda",
                operation: "fused matvec profile selection",
                reason: format!("no SM86 kernel descriptor for dtype {dtype:?}"),
            })
    }

    /// Fail-closed validation of a candidate module against this ABI.
    ///
    /// `reported_cc` is the compute capability baked into the image and
    /// `exported_symbols` is the `CUfunction` symbol table the driver
    /// resolved. Any mismatch is a hard error; there is no degraded mode.
    pub fn validate_module(
        &self,
        reported_cc: (u32, u32),
        exported_symbols: &[&str],
    ) -> Result<()> {
        if reported_cc != self.compute_capability {
            return Err(EngineError::UnsupportedOperation {
                backend: "cuda",
                operation: "module ABI validation",
                reason: format!(
                    "module targets compute capability {}.{}, required {}.{} for this profile",
                    reported_cc.0,
                    reported_cc.1,
                    self.compute_capability.0,
                    self.compute_capability.1
                ),
            });
        }
        for kernel in self.kernels {
            if kernel.block_len != BLOCK_LEN {
                return Err(EngineError::InvalidArtifact(format!(
                    "kernel {} block length {} does not match quant layout {BLOCK_LEN}",
                    kernel.symbol, kernel.block_len
                )));
            }
            let expected_block_bytes = match kernel.dtype {
                TensorDType::Q2B64 => Q2_BLOCK_BYTES,
                TensorDType::Q4B64 => Q4_BLOCK_BYTES,
                other => {
                    return Err(EngineError::UnsupportedDType(format!(
                        "kernel {} declares unsupported dtype {other:?}",
                        kernel.symbol
                    )))
                }
            };
            if kernel.block_bytes != expected_block_bytes {
                return Err(EngineError::InvalidArtifact(format!(
                    "kernel {} block size {} does not match quant layout {expected_block_bytes}",
                    kernel.symbol, kernel.block_bytes
                )));
            }
            validate_param_layout(kernel)?;
            if !exported_symbols.contains(&kernel.symbol) {
                return Err(EngineError::UnsupportedOperation {
                    backend: "cuda",
                    operation: "module ABI validation",
                    reason: format!("module does not export required symbol {}", kernel.symbol),
                });
            }
        }
        Ok(())
    }
}

/// Parameter buffers must be tightly packed and self-consistent; a sloppy
/// layout would silently corrupt launches, so it is rejected outright.
fn validate_param_layout(kernel: &CudaKernelAbi) -> Result<()> {
    let mut cursor = 0_u32;
    for param in kernel.params {
        if param.offset_bytes != cursor {
            return Err(EngineError::InvalidArtifact(format!(
                "kernel {} parameter {} has offset {}, expected {cursor}",
                kernel.symbol, param.name, param.offset_bytes
            )));
        }
        cursor += param.size_bytes;
    }
    if cursor != FUSED_MATVEC_PARAM_BYTES {
        return Err(EngineError::InvalidArtifact(format!(
            "kernel {} parameter buffer is {cursor} bytes, expected {FUSED_MATVEC_PARAM_BYTES}",
            kernel.symbol
        )));
    }
    Ok(())
}

/// CUDA contract. Kernel promotion remains fail-closed until vendored kernels,
/// immutable pins, per-op verifiers, and SM86 benchmark evidence land together.
pub struct CudaBackend;

impl CudaBackend {
    /// ABI this backend validates candidate modules against.
    pub fn module_abi(&self) -> CudaModuleAbi {
        SM86_MODULE_ABI
    }
}

impl Backend for CudaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn promotion_state(&self) -> PromotionState {
        PromotionState::Contract
    }

    fn profile(&self) -> &'static str {
        "cuda-contract"
    }

    fn fused_matvec(&self, _operation: &FusedMatVec<'_>) -> Result<Vec<f32>> {
        Err(EngineError::UnsupportedOperation {
            backend: "cuda",
            operation: "q2/q4 fused matvec",
            reason: "kernel has not passed the SM86 verifier and benchmark gates".into(),
        })
    }

    fn recovered_row(&self, _operation: &RecoveredRow<'_>) -> Result<Vec<f32>> {
        Err(EngineError::UnsupportedOperation {
            backend: "cuda",
            operation: "Q2/Q4 recovered row gather",
            reason: "embedding gather kernel has not passed the SM86 verifier and benchmark gates"
                .into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Activation, Backend};

    #[test]
    fn profile_selection_maps_q2_and_q4() {
        let abi = SM86_MODULE_ABI;
        let q2 = abi.descriptor_for(TensorDType::Q2B64).unwrap();
        let q4 = abi.descriptor_for(TensorDType::Q4B64).unwrap();
        assert_eq!(q2.symbol, "ctox_q2_b64_fused_matvec_sm86");
        assert_eq!(q4.symbol, "ctox_q4_b64_fused_matvec_sm86");
        assert_eq!(q2.block_bytes, Q2_BLOCK_BYTES);
        assert_eq!(q4.block_bytes, Q4_BLOCK_BYTES);
        assert_ne!(q2.symbol, q4.symbol);
    }

    #[test]
    fn profile_selection_rejects_non_packed_dtypes() {
        let abi = SM86_MODULE_ABI;
        assert!(abi.descriptor_for(TensorDType::F16).is_err());
        assert!(abi.descriptor_for(TensorDType::F32).is_err());
    }

    #[test]
    fn abi_metadata_targets_sm86() {
        let abi = SM86_MODULE_ABI;
        assert_eq!(abi.compute_capability, (8, 6));
        assert_eq!(SM86_WARP_SIZE, 32);
        assert_eq!(abi.kernels.len(), 2);
        let symbols: Vec<_> = abi.expected_symbols().collect();
        assert_eq!(symbols.len(), 2);
        for kernel in abi.kernels {
            assert_eq!(kernel.block_len, BLOCK_LEN);
            assert_eq!(kernel.max_threads_per_block % SM86_WARP_SIZE, 0);
            let end = kernel.params.last().unwrap();
            assert_eq!(
                end.offset_bytes + end.size_bytes,
                FUSED_MATVEC_PARAM_BYTES,
                "parameter buffer must be tightly packed"
            );
        }
    }

    #[test]
    fn validates_a_complete_module() {
        let symbols = [
            "ctox_q2_b64_fused_matvec_sm86",
            "ctox_q4_b64_fused_matvec_sm86",
        ];
        SM86_MODULE_ABI.validate_module((8, 6), &symbols).unwrap();
    }

    #[test]
    fn rejects_missing_symbols() {
        let only_q2 = ["ctox_q2_b64_fused_matvec_sm86"];
        let err = SM86_MODULE_ABI
            .validate_module((8, 6), &only_q2)
            .unwrap_err();
        assert!(err.to_string().contains("ctox_q4_b64_fused_matvec_sm86"));

        let err = SM86_MODULE_ABI.validate_module((8, 6), &[]).unwrap_err();
        assert!(err.to_string().contains("ctox_q2_b64_fused_matvec_sm86"));
    }

    #[test]
    fn rejects_wrong_compute_capability() {
        let symbols = [
            "ctox_q2_b64_fused_matvec_sm86",
            "ctox_q4_b64_fused_matvec_sm86",
        ];
        assert!(SM86_MODULE_ABI.validate_module((8, 0), &symbols).is_err());
        assert!(SM86_MODULE_ABI.validate_module((9, 0), &symbols).is_err());
        assert!(SM86_MODULE_ABI.validate_module((7, 5), &symbols).is_err());
    }

    #[test]
    fn execution_stays_fail_closed() {
        let backend = CudaBackend;
        assert_eq!(backend.promotion_state(), PromotionState::Contract);
        assert_eq!(backend.module_abi(), SM86_MODULE_ABI);
        for dtype in [TensorDType::Q2B64, TensorDType::Q4B64] {
            let weights = vec![0_u8; BLOCK_LEN.div_ceil(8).max(2) * BLOCK_LEN];
            let input = vec![0.0_f32; BLOCK_LEN];
            let operation = FusedMatVec {
                dtype,
                weights: &weights,
                segments: &[],
                rows: 1,
                columns: BLOCK_LEN,
                input: &input,
                s_in: None,
                s_out: None,
                bias: None,
                activation: Activation::Identity,
            };
            let err = backend.fused_matvec(&operation).unwrap_err();
            assert!(matches!(
                err,
                EngineError::UnsupportedOperation {
                    backend: "cuda",
                    ..
                }
            ));
        }
    }
}
