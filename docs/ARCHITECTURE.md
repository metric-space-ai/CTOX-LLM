# Architecture

The repository deliberately has no shared inference runtime crate. Every model
directory owns the complete runtime path. Documentation and offline tooling may
be copied when a new model starts, but the resulting model must build and run
without linking another model integration.

For Qwen3.8-27B, Rust owns model validation, graph scheduling, memory planning,
backend selection, sampling, and the local Responses-shaped transport.
Hardware modules own packed layouts and fused kernels:

- CPU: AVX2/AVX-512/VNNI and NEON/DotProd/I8MM profiles;
- CUDA: driver-API launches of pinned in-crate CUDA sources;
- Metal: direct MSL compilation and dispatch;
- Snapdragon: QNN HTP op packages plus Adreno Vulkan compute.

The logical checkpoint contains identical Q2/Q4 codes and recovery scales.
Offline packers reorder those values into backend-specific tile layouts. A
logical tensor digest in every target manifest proves equivalence.

The Android data path uses AHardwareBuffer/DMA-BUF imports. A loader must drop
staging mappings after accelerator import and may not keep file, CPU, and device
copies resident simultaneously.
