// Qwen3.8-27B Q2_B64/Q4_B64 fused matvec CUDA verifier candidate.
//
// Candidate status: NOT promoted and NOT a production path. This isolated
// module exists to establish same-device numerical and roofline evidence. It
// is derived from the pinned llama.cpp CUDA dequant/MMVQ organization; the
// original sources and immutable digests live under vendor/cuda/.
//
// ref: ggml/src/ggml-cuda/dequantize.cuh:25-38
// ref: ggml/src/ggml-cuda/vecdotq.cuh:7-32
// ref: ggml/src/ggml-cuda/vecdotq.cuh:115-137
// ref: ggml/src/ggml-cuda/mmq.cuh:143-229
// ref: ggml/src/ggml-cuda/mmq.cuh:3542-3615
//
// CTOX layouts deliberately differ from upstream Q*_K super-blocks:
//   Q2_B64 = fp16 scale + 16 code bytes (64 two-bit codes)
//   Q4_B64 = fp16 scale + 32 code bytes (64 four-bit codes)
// Rows and blocks are little-endian and densely packed. Recovery scales stay
// fp16 in device memory and are widened only in registers.

#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr unsigned kWarpSize = 32;
constexpr unsigned kBlockLen = 64;
constexpr unsigned kQ2BlockBytes = 18;
constexpr unsigned kQ4BlockBytes = 34;

__device__ __forceinline__ float load_f16(const unsigned char* bytes) {
    // Every CTOX block begins on a two-byte boundary: cudaMalloc is aligned
    // and both packed block strides are even.
    return __half2float(*reinterpret_cast<const __half*>(bytes));
}

__device__ __forceinline__ float load_optional_f16(const __half* values,
                                                    unsigned index) {
    return values == nullptr ? 1.0f : __half2float(values[index]);
}

__device__ __forceinline__ float warp_sum(float value) {
    // Same warp-reduction structure used by the upstream MMVQ family. All 32
    // lanes participate; inactive Q2 lanes enter with zero.
#pragma unroll
    for (unsigned offset = kWarpSize / 2; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float apply_activation(float value,
                                                   unsigned activation) {
    return activation == 1u ? value / (1.0f + __expf(-value)) : value;
}

__device__ __forceinline__ void finish_row(float partial,
                                            const __half* s_out,
                                            const float* bias,
                                            float* output,
                                            unsigned row,
                                            unsigned activation,
                                            unsigned lane) {
    const float total = warp_sum(partial);
    if (lane == 0u) {
        const float shifted = total + (bias == nullptr ? 0.0f : bias[row]);
        output[row] = apply_activation(
            shifted * load_optional_f16(s_out, row), activation);
    }
}

}  // namespace

extern "C" __global__ __launch_bounds__(256, 2)
void ctox_q2_b64_fused_matvec_sm86(const unsigned char* __restrict__ weights,
                                   const float* __restrict__ input,
                                   const __half* __restrict__ s_in,
                                   const __half* __restrict__ s_out,
                                   const float* __restrict__ bias,
                                   float* __restrict__ output,
                                   unsigned rows,
                                   unsigned columns,
                                   unsigned activation) {
    const unsigned lane = threadIdx.x & (kWarpSize - 1u);
    const unsigned warp = threadIdx.x / kWarpSize;
    const unsigned warps_per_block = blockDim.x / kWarpSize;
    const unsigned row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) {
        return;
    }

    const unsigned blocks_per_row = columns / kBlockLen;
    const unsigned long long row_stride =
        static_cast<unsigned long long>(blocks_per_row) * kQ2BlockBytes;
    const unsigned char* row_weights = weights + row * row_stride;
    float partial = 0.0f;
    for (unsigned block = 0; block < blocks_per_row; ++block) {
        const unsigned char* packed = row_weights + block * kQ2BlockBytes;
        const float weight_scale = load_f16(packed);
        // All 32 lanes participate: two adjacent lanes share one code byte,
        // and each lane consumes one 4-bit half containing two Q2 codes.
        const unsigned byte = packed[2u + lane / 2u];
        const unsigned pair = (byte >> ((lane & 1u) * 4u)) & 0xfu;
        const unsigned code0 = pair & 0x3u;
        const unsigned code1 = pair >> 2u;
        const unsigned column = block * kBlockLen + lane * 2u;
        const float signed0 = static_cast<float>(code0 * 2u) - 3.0f;
        const float signed1 = static_cast<float>(code1 * 2u) - 3.0f;
        const float scaled_weight = weight_scale * (1.0f / 3.0f);
        const float input0 =
            input[column] * load_optional_f16(s_in, column);
        const float input1 =
            input[column + 1u] * load_optional_f16(s_in, column + 1u);
        partial = fmaf(scaled_weight * signed0, input0, partial);
        partial = fmaf(scaled_weight * signed1, input1, partial);
    }
    finish_row(partial, s_out, bias, output, row, activation, lane);
}

extern "C" __global__ __launch_bounds__(256, 2)
void ctox_q4_b64_fused_matvec_sm86(const unsigned char* __restrict__ weights,
                                   const float* __restrict__ input,
                                   const __half* __restrict__ s_in,
                                   const __half* __restrict__ s_out,
                                   const float* __restrict__ bias,
                                   float* __restrict__ output,
                                   unsigned rows,
                                   unsigned columns,
                                   unsigned activation) {
    const unsigned lane = threadIdx.x & (kWarpSize - 1u);
    const unsigned warp = threadIdx.x / kWarpSize;
    const unsigned warps_per_block = blockDim.x / kWarpSize;
    const unsigned row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) {
        return;
    }

    const unsigned blocks_per_row = columns / kBlockLen;
    const unsigned long long row_stride =
        static_cast<unsigned long long>(blocks_per_row) * kQ4BlockBytes;
    const unsigned char* row_weights = weights + row * row_stride;
    float partial = 0.0f;
    for (unsigned block = 0; block < blocks_per_row; ++block) {
        const unsigned char* packed = row_weights + block * kQ4BlockBytes;
        const float weight_scale = load_f16(packed);
        const unsigned codes = packed[2u + lane];
        const unsigned column = block * kBlockLen + lane * 2u;
        const unsigned code0 = codes & 0xfu;
        const unsigned code1 = codes >> 4u;
        const float signed0 = static_cast<float>(code0 * 2u) - 15.0f;
        const float signed1 = static_cast<float>(code1 * 2u) - 15.0f;
        const float scaled_weight = weight_scale * (1.0f / 15.0f);
        const float input0 =
            input[column] * load_optional_f16(s_in, column);
        const float input1 =
            input[column + 1u] * load_optional_f16(s_in, column + 1u);
        partial = fmaf(scaled_weight * signed0, input0, partial);
        partial = fmaf(scaled_weight * signed1, input1, partial);
    }
    finish_row(partial, s_out, bias, output, row, activation, lane);
}
