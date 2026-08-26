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
constexpr unsigned kRowsPerWarpA8 = 2;

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

__device__ __forceinline__ float half_warp_sum(float value) {
#pragma unroll
    for (unsigned offset = 8; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset, 16);
    }
    return value;
}

__device__ __forceinline__ int pack_signed_q2(unsigned codes) {
    // Spread the four two-bit selectors into the low two bits of four
    // selector nibbles, then use PRMT to look up {-3,-1,1,3}. This is the
    // same byte-permutation technique used by the pinned upstream CUDA
    // unpack helpers, specialized to CTOX's affine Q2 codebook.
    const unsigned selectors = (codes & 0x03u)
        | ((codes & 0x0cu) << 2u)
        | ((codes & 0x30u) << 4u)
        | ((codes & 0xc0u) << 6u);
    return static_cast<int>(__byte_perm(0x0301fffdu, 0u, selectors));
}

__device__ __forceinline__ int pack_signed_q4(unsigned codes) {
    unsigned packed = 0;
#pragma unroll
    for (unsigned index = 0; index < 4; ++index) {
        const int value = static_cast<int>((codes >> (index * 4u)) & 0xfu) * 2 - 15;
        packed |= (static_cast<unsigned>(value) & 0xffu) << (index * 8u);
    }
    return static_cast<int>(packed);
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

// Explicit A8 activation contract used by the dp4a verifier path. One block
// produces one 64-value symmetric Q8 block after applying packed s_in. This
// follows the separate upstream activation-quantization + MMVQ organization;
// it is not a backend-specific weight requantization.
// ref: ggml/src/ggml-cuda/vecdotq.cuh:115-137
// ref: ggml/src/ggml-cuda/mmq.cu:121-122
extern "C" __global__ __launch_bounds__(64, 4)
void ctox_quantize_a8_b64_sm86(const float* __restrict__ input,
                               const __half* __restrict__ s_in,
                               int8_t* __restrict__ q8_codes,
                               float* __restrict__ q8_scales,
                               unsigned columns) {
    const unsigned local = threadIdx.x;
    const unsigned block = blockIdx.x;
    const unsigned index = block * kBlockLen + local;
    if (local >= kBlockLen || index >= columns) {
        return;
    }
    const float value = input[index] * load_optional_f16(s_in, index);
    float maximum = fabsf(value);
#pragma unroll
    for (unsigned offset = 16; offset != 0; offset >>= 1) {
        maximum = fmaxf(maximum,
                        __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    __shared__ float warp_maximum[2];
    __shared__ float block_scale;
    if ((local & 31u) == 0u) {
        warp_maximum[local / 32u] = maximum;
    }
    __syncthreads();
    if (local == 0u) {
        block_scale = fmaxf(warp_maximum[0], warp_maximum[1]) * (1.0f / 127.0f);
        q8_scales[block] = block_scale;
    }
    __syncthreads();
    int code = 0;
    if (block_scale != 0.0f) {
        code = __float2int_rn(value / block_scale);
        code = max(-127, min(127, code));
    }
    q8_codes[index] = static_cast<int8_t>(code);
}

extern "C" __global__ __launch_bounds__(256, 2)
void ctox_q2_b64_a8_matvec_sm86(const unsigned char* __restrict__ weights,
                                const int8_t* __restrict__ q8_codes,
                                const float* __restrict__ q8_scales,
                                const __half* __restrict__ s_out,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                unsigned rows,
                                unsigned columns,
                                unsigned activation) {
    const unsigned lane = threadIdx.x & (kWarpSize - 1u);
    const unsigned local_lane = lane & 15u;
    const unsigned half_warp = lane / 16u;
    const unsigned warp = threadIdx.x / kWarpSize;
    const unsigned rows_per_block =
        (blockDim.x / kWarpSize) * kRowsPerWarpA8;
    const unsigned row =
        blockIdx.x * rows_per_block + warp * kRowsPerWarpA8 + half_warp;
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
        const int weights4 = pack_signed_q2(packed[2u + local_lane]);
        const int activations4 = *reinterpret_cast<const int*>(
            q8_codes + block * kBlockLen + local_lane * 4u);
        const int dot = __dp4a(weights4, activations4, 0);
        partial = fmaf(static_cast<float>(dot),
                       load_f16(packed) * q8_scales[block] * (1.0f / 3.0f),
                       partial);
    }
    const float total = half_warp_sum(partial);
    if (local_lane == 0u) {
        const float shifted = total + (bias == nullptr ? 0.0f : bias[row]);
        output[row] = apply_activation(
            shifted * load_optional_f16(s_out, row), activation);
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void ctox_q4_b64_a8_matvec_sm86(const unsigned char* __restrict__ weights,
                                const int8_t* __restrict__ q8_codes,
                                const float* __restrict__ q8_scales,
                                const __half* __restrict__ s_out,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                unsigned rows,
                                unsigned columns,
                                unsigned activation) {
    const unsigned lane = threadIdx.x & (kWarpSize - 1u);
    const unsigned local_lane = lane & 15u;
    const unsigned half_warp = lane / 16u;
    const unsigned warp = threadIdx.x / kWarpSize;
    const unsigned rows_per_block =
        (blockDim.x / kWarpSize) * kRowsPerWarpA8;
    const unsigned row =
        blockIdx.x * rows_per_block + warp * kRowsPerWarpA8 + half_warp;
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
        const unsigned codes = static_cast<unsigned>(packed[2u + local_lane * 2u])
            | (static_cast<unsigned>(packed[3u + local_lane * 2u]) << 8u);
        const int weights4 = pack_signed_q4(codes);
        const int activations4 = *reinterpret_cast<const int*>(
            q8_codes + block * kBlockLen + local_lane * 4u);
        const int dot = __dp4a(weights4, activations4, 0);
        partial = fmaf(static_cast<float>(dot),
                       load_f16(packed) * q8_scales[block] * (1.0f / 15.0f),
                       partial);
    }
    const float total = half_warp_sum(partial);
    if (local_lane == 0u) {
        const float shifted = total + (bias == nullptr ? 0.0f : bias[row]);
        output[row] = apply_activation(
            shifted * load_optional_f16(s_out, row), activation);
    }
}
