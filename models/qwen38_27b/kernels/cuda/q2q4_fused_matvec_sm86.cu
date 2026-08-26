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

// One packed embedding row is resolved by the artifact loader. Decode and
// both recovery corrections stay fused so the graph receives only the final
// resident activation vector. The code extraction mirrors the pinned
// upstream dequantization organization, specialized to CTOX B64 blocks.
// ref: ggml/src/ggml-cuda/dequantize.cuh:25-38
extern "C" __global__ __launch_bounds__(256, 2)
void ctox_q2_b64_recovered_row_sm86(const unsigned char* __restrict__ weights,
                                    const __half* __restrict__ s_in,
                                    float s_out,
                                    float* __restrict__ output,
                                    unsigned columns) {
    const unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) {
        return;
    }
    const unsigned block = column / kBlockLen;
    const unsigned local = column % kBlockLen;
    const unsigned char* packed = weights + block * kQ2BlockBytes;
    const unsigned code = (packed[2u + local / 4u] >> ((local % 4u) * 2u)) & 0x3u;
    const float value = static_cast<float>(static_cast<int>(code) * 2 - 3)
        * (1.0f / 3.0f) * load_f16(packed);
    output[column] = value * load_optional_f16(s_in, column) * s_out;
}

extern "C" __global__ __launch_bounds__(256, 2)
void ctox_q4_b64_recovered_row_sm86(const unsigned char* __restrict__ weights,
                                    const __half* __restrict__ s_in,
                                    float s_out,
                                    float* __restrict__ output,
                                    unsigned columns) {
    const unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) {
        return;
    }
    const unsigned block = column / kBlockLen;
    const unsigned local = column % kBlockLen;
    const unsigned char* packed = weights + block * kQ4BlockBytes;
    const unsigned byte = packed[2u + local / 2u];
    const unsigned code = (byte >> ((local % 2u) * 4u)) & 0xfu;
    const float value = static_cast<float>(static_cast<int>(code) * 2 - 15)
        * (1.0f / 15.0f) * load_f16(packed);
    output[column] = value * load_optional_f16(s_in, column) * s_out;
}

// Single-token Qwen GatedDeltaNet recurrence with persistent FP16 state.
// This verifier candidate preserves the pinned upstream organization (one
// block per value head, register-local column updates, warp reductions) while
// using CTOX's signed memory contract: persistent state is FP16 and every
// decay/update write rounds immediately. It is not part of the promoted CUDA
// module ABI until same-device oracle and roofline evidence are recorded.
//
// ref: ggml/src/ggml-cuda/gated_delta_net.cu:1-135
extern "C" __global__ __launch_bounds__(128, 4)
void ctox_gated_delta_recurrent_f16_sm86(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ log_decay,
    const float* __restrict__ beta,
    __half* __restrict__ state,
    float* __restrict__ output,
    unsigned heads,
    unsigned key_dim,
    unsigned value_dim,
    float epsilon) {
    const unsigned head = blockIdx.x;
    const unsigned value_index = threadIdx.x;
    if (head >= heads || key_dim != 128u || value_dim != 128u) {
        return;
    }

    const unsigned lane = value_index & (kWarpSize - 1u);
    const unsigned warp = value_index / kWarpSize;
    const unsigned qk_base = head * key_dim;
    const float q = query[qk_base + value_index];
    const float k = key[qk_base + value_index];
    float q_norm = warp_sum(q * q);
    float k_norm = warp_sum(k * k);

    __shared__ float q_warp_norm[4];
    __shared__ float k_warp_norm[4];
    __shared__ float q_inverse;
    __shared__ float k_inverse;
    if (lane == 0u) {
        q_warp_norm[warp] = q_norm;
        k_warp_norm[warp] = k_norm;
    }
    __syncthreads();
    if (warp == 0u) {
        q_norm = lane < 4u ? q_warp_norm[lane] : 0.0f;
        k_norm = lane < 4u ? k_warp_norm[lane] : 0.0f;
        q_norm = warp_sum(q_norm);
        k_norm = warp_sum(k_norm);
        if (lane == 0u) {
            q_inverse = rsqrtf(q_norm + epsilon);
            k_inverse = rsqrtf(k_norm + epsilon);
        }
    }
    __syncthreads();

    const unsigned state_head_base = head * key_dim * value_dim;
    const unsigned value_offset = head * value_dim + value_index;
    const float decay = expf(log_decay[head]);
    float memory = 0.0f;
#pragma unroll 1
    for (unsigned key_index = 0u; key_index < 128u; ++key_index) {
        const unsigned state_index = state_head_base
            + key_index * value_dim + value_index;
        const __half decayed = __float2half_rn(
            __half2float(state[state_index]) * decay);
        state[state_index] = decayed;
        memory = fmaf(__half2float(decayed),
                      key[qk_base + key_index] * k_inverse,
                      memory);
    }

    const float delta = (value[value_offset] - memory) * beta[head];
    const float query_scale = rsqrtf(static_cast<float>(key_dim));
    float result = 0.0f;
#pragma unroll 1
    for (unsigned key_index = 0u; key_index < 128u; ++key_index) {
        const unsigned state_index = state_head_base
            + key_index * value_dim + value_index;
        const __half updated = __float2half_rn(
            __half2float(state[state_index])
            + key[qk_base + key_index] * k_inverse * delta);
        state[state_index] = updated;
        result = fmaf(__half2float(updated),
                      query[qk_base + key_index] * q_inverse * query_scale,
                      result);
    }
    output[value_offset] = result;
}

// Qwen's width-4 depthwise causal convolution. Each channel owns an
// independent FP16 history and FP16 weight row; input/output arithmetic and
// fused SiLU stay f32. The exact frozen production geometry is validated by
// the Rust host before this verifier candidate is launched.
// ref: ggml/src/ggml-cuda/ssm-conv.cu:1-95
extern "C" __global__ __launch_bounds__(256, 2)
void ctox_causal_conv_silu_f16_sm86(
    const float* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ state,
    float* __restrict__ output,
    unsigned channels,
    unsigned kernel_width) {
    const unsigned channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels || kernel_width != 4u) {
        return;
    }
    const unsigned base = channel * kernel_width;
    state[base] = state[base + 1u];
    state[base + 1u] = state[base + 2u];
    state[base + 2u] = state[base + 3u];
    state[base + 3u] = __float2half_rn(input[channel]);
    float sum = 0.0f;
#pragma unroll
    for (unsigned index = 0u; index < 4u; ++index) {
        sum = fmaf(__half2float(state[base + index]),
                   __half2float(weight[base + index]),
                   sum);
    }
    output[channel] = sum / (1.0f + expf(-sum));
}

// GatedDeltaNet's direct-weight RMSNorm fused with SiLU(gate). One warp owns
// one 128-value head. The learned weight stays FP16 and is widened only in
// registers; this deliberately does not apply Qwen's residual-norm +1 rule.
// ref: ggml/src/ggml-cuda/norm.cu:1-148
extern "C" __global__ __launch_bounds__(256, 2)
void ctox_gated_rms_norm_f16_sm86(
    const float* __restrict__ input,
    const float* __restrict__ gate,
    const __half* __restrict__ weight,
    float* __restrict__ output,
    unsigned rows,
    unsigned columns,
    float epsilon) {
    const unsigned lane = threadIdx.x & (kWarpSize - 1u);
    const unsigned warp = threadIdx.x / kWarpSize;
    const unsigned warps_per_block = blockDim.x / kWarpSize;
    const unsigned row = blockIdx.x * warps_per_block + warp;
    if (row >= rows || columns != 128u) {
        return;
    }
    const unsigned row_offset = row * columns;
    float sum_squares = 0.0f;
#pragma unroll
    for (unsigned slot = 0u; slot < 4u; ++slot) {
        const float item = input[row_offset + lane + slot * kWarpSize];
        sum_squares = fmaf(item, item, sum_squares);
    }
    const float inverse = rsqrtf(warp_sum(sum_squares)
        * (1.0f / static_cast<float>(columns)) + epsilon);
#pragma unroll
    for (unsigned slot = 0u; slot < 4u; ++slot) {
        const unsigned column = lane + slot * kWarpSize;
        const unsigned index = row_offset + column;
        const float gate_value = gate[index];
        const float silu_gate = gate_value / (1.0f + expf(-gate_value));
        output[index] = input[index] * inverse
            * __half2float(weight[column]) * silu_gate;
    }
}

// General Qwen residual RMSNorm. One 256-thread block owns one row and
// reduces arbitrary 32-aligned widths (notably hidden_size=5120). The FP16
// learned weight is widened only in registers and uses Qwen's (1 + weight)
// convention.
// ref: ggml/src/ggml-cuda/norm.cu:48-183
extern "C" __global__ __launch_bounds__(256, 2)
void ctox_qwen_rms_norm_f16_sm86(
    const float* __restrict__ input,
    const __half* __restrict__ weight,
    float* __restrict__ output,
    unsigned rows,
    unsigned columns,
    float epsilon) {
    const unsigned row = blockIdx.x;
    const unsigned lane = threadIdx.x & (kWarpSize - 1u);
    const unsigned warp = threadIdx.x / kWarpSize;
    if (row >= rows || columns == 0u || (columns & 31u) != 0u) {
        return;
    }
    const unsigned row_offset = row * columns;
    float sum_squares = 0.0f;
    for (unsigned column = threadIdx.x; column < columns; column += blockDim.x) {
        const float item = input[row_offset + column];
        sum_squares = fmaf(item, item, sum_squares);
    }
    sum_squares = warp_sum(sum_squares);
    __shared__ float warp_sums[8];
    __shared__ float inverse;
    if (lane == 0u) {
        warp_sums[warp] = sum_squares;
    }
    __syncthreads();
    if (warp == 0u) {
        sum_squares = lane < 8u ? warp_sums[lane] : 0.0f;
        sum_squares = warp_sum(sum_squares);
        if (lane == 0u) {
            inverse = rsqrtf(sum_squares
                * (1.0f / static_cast<float>(columns)) + epsilon);
        }
    }
    __syncthreads();
    for (unsigned column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned index = row_offset + column;
        output[index] = input[index] * inverse
            * (1.0f + __half2float(weight[column]));
    }
}

// Qwen uses NeoX/non-interleaved pairing: the first rotary_dim/2 values are
// paired with the following rotary_dim/2 values. Cosine/sine values are tiny
// per-position control buffers prepared by the host; the tail remains exactly
// untouched in this in-place kernel.
// ref: ggml/src/ggml-cuda/rope.cu:116-183
extern "C" __global__ __launch_bounds__(256, 2)
void ctox_partial_rope_f32_sm86(
    float* __restrict__ values,
    const float* __restrict__ cosine,
    const float* __restrict__ sine,
    unsigned heads,
    unsigned head_dim,
    unsigned rotary_dim) {
    const unsigned pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned half_dim = rotary_dim / 2u;
    const unsigned pair_count = heads * half_dim;
    if (pair_index >= pair_count || rotary_dim == 0u
        || (rotary_dim & 1u) != 0u || rotary_dim > head_dim) {
        return;
    }
    const unsigned head = pair_index / half_dim;
    const unsigned index = pair_index - head * half_dim;
    const unsigned base = head * head_dim;
    const float left = values[base + index];
    const float right = values[base + index + half_dim];
    values[base + index] = left * cosine[index] - right * sine[index];
    values[base + index + half_dim] = right * cosine[index] + left * sine[index];
}
