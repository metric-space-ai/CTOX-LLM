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

// Qwen FFN down-projection input: fuse SiLU(gate) * up, recovery s_in, and
// symmetric A8 block quantization. The two producer projections remain f32;
// no 17,408-value SwiGLU tensor is materialized or reread.
// ref: ggml/src/ggml-cuda/ssm-conv.cu:48-57
// ref: ggml/src/ggml-cuda/vecdotq.cuh:115-137
extern "C" __global__ __launch_bounds__(64, 4)
void ctox_quantize_swiglu_a8_b64_sm86(
    const float* __restrict__ gate,
    const float* __restrict__ up,
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
    const float gate_value = gate[index];
    const float silu = gate_value / (1.0f + expf(-gate_value));
    const float value = silu * up[index] * load_optional_f16(s_in, index);
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

// Prepare the exact Qwen3.8-27B single-token GatedDeltaNet inputs without a
// host-side repeat or elementwise pass. in_proj_qkv/conv emits compact
// [Q:16x128, K:16x128, V:48x128]; Q and K repeat each source head three
// times. in_proj_a/in_proj_b emit one scalar per value head. A_log and
// dt_bias are resident f32 model parameters.
//
// This is an unpromoted verifier candidate. The recurrence semantics and
// tensor traversal follow the pinned upstream GatedDeltaNet implementation;
// the model-specific repeat and transforms are frozen by decoder.rs.
// ref: ggml/src/ggml-cuda/gated_delta_net.cu:36-90
extern "C" __global__ __launch_bounds__(256, 2)
void ctox_qwen_gated_delta_prepare_f32_sm86(
    const float* __restrict__ convolved_qkv,
    const float* __restrict__ raw_a,
    const float* __restrict__ raw_b,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ log_decay,
    float* __restrict__ beta,
    unsigned key_heads,
    unsigned value_heads,
    unsigned key_dim) {
    const unsigned output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (key_heads != 16u || value_heads != 48u || key_dim != 128u) {
        return;
    }
    const unsigned qk_values = value_heads * key_dim;
    if (output_index < qk_values) {
        const unsigned output_head = output_index / key_dim;
        const unsigned column = output_index - output_head * key_dim;
        const unsigned source_head = output_head / (value_heads / key_heads);
        const unsigned source_index = source_head * key_dim + column;
        const unsigned compact_values = key_heads * key_dim;
        query[output_index] = convolved_qkv[source_index];
        key[output_index] = convolved_qkv[compact_values + source_index];
    }
    if (output_index < value_heads) {
        const float a = raw_a[output_index] + dt_bias[output_index];
        const float softplus = a > 20.0f ? a : log1pf(expf(a));
        log_decay[output_index] = -expf(a_log[output_index]) * softplus;
        beta[output_index] = 1.0f / (1.0f + expf(-raw_b[output_index]));
    }
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

// Fuse the transformer residual edge with the following Qwen RMSNorm. The
// updated residual remains available for the next sublayer while the second
// output feeds its projection directly, avoiding a standalone vector-add
// launch and a second read of the summed activation.
// ref: ggml/src/ggml-cuda/norm.cu:76-151
extern "C" __global__ __launch_bounds__(256, 2)
void ctox_qwen_residual_rms_norm_f16_sm86(
    const float* __restrict__ residual,
    const float* __restrict__ update,
    const __half* __restrict__ weight,
    float* __restrict__ residual_output,
    float* __restrict__ normalized_output,
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
        const unsigned index = row_offset + column;
        const float sum = residual[index] + update[index];
        residual_output[index] = sum;
        sum_squares = fmaf(sum, sum, sum_squares);
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
        normalized_output[index] = residual_output[index] * inverse
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

// Qwen q_proj emits one [query(256), gate(256)] pair per head. This candidate
// fuses that semantic deinterleave with per-head Qwen (1 + weight) RMSNorm and
// NeoX partial RoPE, while copying gate rows into a contiguous producer buffer.
// It remains verifier-only until the composite device graph and roofline gates
// pass; it is not part of the promoted production ABI.
// ref: ggml/src/ggml-cuda/norm.cu:48-148
// ref: ggml/src/ggml-cuda/rope.cu:116-183
extern "C" __global__ __launch_bounds__(256, 2)
void ctox_qwen_query_gate_norm_rope_f32_sm86(
    const float* __restrict__ query_gate,
    const __half* __restrict__ q_norm_weight,
    const float* __restrict__ cosine,
    const float* __restrict__ sine,
    float* __restrict__ query,
    float* __restrict__ gate,
    unsigned heads,
    unsigned head_dim,
    unsigned rotary_dim,
    float epsilon) {
    const unsigned head = blockIdx.x;
    const unsigned column = threadIdx.x;
    const unsigned lane = column & (kWarpSize - 1u);
    const unsigned warp = column / kWarpSize;
    if (head >= heads || head_dim != 256u || rotary_dim != 64u) {
        return;
    }
    const unsigned input_base = head * head_dim * 2u;
    const unsigned output_base = head * head_dim;
    const float query_value = query_gate[input_base + column];
    gate[output_base + column] = query_gate[input_base + head_dim + column];

    float sum_squares = query_value * query_value;
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
            inverse = rsqrtf(sum_squares * (1.0f / 256.0f) + epsilon);
        }
    }
    __syncthreads();

    if (column < rotary_dim / 2u) {
        const unsigned right_column = column + rotary_dim / 2u;
        const float left = query_value * inverse
            * (1.0f + __half2float(q_norm_weight[column]));
        const float right = query_gate[input_base + right_column] * inverse
            * (1.0f + __half2float(q_norm_weight[right_column]));
        query[output_base + column] =
            left * cosine[column] - right * sine[column];
        query[output_base + right_column] =
            right * cosine[column] + left * sine[column];
    } else if (column >= rotary_dim) {
        query[output_base + column] = query_value * inverse
            * (1.0f + __half2float(q_norm_weight[column]));
    }
}

struct CtoxPagedKvDescriptor {
    unsigned precision;
    unsigned physical_slot;
    unsigned tokens;
    unsigned first_token;
};

struct CtoxPagedGqaParams {
    unsigned query_heads;
    unsigned key_value_heads;
    unsigned head_dim;
    unsigned tokens;
    unsigned page_tokens;
    unsigned page_count;
    unsigned combined_values;
    unsigned q2_token_bytes;
    unsigned q4_token_bytes;
    unsigned q2_page_bytes;
    unsigned q4_page_bytes;
    float scale;
};

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (unsigned offset = kWarpSize / 2; offset != 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

// Append one f32 K/V token directly into a canonical Q4_B64 page. One block
// owns one 64-value quantization block; the packed page is never materialized
// in host memory.
extern "C" __global__ __launch_bounds__(64, 8)
void ctox_pack_paged_kv_q4_f32_sm86(
    unsigned char* __restrict__ q4_pages,
    const float* __restrict__ key,
    const float* __restrict__ value,
    unsigned physical_slot,
    unsigned token_in_page,
    unsigned component_values,
    unsigned q4_token_bytes,
    unsigned q4_page_bytes) {
    const unsigned combined_index = blockIdx.x * kBlockLen + threadIdx.x;
    const float item = combined_index < component_values
        ? key[combined_index]
        : value[combined_index - component_values];
    float maximum = warp_max(fabsf(item));
    __shared__ float warp_maxima[2];
    __shared__ float block_maximum;
    __shared__ unsigned char codes[kBlockLen];
    const unsigned lane = threadIdx.x & (kWarpSize - 1u);
    const unsigned warp = threadIdx.x / kWarpSize;
    if (lane == 0u) {
        warp_maxima[warp] = maximum;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        block_maximum = fmaxf(warp_maxima[0], warp_maxima[1]);
    }
    __syncthreads();
    unsigned code = 0u;
    if (block_maximum != 0.0f) {
        // Explicit RN intrinsics keep canonical KV codes independent of the
        // module's fast-math setting and match Rust's stepwise f32 formula.
        const float normalized = fminf(
            1.0f, fmaxf(-1.0f, __fdiv_rn(item, block_maximum)));
        const float selector = __fadd_rn(__fmul_rn(normalized, 7.5f), 7.5f);
        code = static_cast<unsigned>(
            fminf(15.0f, fmaxf(0.0f, roundf(selector))));
    }
    codes[threadIdx.x] = static_cast<unsigned char>(code);
    __syncthreads();
    unsigned char* packed = q4_pages
        + static_cast<unsigned long long>(physical_slot) * q4_page_bytes
        + static_cast<unsigned long long>(token_in_page) * q4_token_bytes
        + blockIdx.x * kQ4BlockBytes;
    if (threadIdx.x == 0u) {
        *reinterpret_cast<__half*>(packed) = __float2half_rn(block_maximum);
    }
    if (threadIdx.x < kBlockLen / 2u) {
        packed[2u + threadIdx.x] = codes[threadIdx.x * 2u]
            | static_cast<unsigned char>(codes[threadIdx.x * 2u + 1u] << 4u);
    }
}

__device__ __forceinline__ unsigned q4_code_to_q2(unsigned code) {
    return code <= 2u ? 0u : code <= 7u ? 1u : code <= 12u ? 2u : 3u;
}

// A Q4 page is demoted without an f32 intermediate. Q4 quantization always
// contains an extreme code for each non-zero block, so canonical Q4->f32->Q2
// preserves the same FP16 scale and reduces to this exact codebook mapping.
extern "C" __global__ __launch_bounds__(16, 16)
void ctox_demote_paged_kv_q4_to_q2_sm86(
    const unsigned char* __restrict__ q4_page,
    unsigned char* __restrict__ q2_page,
    unsigned tokens,
    unsigned blocks_per_token) {
    const unsigned page_block = blockIdx.x;
    if (page_block >= tokens * blocks_per_token) {
        return;
    }
    const unsigned token = page_block / blocks_per_token;
    const unsigned block = page_block - token * blocks_per_token;
    const unsigned char* source = q4_page
        + static_cast<unsigned long long>(token) * blocks_per_token * kQ4BlockBytes
        + block * kQ4BlockBytes;
    unsigned char* target = q2_page
        + static_cast<unsigned long long>(token) * blocks_per_token * kQ2BlockBytes
        + block * kQ2BlockBytes;
    if (threadIdx.x == 0u) {
        *reinterpret_cast<unsigned short*>(target) =
            *reinterpret_cast<const unsigned short*>(source);
    }
    const unsigned first = source[2u + threadIdx.x * 2u];
    const unsigned second = source[3u + threadIdx.x * 2u];
    target[2u + threadIdx.x] = static_cast<unsigned char>(
        q4_code_to_q2(first & 15u)
        | (q4_code_to_q2(first >> 4u) << 2u)
        | (q4_code_to_q2(second & 15u) << 4u)
        | (q4_code_to_q2(second >> 4u) << 6u));
}

__device__ __forceinline__ float decode_paged_kv(
    const unsigned char* q2_pages,
    const unsigned char* q4_pages,
    const CtoxPagedKvDescriptor& descriptor,
    unsigned token_in_page,
    unsigned value_index,
    const CtoxPagedGqaParams& params) {
    if (descriptor.precision == 0u) {
        const unsigned char* token_base = q2_pages
            + static_cast<unsigned long long>(descriptor.physical_slot)
                * params.q2_page_bytes
            + static_cast<unsigned long long>(token_in_page)
                * params.q2_token_bytes;
        const unsigned block = value_index / kBlockLen;
        const unsigned index = value_index - block * kBlockLen;
        const unsigned char* packed = token_base + block * kQ2BlockBytes;
        const unsigned code = (packed[2u + index / 4u]
            >> ((index & 3u) * 2u)) & 3u;
        return load_f16(packed)
            * (static_cast<float>(static_cast<int>(code) * 2 - 3) * (1.0f / 3.0f));
    }
    const unsigned char* token_base = q4_pages
        + static_cast<unsigned long long>(descriptor.physical_slot)
            * params.q4_page_bytes
        + static_cast<unsigned long long>(token_in_page)
            * params.q4_token_bytes;
    const unsigned block = value_index / kBlockLen;
    const unsigned index = value_index - block * kBlockLen;
    const unsigned char* packed = token_base + block * kQ4BlockBytes;
    const unsigned byte = packed[2u + index / 2u];
    const unsigned code = (byte >> ((index & 1u) * 4u)) & 15u;
    return load_f16(packed)
        * (static_cast<float>(static_cast<int>(code) * 2 - 15) * (1.0f / 15.0f));
}

// Decode-only grouped-query attention over persistent mixed Q2/Q4 pages.
// One warp owns one query head and decodes packed K/V values directly into
// registers. Milakov-Gimelshein online softmax merges max, denominator and
// value accumulation in one cache scan, following the pinned FATTN vector
// organization. Promotion still requires controlled roofline evidence and a
// device-side page pack/demotion path.
// ref: ggml/src/ggml-cuda/fattn-vec.cuh:1-611
// ref: ggml/src/ggml-cuda/fattn-common.cuh:1-1274
extern "C" __global__ __launch_bounds__(32, 16)
void ctox_paged_q2q4_gqa_decode_f32_sm86(
    const float* __restrict__ query,
    const unsigned char* __restrict__ q2_pages,
    const unsigned char* __restrict__ q4_pages,
    const CtoxPagedKvDescriptor* __restrict__ descriptors,
    float* __restrict__ output,
    const CtoxPagedGqaParams* __restrict__ params_ptr) {
    const CtoxPagedGqaParams params = *params_ptr;
    const unsigned query_head = blockIdx.x;
    const unsigned lane = threadIdx.x;
    if (query_head >= params.query_heads || params.tokens == 0u
        || params.query_heads != 24u || params.key_value_heads != 4u
        || params.head_dim != 256u) {
        return;
    }
    const unsigned query_heads_per_kv = params.query_heads / params.key_value_heads;
    const unsigned key_value_head = query_head / query_heads_per_kv;
    const unsigned query_base = query_head * params.head_dim;
    const unsigned key_base = key_value_head * params.head_dim;
    const unsigned value_base = params.combined_values / 2u + key_base;

    float query_values[8];
    float accumulated[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (unsigned slot = 0u; slot < 8u; ++slot) {
        query_values[slot] = query[query_base + lane + slot * kWarpSize];
    }
    float maximum = -3.402823466e+38f;
    float denominator = 0.0f;
    for (unsigned page = 0u; page < params.page_count; ++page) {
        const CtoxPagedKvDescriptor descriptor = descriptors[page];
        for (unsigned token = 0u; token < descriptor.tokens; ++token) {
            float partial = 0.0f;
#pragma unroll
            for (unsigned slot = 0u; slot < 8u; ++slot) {
                const unsigned dim = lane + slot * kWarpSize;
                partial = fmaf(query_values[slot],
                               decode_paged_kv(q2_pages, q4_pages, descriptor,
                                               token, key_base + dim, params),
                               partial);
            }
            const float score = __shfl_sync(0xffffffffu, warp_sum(partial), 0)
                * params.scale;
            const float next_maximum = fmaxf(maximum, score);
            const float previous_factor = expf(maximum - next_maximum);
            const float score_factor = expf(score - next_maximum);
            denominator = denominator * previous_factor + score_factor;
#pragma unroll
            for (unsigned slot = 0u; slot < 8u; ++slot) {
                const unsigned dim = lane + slot * kWarpSize;
                accumulated[slot] = fmaf(
                    score_factor,
                    decode_paged_kv(q2_pages, q4_pages, descriptor,
                                    token, value_base + dim, params),
                    accumulated[slot] * previous_factor);
            }
            maximum = next_maximum;
        }
    }
#pragma unroll
    for (unsigned slot = 0u; slot < 8u; ++slot) {
        const unsigned dim = lane + slot * kWarpSize;
        output[query_base + dim] = accumulated[slot] / denominator;
    }
}
