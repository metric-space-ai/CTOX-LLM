#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Unpromoted SM86 prefill candidate derived from the pinned upstream MMA/MMQ
// organization. It preserves CTOX Q2_B64/Q4_B64 bytes and expands only one
// 16-row x 64-column weight tile per warp into shared signed-int8 fragments.
// Eight warps reuse one 8-token A8 tile and issue Ampere integer tensor-core
// MMA. This file is isolated from the verifier baseline until same-device
// numerical and roofline gates pass.
// ref: ggml/src/ggml-cuda/mma.cuh:99-274
// ref: ggml/src/ggml-cuda/mma.cuh:944-969
// ref: ggml/src/ggml-cuda/mmq.cuh:3542-3615

namespace ctox_mma {

template <int I, int J>
struct tile {
    static constexpr int ne = I * J / 32;
    int x[ne] = {0};

    static __device__ __forceinline__ int get_i(int element) {
        if constexpr (I == 16 && J == 8) {
            return ((element / 2) * 8) + (threadIdx.x % 32) / 4;
        } else {
            static_assert(I == 8 && J == 8);
            return (threadIdx.x % 32) / 4;
        }
    }

    static __device__ __forceinline__ int get_j(int element) {
        if constexpr (I == 16 && J == 8) {
            return (((threadIdx.x % 32) % 4) * 2) + (element % 2);
        } else {
            static_assert(I == 8 && J == 8);
            return element * 4 + ((threadIdx.x % 32) % 4);
        }
    }
};

static __device__ __forceinline__ void mma(
    tile<16, 8>& destination,
    const tile<16, 8>& weights,
    const tile<8, 8>& activations) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+r"(destination.x[0]), "+r"(destination.x[1]),
          "+r"(destination.x[2]), "+r"(destination.x[3])
        : "r"(weights.x[0]), "r"(weights.x[1]),
          "r"(weights.x[2]), "r"(weights.x[3]),
          "r"(activations.x[0]), "r"(activations.x[1]));
}

// Ampere's A operand register layout is produced by ldmatrix rather than the
// logical C-tile element mapping. The source is a 16 x 32 signed-int8 shared
// tile expressed as eight packed int32 columns per row.
// ref: ggml/src/ggml-cuda/mma.cuh:787-805
static __device__ __forceinline__ void load_a(
    tile<16, 8>& destination,
    const int* values,
    int stride) {
    const unsigned lane = threadIdx.x % 32u;
    const int* address = values + (lane % 16u) * stride + (lane / 16u) * 4u;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(destination.x[0]), "=r"(destination.x[1]),
          "=r"(destination.x[2]), "=r"(destination.x[3])
        : "l"(address));
}

}  // namespace ctox_mma

namespace ctox_batched_mmq {

constexpr unsigned kBlockLen = 64;
constexpr unsigned kQ2BlockBytes = 18;
constexpr unsigned kQ4BlockBytes = 34;
constexpr unsigned kWarps = 8;
constexpr unsigned kRowsPerWarp = 16;
constexpr unsigned kRowsPerCta = kWarps * kRowsPerWarp;
constexpr unsigned kTokensPerCta = 8;

static __device__ __forceinline__ float optional_f16(
    const __half* values, unsigned index) {
    return values == nullptr ? 1.0f : __half2float(values[index]);
}

static __device__ __forceinline__ float activate(float value, unsigned code) {
    return code == 0u ? value : value / (1.0f + expf(-value));
}

template <bool q4>
static __device__ __forceinline__ void batched_mmq_body(
    const unsigned char* __restrict__ packed_weights,
    const int8_t* __restrict__ q8_codes,
    const float* __restrict__ q8_scales,
    const __half* __restrict__ s_out,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned rows,
    unsigned columns,
    unsigned batch_rows,
    unsigned output_stride,
    unsigned activation,
    int8_t* __restrict__ weight_tile,
    int8_t* __restrict__ activation_tile) {
    const unsigned thread = threadIdx.x;
    const unsigned warp = thread / 32u;
    const unsigned row_base = blockIdx.x * kRowsPerCta;
    const unsigned batch_base = blockIdx.y * kTokensPerCta;
    const unsigned blocks_per_row = columns / kBlockLen;
    constexpr unsigned block_bytes = q4 ? kQ4BlockBytes : kQ2BlockBytes;
    constexpr float scale_denominator = q4 ? (1.0f / 15.0f) : (1.0f / 3.0f);
    float sums[ctox_mma::tile<16, 8>::ne] = {0.0f};

    for (unsigned block = 0; block < blocks_per_row; ++block) {
        for (unsigned index = thread;
             index < kTokensPerCta * kBlockLen;
             index += blockDim.x) {
            const unsigned local_token = index / kBlockLen;
            const unsigned column = index % kBlockLen;
            const unsigned batch_row = batch_base + local_token;
            activation_tile[index] = batch_row < batch_rows
                ? q8_codes[static_cast<unsigned long long>(batch_row) * columns
                           + block * kBlockLen + column]
                : int8_t{0};
        }
        constexpr unsigned code_bytes_per_block = q4 ? 32u : 16u;
        for (unsigned index = thread;
             index < kRowsPerCta * code_bytes_per_block;
             index += blockDim.x) {
            const unsigned local_row = index / code_bytes_per_block;
            const unsigned code_byte = index % code_bytes_per_block;
            const unsigned row = row_base + local_row;
            unsigned codes = 0u;
            if (row < rows) {
                const unsigned long long packed_offset =
                    (static_cast<unsigned long long>(row) * blocks_per_row + block)
                    * block_bytes;
                codes = packed_weights[packed_offset + 2u + code_byte];
            }
            if constexpr (q4) {
                const unsigned output = local_row * kBlockLen + code_byte * 2u;
                weight_tile[output] = static_cast<int8_t>(
                    static_cast<int>(codes & 0x0fu) * 2 - 15);
                weight_tile[output + 1u] = static_cast<int8_t>(
                    static_cast<int>((codes >> 4u) & 0x0fu) * 2 - 15);
            } else {
                const unsigned output = local_row * kBlockLen + code_byte * 4u;
#pragma unroll
                for (unsigned selector = 0; selector < 4u; ++selector) {
                    weight_tile[output + selector] = static_cast<int8_t>(
                        static_cast<int>((codes >> (selector * 2u)) & 0x03u) * 2 - 3);
                }
            }
        }
        __syncthreads();

        ctox_mma::tile<16, 8> weight_fragment;
        ctox_mma::tile<8, 8> activation_fragment;
        ctox_mma::tile<16, 8> dot_fragment;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int packed_column_offset = half * 8;
            const int* warp_values = reinterpret_cast<const int*>(
                weight_tile + warp * kRowsPerWarp * kBlockLen);
            ctox_mma::load_a(weight_fragment,
                             warp_values + packed_column_offset,
                             kBlockLen / sizeof(int));
#pragma unroll
            for (int element = 0; element < activation_fragment.ne; ++element) {
                const int token = ctox_mma::tile<8, 8>::get_i(element);
                const int packed_column = ctox_mma::tile<8, 8>::get_j(element);
                const int* token_values = reinterpret_cast<const int*>(
                    activation_tile + token * kBlockLen);
                activation_fragment.x[element] =
                    token_values[packed_column_offset + packed_column];
            }
            ctox_mma::mma(dot_fragment, weight_fragment, activation_fragment);
        }

#pragma unroll
        for (int element = 0; element < dot_fragment.ne; ++element) {
            const unsigned local_row = warp * kRowsPerWarp
                + ctox_mma::tile<16, 8>::get_i(element);
            const unsigned local_token = ctox_mma::tile<16, 8>::get_j(element);
            const unsigned row = row_base + local_row;
            const unsigned batch_row = batch_base + local_token;
            if (row < rows && batch_row < batch_rows) {
                const unsigned long long packed_offset =
                    (static_cast<unsigned long long>(row) * blocks_per_row + block)
                    * block_bytes;
                const float weight_scale = __half2float(
                    *reinterpret_cast<const __half*>(packed_weights + packed_offset));
                const float input_scale = q8_scales[
                    static_cast<unsigned long long>(batch_row) * blocks_per_row + block];
                sums[element] = fmaf(
                    static_cast<float>(dot_fragment.x[element]),
                    weight_scale * input_scale * scale_denominator,
                    sums[element]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int element = 0; element < ctox_mma::tile<16, 8>::ne; ++element) {
        const unsigned local_row = warp * kRowsPerWarp
            + ctox_mma::tile<16, 8>::get_i(element);
        const unsigned local_token = ctox_mma::tile<16, 8>::get_j(element);
        const unsigned row = row_base + local_row;
        const unsigned batch_row = batch_base + local_token;
        if (row < rows && batch_row < batch_rows) {
            const float shifted = sums[element] + (bias == nullptr ? 0.0f : bias[row]);
            output[static_cast<unsigned long long>(batch_row) * output_stride + row] =
                activate(shifted * optional_f16(s_out, row), activation);
        }
    }
}

}  // namespace ctox_batched_mmq

extern "C" __global__ __launch_bounds__(256, 2)
void ctox_q2_b64_a8_batched_mmq_sm86(
    const unsigned char* __restrict__ weights,
    const int8_t* __restrict__ q8_codes,
    const float* __restrict__ q8_scales,
    const __half* __restrict__ s_out,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned rows,
    unsigned columns,
    unsigned batch_rows,
    unsigned output_stride,
    unsigned activation) {
    __shared__ __align__(16) int8_t weight_tile[
        ctox_batched_mmq::kRowsPerCta * ctox_batched_mmq::kBlockLen];
    __shared__ __align__(16) int8_t activation_tile[
        ctox_batched_mmq::kTokensPerCta * ctox_batched_mmq::kBlockLen];
    ctox_batched_mmq::batched_mmq_body<false>(
        weights, q8_codes, q8_scales, s_out, bias, output, rows, columns,
        batch_rows, output_stride, activation, weight_tile, activation_tile);
}

extern "C" __global__ __launch_bounds__(256, 2)
void ctox_q4_b64_a8_batched_mmq_sm86(
    const unsigned char* __restrict__ weights,
    const int8_t* __restrict__ q8_codes,
    const float* __restrict__ q8_scales,
    const __half* __restrict__ s_out,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned rows,
    unsigned columns,
    unsigned batch_rows,
    unsigned output_stride,
    unsigned activation) {
    __shared__ __align__(16) int8_t weight_tile[
        ctox_batched_mmq::kRowsPerCta * ctox_batched_mmq::kBlockLen];
    __shared__ __align__(16) int8_t activation_tile[
        ctox_batched_mmq::kTokensPerCta * ctox_batched_mmq::kBlockLen];
    ctox_batched_mmq::batched_mmq_body<true>(
        weights, q8_codes, q8_scales, s_out, bias, output, rows, columns,
        batch_rows, output_stride, activation, weight_tile, activation_tile);
}
