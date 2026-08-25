// Qwen3.8-27B Q2_B64/Q4_B64 fused matvec candidate kernels (MSL).
//
// Candidate status: NOT promoted. These entry points are the Metal candidate
// for the fused Q2/Q4 matvec and must pass the same-device CPU-oracle verifier
// and benchmark gates before production dispatch may use them. The kernels
// are direct MSL only, with no framework or runtime dependencies and no
// scalar fallback path.
//
// Packed block layouts (little-endian, row-major blocks):
//   Q2_B64: 18 bytes = fp16 scale (2 bytes) + 16 code bytes; 4 values per
//           byte, 2 bits each, value = scale * codebook[code] with
//           codebook = {-1, -1/3, 1/3, 1}.
//   Q4_B64: 34 bytes = fp16 scale (2 bytes) + 32 code bytes; 2 values per
//           byte, 4 bits each, value = scale * ((code - 7.5) / 7.5).
//
// Fused semantics (matching the CPU oracle in src/backend/cpu.rs):
//   y[row] = act(s_out[row] * (sum_c w[row, c] * x[c] * s_in[c] + bias[row]))
// with act in {identity, SiLU}. Recovery scales stay in their packed CTOXQ
// fp16 representation and are widened only in registers. Scales are optional;
// absent pointers imply 1.0 for s_in/s_out and 0.0 for bias.
//
// Dispatch organization: every 32-wide simdgroup produces four output rows.
// Each lane owns two values of every 64-value block, loads input/s_in once,
// and reuses those values across the four rows. This follows the multi-row
// matvec organization used by upstream ggml Metal while preserving
// CTOX's distinct Q2_B64/Q4_B64 codes and fused recovery semantics.

#include <metal_stdlib>
using namespace metal;

constant uint Q2Q4_BLOCK_LEN = 64;
constant uint Q2_BLOCK_BYTES = 18; // 2-byte fp16 scale + 16 code bytes
constant uint Q4_BLOCK_BYTES = 34; // 2-byte fp16 scale + 32 code bytes

struct FusedMatVecParams {
    uint rows;                  // output rows
    uint columns;               // inner dimension
    uint blocks_per_row;        // ceil(columns / 64)
    uint has_s_in;              // 1 when s_in buffer is bound
    uint has_s_out;             // 1 when s_out buffer is bound
    uint has_bias;              // 1 when bias buffer is bound
    uint activation;            // 0 = identity, 1 = SiLU
    uint reserved0;
};

inline float apply_activation(float value, uint activation) {
    if (activation == 1u) {
        // SiLU: x / (1 + exp(-x)), matching the Rust oracle.
        return value / (1.0f + exp(-value));
    }
    return value;
}

inline float read_scale(device const uchar* block_base) {
    ushort bits = ushort(block_base[0]) | (ushort(block_base[1]) << 8);
    return float(as_type<half>(bits));
}

inline float q2_normalized(uint code) {
    // Exact src/quant.rs code order: {-1, -1/3, 1/3, 1}.
    return code == 0u ? -1.0f
         : code == 1u ? -(1.0f / 3.0f)
         : code == 2u ?  (1.0f / 3.0f)
                      :  1.0f;
}

constant uint ROWS_PER_SIMDGROUP = 4;

inline void finish_rows(thread float* partial,
                        device const half* s_out,
                        device const float* bias,
                        device float* output,
                        constant FusedMatVecParams& params,
                        uint first_row,
                        uint simd_lane) {
    for (uint row_offset = 0; row_offset < ROWS_PER_SIMDGROUP; ++row_offset) {
        uint row = first_row + row_offset;
        if (row >= params.rows) {
            continue;
        }
        float total = simd_sum(partial[row_offset]);
        if (simd_lane == 0u) {
            total += params.has_bias != 0u ? bias[row] : 0.0f;
            total *= params.has_s_out != 0u ? float(s_out[row]) : 1.0f;
            output[row] = apply_activation(total, params.activation);
        }
    }
}

template <uint BLOCK_BYTES, bool IS_Q2>
inline void fused_rows(device const uchar* weights,
                       device const float* input,
                       device const half* s_in,
                       device const half* s_out,
                       device const float* bias,
                       device float* output,
                       constant FusedMatVecParams& params,
                       uint first_row,
                       uint simd_lane) {
    float partial[ROWS_PER_SIMDGROUP] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < params.blocks_per_row; ++block) {
        uint column_start = block * Q2Q4_BLOCK_LEN;
        uint value0 = simd_lane;
        uint value1 = simd_lane + 32u;
        uint column0 = column_start + value0;
        uint column1 = column_start + value1;
        float x0 = input[column0];
        float x1 = input[column1];
        if (params.has_s_in != 0u) {
            x0 *= float(s_in[column0]);
            x1 *= float(s_in[column1]);
        }
        for (uint row_offset = 0; row_offset < ROWS_PER_SIMDGROUP; ++row_offset) {
            uint row = first_row + row_offset;
            if (row >= params.rows) {
                continue;
            }
            device const uchar* block_base = weights
                + ulong(row) * params.blocks_per_row * BLOCK_BYTES
                + ulong(block) * BLOCK_BYTES;
            float scale = read_scale(block_base);
            device const uchar* codes = block_base + 2;
            if (IS_Q2) {
                uint packed0 = uint(codes[value0 / 4u]);
                uint packed1 = uint(codes[value1 / 4u]);
                uint code0 = (packed0 >> ((value0 % 4u) * 2u)) & 0x3u;
                uint code1 = (packed1 >> ((value1 % 4u) * 2u)) & 0x3u;
                partial[row_offset] += scale
                    * (q2_normalized(code0) * x0 + q2_normalized(code1) * x1);
            } else {
                uint packed0 = uint(codes[value0 / 2u]);
                uint packed1 = uint(codes[value1 / 2u]);
                uint code0 = (packed0 >> ((value0 % 2u) * 4u)) & 0xfu;
                uint code1 = (packed1 >> ((value1 % 2u) * 4u)) & 0xfu;
                float normalized0 = (float(code0) - 7.5f) * (1.0f / 7.5f);
                float normalized1 = (float(code1) - 7.5f) * (1.0f / 7.5f);
                partial[row_offset] += scale * (normalized0 * x0 + normalized1 * x1);
            }
        }
    }
    finish_rows(partial, s_out, bias, output, params, first_row, simd_lane);
}

// Q2_B64 fused matvec candidate entry point.
kernel void q2_b64_fused_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const half* s_in [[buffer(2)]],
    device const half* s_out [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant FusedMatVecParams& params [[buffer(6)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lanes_per_group [[threads_per_threadgroup]]) {
    uint simd_groups = (lanes_per_group + 31u) / 32u;
    uint first_row = (group_id * simd_groups + simd_group) * ROWS_PER_SIMDGROUP;
    if (first_row >= params.rows) {
        return;
    }
    fused_rows<Q2_BLOCK_BYTES, true>(weights, input, s_in, s_out, bias,
                                     output, params, first_row, simd_lane);
}

// Q4_B64 fused matvec candidate entry point.
kernel void q4_b64_fused_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const half* s_in [[buffer(2)]],
    device const half* s_out [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant FusedMatVecParams& params [[buffer(6)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lanes_per_group [[threads_per_threadgroup]]) {
    uint simd_groups = (lanes_per_group + 31u) / 32u;
    uint first_row = (group_id * simd_groups + simd_group) * ROWS_PER_SIMDGROUP;
    if (first_row >= params.rows) {
        return;
    }
    fused_rows<Q4_BLOCK_BYTES, false>(weights, input, s_in, s_out, bias,
                                      output, params, first_row, simd_lane);
}
