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
// For Q2, lanes 0..15 each own one packed byte/four values; for Q4, all lanes
// own two values. Input/s_in values are loaded once and reused across the four
// rows. This follows the multi-row matvec organization used by upstream ggml
// Metal while preserving CTOX's distinct Q2_B64/Q4_B64 codes and fused
// recovery semantics.

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
    // Exact src/quant.rs code order: {-1, -1/3, 1/3, 1}. The four values
    // form an affine sequence, so this avoids a lane-divergent select chain.
    return fma(float(code), 2.0f / 3.0f, -1.0f);
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

inline void fused_q2_rows(device const uchar* weights,
                          device const float* input,
                          device const half* s_in,
                          device const half* s_out,
                          device const float* bias,
                          device float* output,
                          constant FusedMatVecParams& params,
                          uint first_row,
                          uint simd_lane) {
    float partial[ROWS_PER_SIMDGROUP] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (simd_lane < 16u) {
        for (uint block = 0; block < params.blocks_per_row; ++block) {
            uint column = block * Q2Q4_BLOCK_LEN + simd_lane * 4u;
            float4 x(input[column], input[column + 1u],
                     input[column + 2u], input[column + 3u]);
            if (params.has_s_in != 0u) {
                x *= float4(float(s_in[column]), float(s_in[column + 1u]),
                            float(s_in[column + 2u]), float(s_in[column + 3u]));
            }
            for (uint row_offset = 0; row_offset < ROWS_PER_SIMDGROUP; ++row_offset) {
                uint row = first_row + row_offset;
                if (row >= params.rows) {
                    continue;
                }
                device const uchar* block_base = weights
                    + ulong(row) * params.blocks_per_row * Q2_BLOCK_BYTES
                    + ulong(block) * Q2_BLOCK_BYTES;
                float scale = read_scale(block_base);
                uint packed = uint(block_base[2u + simd_lane]);
                float4 normalized(q2_normalized(packed & 0x3u),
                                  q2_normalized((packed >> 2u) & 0x3u),
                                  q2_normalized((packed >> 4u) & 0x3u),
                                  q2_normalized((packed >> 6u) & 0x3u));
                partial[row_offset] += scale * dot(normalized, x);
            }
        }
    }
    finish_rows(partial, s_out, bias, output, params, first_row, simd_lane);
}

inline void fused_q4_rows(device const uchar* weights,
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
        uint column0 = column_start + simd_lane;
        uint column1 = column0 + 32u;
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
                + ulong(row) * params.blocks_per_row * Q4_BLOCK_BYTES
                + ulong(block) * Q4_BLOCK_BYTES;
            float scale = read_scale(block_base);
            device const uchar* codes = block_base + 2;
            uint byte_index = simd_lane >> 1u;
            uint shift = (simd_lane & 0x1u) << 2u;
            uint packed0 = uint(codes[byte_index]);
            uint packed1 = uint(codes[byte_index + 16u]);
            uint code0 = (packed0 >> shift) & 0xfu;
            uint code1 = (packed1 >> shift) & 0xfu;
            float normalized0 = (float(code0) - 7.5f) * (1.0f / 7.5f);
            float normalized1 = (float(code1) - 7.5f) * (1.0f / 7.5f);
            partial[row_offset] += scale * (normalized0 * x0 + normalized1 * x1);
        }
    }
    finish_rows(partial, s_out, bias, output, params, first_row, simd_lane);
}

inline void finish_gathered_rows(thread float* partial,
                                 device const uint* row_ids,
                                 device const half* s_out,
                                 device const float* bias,
                                 device float* output,
                                 constant FusedMatVecParams& params,
                                 uint first_request,
                                 uint simd_lane) {
    for (uint row_offset = 0; row_offset < ROWS_PER_SIMDGROUP; ++row_offset) {
        uint request = first_request + row_offset;
        if (request >= params.rows) {
            continue;
        }
        uint row = row_ids[request];
        float total = simd_sum(partial[row_offset]);
        if (simd_lane == 0u) {
            total += params.has_bias != 0u ? bias[row] : 0.0f;
            total *= params.has_s_out != 0u ? float(s_out[row]) : 1.0f;
            output[request] = apply_activation(total, params.activation);
        }
    }
}

inline void gathered_q2_rows(device const uchar* weights,
                             device const float* input,
                             device const half* s_in,
                             device const half* s_out,
                             device const float* bias,
                             device const uint* row_ids,
                             device float* output,
                             constant FusedMatVecParams& params,
                             uint first_request,
                             uint simd_lane) {
    float partial[ROWS_PER_SIMDGROUP] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (simd_lane < 16u) {
        for (uint block = 0; block < params.blocks_per_row; ++block) {
            uint column = block * Q2Q4_BLOCK_LEN + simd_lane * 4u;
            float4 x(input[column], input[column + 1u],
                     input[column + 2u], input[column + 3u]);
            if (params.has_s_in != 0u) {
                x *= float4(float(s_in[column]), float(s_in[column + 1u]),
                            float(s_in[column + 2u]), float(s_in[column + 3u]));
            }
            for (uint row_offset = 0; row_offset < ROWS_PER_SIMDGROUP; ++row_offset) {
                uint request = first_request + row_offset;
                if (request >= params.rows) {
                    continue;
                }
                uint row = row_ids[request];
                device const uchar* block_base = weights
                    + ulong(row) * params.blocks_per_row * Q2_BLOCK_BYTES
                    + ulong(block) * Q2_BLOCK_BYTES;
                float scale = read_scale(block_base);
                uint packed = uint(block_base[2u + simd_lane]);
                float4 normalized(q2_normalized(packed & 0x3u),
                                  q2_normalized((packed >> 2u) & 0x3u),
                                  q2_normalized((packed >> 4u) & 0x3u),
                                  q2_normalized((packed >> 6u) & 0x3u));
                partial[row_offset] += scale * dot(normalized, x);
            }
        }
    }
    finish_gathered_rows(partial, row_ids, s_out, bias, output, params,
                         first_request, simd_lane);
}

inline void gathered_q4_rows(device const uchar* weights,
                             device const float* input,
                             device const half* s_in,
                             device const half* s_out,
                             device const float* bias,
                             device const uint* row_ids,
                             device float* output,
                             constant FusedMatVecParams& params,
                             uint first_request,
                             uint simd_lane) {
    float partial[ROWS_PER_SIMDGROUP] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < params.blocks_per_row; ++block) {
        uint column_start = block * Q2Q4_BLOCK_LEN;
        uint column0 = column_start + simd_lane;
        uint column1 = column0 + 32u;
        float x0 = input[column0];
        float x1 = input[column1];
        if (params.has_s_in != 0u) {
            x0 *= float(s_in[column0]);
            x1 *= float(s_in[column1]);
        }
        for (uint row_offset = 0; row_offset < ROWS_PER_SIMDGROUP; ++row_offset) {
            uint request = first_request + row_offset;
            if (request >= params.rows) {
                continue;
            }
            uint row = row_ids[request];
            device const uchar* block_base = weights
                + ulong(row) * params.blocks_per_row * Q4_BLOCK_BYTES
                + ulong(block) * Q4_BLOCK_BYTES;
            float scale = read_scale(block_base);
            device const uchar* codes = block_base + 2;
            uint byte_index = simd_lane >> 1u;
            uint shift = (simd_lane & 0x1u) << 2u;
            uint packed0 = uint(codes[byte_index]);
            uint packed1 = uint(codes[byte_index + 16u]);
            uint code0 = (packed0 >> shift) & 0xfu;
            uint code1 = (packed1 >> shift) & 0xfu;
            float normalized0 = (float(code0) - 7.5f) * (1.0f / 7.5f);
            float normalized1 = (float(code1) - 7.5f) * (1.0f / 7.5f);
            partial[row_offset] += scale * (normalized0 * x0 + normalized1 * x1);
        }
    }
    finish_gathered_rows(partial, row_ids, s_out, bias, output, params,
                         first_request, simd_lane);
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
    fused_q2_rows(weights, input, s_in, s_out, bias, output, params,
                  first_row, simd_lane);
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
    fused_q4_rows(weights, input, s_in, s_out, bias, output, params,
                  first_row, simd_lane);
}

kernel void q2_b64_gathered_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const half* s_in [[buffer(2)]],
    device const half* s_out [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant FusedMatVecParams& params [[buffer(6)]],
    device const uint* row_ids [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lanes_per_group [[threads_per_threadgroup]]) {
    uint simd_groups = (lanes_per_group + 31u) / 32u;
    uint first_request = (group_id * simd_groups + simd_group) * ROWS_PER_SIMDGROUP;
    if (first_request >= params.rows) {
        return;
    }
    gathered_q2_rows(weights, input, s_in, s_out, bias, row_ids, output,
                     params, first_request, simd_lane);
}

kernel void q4_b64_gathered_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const half* s_in [[buffer(2)]],
    device const half* s_out [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant FusedMatVecParams& params [[buffer(6)]],
    device const uint* row_ids [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lanes_per_group [[threads_per_threadgroup]]) {
    uint simd_groups = (lanes_per_group + 31u) / 32u;
    uint first_request = (group_id * simd_groups + simd_group) * ROWS_PER_SIMDGROUP;
    if (first_request >= params.rows) {
        return;
    }
    gathered_q4_rows(weights, input, s_in, s_out, bias, row_ids, output,
                     params, first_request, simd_lane);
}

// Decode one loader-resolved Q2 embedding row. One thread owns one packed
// byte and writes four adjacent corrected hidden values. The weight row,
// complete s_in vector, and one s_out half remain offsets into the shared
// CTOXQ mmap; only the f32 hidden vector is a separate Metal buffer.
kernel void q2_b64_recovered_row(
    device const uchar* weights [[buffer(0)]],
    device const half* s_in [[buffer(2)]],
    device const half* s_out [[buffer(3)]],
    device float* output [[buffer(5)]],
    constant FusedMatVecParams& params [[buffer(6)]],
    uint packed_index [[thread_position_in_grid]]) {
    uint column = packed_index * 4u;
    if (column >= params.columns) {
        return;
    }
    uint block = column / Q2Q4_BLOCK_LEN;
    uint byte_in_block = packed_index & 15u;
    device const uchar* block_base = weights + ulong(block) * Q2_BLOCK_BYTES;
    float scale = read_scale(block_base) * float(s_out[0]);
    uint packed = uint(block_base[2u + byte_in_block]);
    output[column] = scale * q2_normalized(packed & 0x3u) * float(s_in[column]);
    output[column + 1u] = scale * q2_normalized((packed >> 2u) & 0x3u)
        * float(s_in[column + 1u]);
    output[column + 2u] = scale * q2_normalized((packed >> 4u) & 0x3u)
        * float(s_in[column + 2u]);
    output[column + 3u] = scale * q2_normalized((packed >> 6u) & 0x3u)
        * float(s_in[column + 3u]);
}

// Decode one loader-resolved Q4 embedding row. One thread owns one packed
// byte and writes two adjacent corrected hidden values.
kernel void q4_b64_recovered_row(
    device const uchar* weights [[buffer(0)]],
    device const half* s_in [[buffer(2)]],
    device const half* s_out [[buffer(3)]],
    device float* output [[buffer(5)]],
    constant FusedMatVecParams& params [[buffer(6)]],
    uint packed_index [[thread_position_in_grid]]) {
    uint column = packed_index * 2u;
    if (column >= params.columns) {
        return;
    }
    uint block = column / Q2Q4_BLOCK_LEN;
    uint byte_in_block = packed_index & 31u;
    device const uchar* block_base = weights + ulong(block) * Q4_BLOCK_BYTES;
    float scale = read_scale(block_base) * float(s_out[0]);
    uint packed = uint(block_base[2u + byte_in_block]);
    float normalized0 = (float(packed & 0xfu) - 7.5f) * (1.0f / 7.5f);
    float normalized1 = (float((packed >> 4u) & 0xfu) - 7.5f) * (1.0f / 7.5f);
    output[column] = scale * normalized0 * float(s_in[column]);
    output[column + 1u] = scale * normalized1 * float(s_in[column + 1u]);
}
