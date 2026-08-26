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

struct RmsNormParams {
    uint rows;
    uint columns;
    float epsilon;
    uint reserved0;
};

struct PartialRopeParams {
    uint heads;
    uint head_dim;
    uint rotary_dim;
    uint position;
    float theta;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct PagedKvDescriptor {
    uint precision;             // 0 = Q2_B64, 1 = Q4_B64
    uint physical_slot;         // arena slot for this logical page
    uint tokens;                // populated tokens in this page
    uint first_token;           // global token index of page start
};

struct PagedGqaParams {
    uint query_heads;
    uint key_value_heads;
    uint head_dim;
    uint tokens;
    uint page_tokens;
    uint page_count;
    uint combined_values;       // [all K heads, all V heads] per token
    uint q2_token_bytes;
    uint q4_token_bytes;
    uint q2_page_bytes;
    uint q4_page_bytes;
    float scale;
};

struct GatedDeltaParams {
    uint heads;
    uint key_dim;
    uint value_dim;
    float epsilon;
};

struct CausalConvParams {
    uint channels;
    uint kernel_width;
    uint reserved0;
    uint reserved1;
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

inline float decode_paged_kv(device const uchar* q2_pages,
                             device const uchar* q4_pages,
                             device const PagedKvDescriptor& descriptor,
                             uint token_in_page,
                             uint value_index,
                             constant PagedGqaParams& params) {
    if (descriptor.precision == 0u) {
        device const uchar* token_base = q2_pages
            + ulong(descriptor.physical_slot) * params.q2_page_bytes
            + ulong(token_in_page) * params.q2_token_bytes;
        uint block = value_index / Q2Q4_BLOCK_LEN;
        uint index = value_index - block * Q2Q4_BLOCK_LEN;
        device const uchar* block_base = token_base + ulong(block) * Q2_BLOCK_BYTES;
        uint packed = uint(block_base[2u + index / 4u]);
        uint code = (packed >> ((index & 3u) * 2u)) & 3u;
        return read_scale(block_base) * q2_normalized(code);
    }

    device const uchar* token_base = q4_pages
        + ulong(descriptor.physical_slot) * params.q4_page_bytes
        + ulong(token_in_page) * params.q4_token_bytes;
    uint block = value_index / Q2Q4_BLOCK_LEN;
    uint index = value_index - block * Q2Q4_BLOCK_LEN;
    device const uchar* block_base = token_base + ulong(block) * Q4_BLOCK_BYTES;
    uint packed = uint(block_base[2u + index / 2u]);
    uint code = (packed >> ((index & 1u) * 4u)) & 15u;
    return read_scale(block_base) * (float(code) - 7.5f) * (1.0f / 7.5f);
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

// Qwen RMSNorm uses normalized * (1 + weight), unlike Llama's direct weight
// convention. One 32-wide simdgroup owns one complete row, accumulates the
// variance in f32, and writes strided columns without threadgroup scratch.
kernel void qwen_rms_norm_1p_f32(
    device const float* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RmsNormParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= params.rows) {
        return;
    }
    ulong row_offset = ulong(row) * params.columns;
    float sum_squares = 0.0f;
    for (uint column = lane; column < params.columns; column += 32u) {
        float value = input[row_offset + column];
        sum_squares = fma(value, value, sum_squares);
    }
    float variance = simd_sum(sum_squares) / float(params.columns);
    float inverse = rsqrt(variance + params.epsilon);
    for (uint column = lane; column < params.columns; column += 32u) {
        float value = input[row_offset + column];
        output[row_offset + column] = value * inverse * (1.0f + float(weight[column]));
    }
}

// GatedDeltaNet output normalization uses the direct learned weight (without
// Qwen's residual-norm +1 convention) and fuses SiLU(z). One simdgroup owns
// one value head; core and gate remain f32 graph activations while the
// mmap-backed norm weight stays FP16.
kernel void qwen_rms_norm_gated_f32(
    device const float* input [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant RmsNormParams& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= params.rows) {
        return;
    }
    ulong row_offset = ulong(row) * params.columns;
    float sum_squares = 0.0f;
    for (uint column = lane; column < params.columns; column += 32u) {
        float value = input[row_offset + column];
        sum_squares = fma(value, value, sum_squares);
    }
    float variance = simd_sum(sum_squares) / float(params.columns);
    float inverse = rsqrt(variance + params.epsilon);
    for (uint column = lane; column < params.columns; column += 32u) {
        uint index = uint(row_offset) + column;
        float gate_value = gate[index];
        float silu_gate = gate_value / (1.0f + exp(-gate_value));
        output[index] = input[index] * inverse * float(weight[column]) * silu_gate;
    }
}

// Qwen uses non-interleaved partial RoPE: the first rotary_dim/2 values are
// paired with the following rotary_dim/2 values. Dimensions at and above
// rotary_dim remain byte-identical in this in-place kernel.
kernel void qwen_partial_rope_f32(
    device float* values [[buffer(0)]],
    device const float* cosine [[buffer(1)]],
    device const float* sine [[buffer(2)]],
    constant PartialRopeParams& params [[buffer(3)]],
    uint pair_index [[thread_position_in_grid]]) {
    uint half_dim = params.rotary_dim / 2u;
    uint pair_count = params.heads * half_dim;
    if (pair_index >= pair_count) {
        return;
    }
    uint head = pair_index / half_dim;
    uint index = pair_index - head * half_dim;
    uint base = head * params.head_dim;
    float left = values[base + index];
    float right = values[base + index + half_dim];
    values[base + index] = left * cosine[index] - right * sine[index];
    values[base + index + half_dim] = right * cosine[index] + left * sine[index];
}

// Decode-only grouped-query attention over a persistent mixed Q2/Q4 paged
// cache. One 32-wide simdgroup owns one query head. The cache is never
// expanded to f32: each key/value is decoded from its packed block directly
// into registers for the score and value reductions.
kernel void qwen_paged_q2q4_gqa_decode_f32(
    device const float* query [[buffer(0)]],
    device const uchar* q2_pages [[buffer(1)]],
    device const uchar* q4_pages [[buffer(2)]],
    device const PagedKvDescriptor* descriptors [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant PagedGqaParams& params [[buffer(5)]],
    uint query_head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (query_head >= params.query_heads || params.tokens == 0u) {
        return;
    }

    uint query_heads_per_kv = params.query_heads / params.key_value_heads;
    uint key_value_head = query_head / query_heads_per_kv;
    uint query_base = query_head * params.head_dim;
    uint key_base = key_value_head * params.head_dim;
    uint value_base = params.combined_values / 2u + key_base;

    float maximum = -3.402823466e+38f;
    for (uint page = 0u; page < params.page_count; ++page) {
        device const PagedKvDescriptor& descriptor = descriptors[page];
        for (uint token = 0u; token < descriptor.tokens; ++token) {
            float partial = 0.0f;
            for (uint dim = lane; dim < params.head_dim; dim += 32u) {
                partial += query[query_base + dim]
                    * decode_paged_kv(q2_pages, q4_pages, descriptor, token,
                                      key_base + dim, params);
            }
            maximum = max(maximum, simd_sum(partial) * params.scale);
        }
    }

    float denominator = 0.0f;
    for (uint page = 0u; page < params.page_count; ++page) {
        device const PagedKvDescriptor& descriptor = descriptors[page];
        for (uint token = 0u; token < descriptor.tokens; ++token) {
            float partial = 0.0f;
            for (uint dim = lane; dim < params.head_dim; dim += 32u) {
                partial += query[query_base + dim]
                    * decode_paged_kv(q2_pages, q4_pages, descriptor, token,
                                      key_base + dim, params);
            }
            denominator += exp(simd_sum(partial) * params.scale - maximum);
        }
    }

    float accumulated[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint page = 0u; page < params.page_count; ++page) {
        device const PagedKvDescriptor& descriptor = descriptors[page];
        for (uint token = 0u; token < descriptor.tokens; ++token) {
            float partial = 0.0f;
            for (uint dim = lane; dim < params.head_dim; dim += 32u) {
                partial += query[query_base + dim]
                    * decode_paged_kv(q2_pages, q4_pages, descriptor, token,
                                      key_base + dim, params);
            }
            float probability = exp(simd_sum(partial) * params.scale - maximum)
                / denominator;
            for (uint slot = 0u; slot < params.head_dim / 32u; ++slot) {
                uint dim = lane + slot * 32u;
                accumulated[slot] += probability
                    * decode_paged_kv(q2_pages, q4_pages, descriptor, token,
                                      value_base + dim, params);
            }
        }
    }

    for (uint slot = 0u; slot < params.head_dim / 32u; ++slot) {
        uint dim = lane + slot * 32u;
        output[query_base + dim] = accumulated[slot];
    }
}

// Single-token Qwen GatedDeltaNet recurrence with persistent FP16 state.
// One threadgroup owns one value head; one thread owns one value column and
// walks the key dimension. All arithmetic is f32, while decay and update
// writes round immediately to half exactly as the pinned Rust oracle does.
kernel void qwen_gated_delta_recurrent_f16(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* log_decay [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device half* state [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant GatedDeltaParams& params [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]],
    uint value_index [[thread_index_in_threadgroup]]) {
    if (head >= params.heads || value_index >= params.value_dim) {
        return;
    }

    threadgroup float q_inverse;
    threadgroup float k_inverse;
    if (value_index == 0u) {
        float q_norm = 0.0f;
        float k_norm = 0.0f;
        uint qk_base = head * params.key_dim;
        for (uint key_index = 0u; key_index < params.key_dim; ++key_index) {
            float q = query[qk_base + key_index];
            float k = key[qk_base + key_index];
            q_norm += q * q;
            k_norm += k * k;
        }
        q_inverse = rsqrt(q_norm + params.epsilon);
        k_inverse = rsqrt(k_norm + params.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint state_head_base = head * params.key_dim * params.value_dim;
    uint qk_base = head * params.key_dim;
    uint value_offset = head * params.value_dim + value_index;
    float decay = exp(log_decay[head]);
    float memory = 0.0f;
    for (uint key_index = 0u; key_index < params.key_dim; ++key_index) {
        uint state_index = state_head_base + key_index * params.value_dim + value_index;
        half decayed = half(float(state[state_index]) * decay);
        state[state_index] = decayed;
        memory += float(decayed) * key[qk_base + key_index] * k_inverse;
    }

    float delta = (value[value_offset] - memory) * beta[head];
    float result = 0.0f;
    float query_scale = rsqrt(float(params.key_dim));
    for (uint key_index = 0u; key_index < params.key_dim; ++key_index) {
        uint state_index = state_head_base + key_index * params.value_dim + value_index;
        half updated = half(float(state[state_index])
            + key[qk_base + key_index] * k_inverse * delta);
        state[state_index] = updated;
        result += float(updated) * query[qk_base + key_index]
            * q_inverse * query_scale;
    }
    output[value_offset] = result;
}

// Single-token depthwise causal convolution. The complete convolution history
// stays FP16; the new input is rounded on insertion and the FP16 mmap-backed
// weight is widened only in registers. SiLU is fused into the output write.
kernel void qwen_causal_conv_silu_f16(
    device const float* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* state [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant CausalConvParams& params [[buffer(4)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= params.channels) {
        return;
    }
    uint base = channel * params.kernel_width;
    for (uint index = 0u; index + 1u < params.kernel_width; ++index) {
        state[base + index] = state[base + index + 1u];
    }
    state[base + params.kernel_width - 1u] = half(input[channel]);
    float sum = 0.0f;
    for (uint index = 0u; index < params.kernel_width; ++index) {
        sum += float(state[base + index]) * float(weight[base + index]);
    }
    output[channel] = sum / (1.0f + exp(-sum));
}
