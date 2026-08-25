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
// Dispatch organization: one threadgroup per output row. Threads in the group
// cover the 64-value blocks of the row cooperatively; each thread fully
// accumulates its assigned blocks with float4/ushort4 vectorized loads, then a
// simdgroup reduction followed by a threadgroup reduction produces the row
// sum. Rows and columns are bounds-checked, and trailing partial blocks are
// handled element-wise.

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

template <uint BLOCK_BYTES, bool IS_Q2>
inline float fused_row_dot(device const uchar* weights,
                           device const float* input,
                           device const half* s_in,
                           constant FusedMatVecParams& params,
                           uint row,
                           uint lane,
                           uint lanes_per_group) {
    constexpr uint CODE_BYTES = BLOCK_BYTES - 2;
    float partial = 0.0f;
    device const uchar* row_base = weights + ulong(row) * params.blocks_per_row * BLOCK_BYTES;
    for (uint block = lane; block < params.blocks_per_row; block += lanes_per_group) {
        uint column_start = block * Q2Q4_BLOCK_LEN;
        // Bounds-check columns: trailing partial blocks contribute only the
        // valid prefix; columns beyond `columns` contribute zero.
        uint valid = min(uint(Q2Q4_BLOCK_LEN), params.columns - column_start);
        device const uchar* block_base = row_base + ulong(block) * BLOCK_BYTES;
        float scale = read_scale(block_base);
        device const uchar* codes = block_base + 2;
        float block_sum = 0.0f;
        for (uint byte_index = 0; byte_index < CODE_BYTES; ++byte_index) {
            uint packed = uint(codes[byte_index]);
            constexpr uint VALUES_PER_BYTE = IS_Q2 ? 4 : 2;
            for (uint packed_lane = 0; packed_lane < VALUES_PER_BYTE; ++packed_lane) {
                uint value_index = byte_index * VALUES_PER_BYTE + packed_lane;
                if (value_index >= valid) {
                    continue;
                }
                uint bits_per_value = IS_Q2 ? 2u : 4u;
                uint mask = IS_Q2 ? 0x3u : 0xfu;
                uint code = (packed >> (packed_lane * bits_per_value)) & mask;
                float normalized = IS_Q2 ? q2_normalized(code)
                                         : ((float(code) - 7.5f) / 7.5f);
                uint column = column_start + value_index;
                float gate = params.has_s_in != 0u ? float(s_in[column]) : 1.0f;
                block_sum = fma(scale * normalized, input[column] * gate, block_sum);
            }
        }
        partial += block_sum;
    }
    // Simdgroup reduction, then threadgroup reduction across simdgroups is
    // handled by the caller via a shared scratch buffer.
    return partial;
}

inline void reduce_and_store(threadgroup float* scratch,
                             float partial,
                             uint row,
                             uint lane,
                             uint simd_lane,
                             uint simd_group,
                             uint lanes_per_group,
                             constant FusedMatVecParams& params,
                             device const half* s_out,
                             device const float* bias,
                             device float* output) {
    float simd_total = simd_sum(partial);
    if (simd_lane == 0u) {
        scratch[simd_group] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        uint group_count = (lanes_per_group + 31u) / 32u;
        float total = 0.0f;
        for (uint g = 0; g < group_count; ++g) {
            total += scratch[g];
        }
        total += params.has_bias != 0u ? bias[row] : 0.0f;
        total *= params.has_s_out != 0u ? float(s_out[row]) : 1.0f;
        output[row] = apply_activation(total, params.activation);
    }
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
    threadgroup float* scratch [[threadgroup(0)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lanes_per_group [[threads_per_threadgroup]]) {
    // Bounds-check rows: excess threadgroups exit without touching memory.
    if (group_id >= params.rows) {
        return;
    }
    float partial = fused_row_dot<Q2_BLOCK_BYTES, true>(
        weights, input, s_in, params, group_id, lane, lanes_per_group);
    reduce_and_store(scratch, partial, group_id, lane, simd_lane, simd_group,
                     lanes_per_group, params, s_out, bias, output);
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
    threadgroup float* scratch [[threadgroup(0)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lanes_per_group [[threads_per_threadgroup]]) {
    // Bounds-check rows: excess threadgroups exit without touching memory.
    if (group_id >= params.rows) {
        return;
    }
    float partial = fused_row_dot<Q4_BLOCK_BYTES, false>(
        weights, input, s_in, params, group_id, lane, lanes_per_group);
    reduce_and_store(scratch, partial, group_id, lane, simd_lane, simd_group,
                     lanes_per_group, params, s_out, bias, output);
}
