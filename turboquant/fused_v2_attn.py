"""Fused V2 Attention kernel — Affine dequant + QJL + Online Softmax in one pass.

Replaces the 5-step pipeline:
  1. mx.quantized_matmul (key scores)
  2. fused_qjl_scores (QJL correction)
  3. score addition
  4. mx.softmax
  5. mx.quantized_matmul (value output)

with a single Metal kernel that reads key/value data once and uses online softmax
to avoid materializing score/weight matrices.

Supports 3-bit and 4-bit affine quantization (mx.quantize format).
"""

import math

import mlx.core as mx

_FUSED_V2_HEADER = """
#include <metal_simdgroup>
using namespace metal;

// Extract B-bit integer from packed uint32 stream.
// Handles word-boundary straddling for 3-bit.
inline uint extract_bits(const device uint32_t* data, uint dim, uint bits) {
    uint bit_start = dim * bits;
    uint word = bit_start >> 5;
    uint bit_offset = bit_start & 31;

    if (bit_offset + bits <= 32) {
        return (data[word] >> bit_offset) & ((1u << bits) - 1u);
    }
    // Straddles word boundary
    uint bits_lo = 32 - bit_offset;
    return ((data[word] >> bit_offset) | (data[word + 1] << bits_lo)) & ((1u << bits) - 1u);
}
"""

# 32 Simdgroups × 32 Lanes = 1024 threads per query head.
# Each lane handles 4 dimensions. Simdgroups process keys with stride 32.
_FUSED_V2_ATTN_SOURCE = """
    uint head = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_id = tid >> 5;
    uint lane_id = tid & 31;

    // Dimensions from shapes
    // q_rot: (n_q_heads, D), key_data: (n_kv_heads, T_kv, PACKED_DIM)
    // key_scales: (n_kv_heads, T_kv, N_GROUPS), sign_bits: (n_kv_heads, T_kv, D/32)
    uint n_q_heads = q_rot_shape[0];
    uint D = q_rot_shape[1];
    uint n_kv_heads = key_scales_shape[0];
    uint T_kv = key_scales_shape[1];
    uint N_GROUPS = key_scales_shape[2];
    uint PACKED_DIM = key_data_shape[2];
    uint SIGN_WORDS = sign_bits_shape[2];    // D / 32
    uint n_repeats = n_q_heads / n_kv_heads;
    uint kv_head = head / n_repeats;
    uint BITS = D * (uint)bits_arr[0] / (PACKED_DIM * 32);  // recover bits from packed dim ratio
    // Actually: PACKED_DIM = D * bits / 32, so bits = PACKED_DIM * 32 / D
    uint group_size = D / N_GROUPS;

    if (head >= n_q_heads) return;

    // Load q_rot and q_sketch into registers (4 dims per lane)
    float q[4], qs[4];
    uint q_off = head * D;
    uint d0 = lane_id * 4;
    for (uint i = 0; i < 4; i++) {
        q[i] = q_rot[q_off + d0 + i];
        qs[i] = q_sketch[q_off + d0 + i];
    }

    // Which group do our 4 dims belong to?
    uint our_group = d0 / group_size;

    // Sign bit extraction: which word and bit offsets for our 4 dims
    // dims d0..d0+3 → sign_word = d0/32, bit offsets = d0%32 .. d0%32+3
    uint sign_word = d0 >> 5;
    uint sign_bit_base = d0 & 31;

    // KV base offsets
    uint kv_data_stride = T_kv * PACKED_DIM;
    uint kv_scale_stride = T_kv * N_GROUPS;
    uint kv_sign_stride = T_kv * SIGN_WORDS;
    uint kv_norm_stride = T_kv;

    uint kd_base = kv_head * kv_data_stride;
    uint ks_base = kv_head * kv_scale_stride;
    uint kb_base = kv_head * kv_scale_stride;  // biases same layout as scales
    uint sb_base = kv_head * kv_sign_stride;
    uint rn_base = kv_head * kv_norm_stride;

    // ===== Scoring + Online Softmax + Value Accumulation =====
    float local_max = -1e10f;
    float local_sum = 0.0f;
    float local_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint bits = (uint)bits_arr[0];

    for (uint k = simd_id; k < T_kv; k += 32) {
        // --- Key MSE Score: affine dequant + dot product ---
        const device uint32_t* k_data_ptr = key_data + kd_base + k * PACKED_DIM;
        float k_scale = key_scales[ks_base + k * N_GROUPS + our_group];
        float k_bias = key_biases[kb_base + k * N_GROUPS + our_group];

        float mse_partial = 0.0f;
        for (uint i = 0; i < 4; i++) {
            uint quant_int = extract_bits(k_data_ptr, d0 + i, bits);
            float k_val = (float)quant_int * k_scale + k_bias;
            mse_partial += q[i] * k_val;
        }
        float mse_score = simd_sum(mse_partial);

        // --- QJL Score: dot(q_sketch, sign_bits) ---
        uint32_t signs = sign_bits[sb_base + k * SIGN_WORDS + sign_word];
        float qjl_partial = 0.0f;
        for (uint i = 0; i < 4; i++) {
            float sign_val = ((signs >> (sign_bit_base + i)) & 1u) ? 1.0f : -1.0f;
            qjl_partial += qs[i] * sign_val;
        }
        float qjl_score = simd_sum(qjl_partial) * qjl_scale[0] * residual_norms[rn_base + k];

        float score = mse_score + qjl_score;

        // --- Online Softmax ---
        float new_max = max(local_max, score);
        float factor = metal::fast::exp(local_max - new_max);
        float exp_s = metal::fast::exp(score - new_max);
        local_max = new_max;
        local_sum = local_sum * factor + exp_s;

        // --- Value accumulation: affine dequant in rotated space ---
        const device uint32_t* v_data_ptr = value_data + kd_base + k * PACKED_DIM;
        float v_scale = value_scales[ks_base + k * N_GROUPS + our_group];
        float v_bias = value_biases[kb_base + k * N_GROUPS + our_group];

        for (uint i = 0; i < 4; i++) {
            uint v_int = extract_bits(v_data_ptr, d0 + i, bits);
            float v_val = (float)v_int * v_scale + v_bias;
            local_acc[i] = local_acc[i] * factor + exp_s * v_val;
        }
    }

    // ===== Cross-Simdgroup Reduction =====
    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];
    threadgroup float tg_acc[32 * 128];  // D=128 max with 32 simdgroups

    if (lane_id == 0) {
        tg_max[simd_id] = local_max;
        tg_sum[simd_id] = local_sum;
    }
    for (uint i = 0; i < 4; i++)
        tg_acc[simd_id * D + d0 + i] = local_acc[i];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e10f;
    for (uint s = 0; s < 32; s++)
        global_max = max(global_max, tg_max[s]);

    float global_sum = 0.0f;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint s = 0; s < 32; s++) {
        float factor = metal::fast::exp(tg_max[s] - global_max);
        global_sum += tg_sum[s] * factor;
        for (uint i = 0; i < 4; i++)
            result[i] += tg_acc[s * D + d0 + i] * factor;
    }

    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    for (uint i = 0; i < 4; i++)
        output_rot[head * D + d0 + i] = result[i] * inv;
"""

_fused_v2_attn_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_v2_attention",
    input_names=[
        "q_rot", "q_sketch",
        "key_data", "key_scales", "key_biases",
        "sign_bits", "residual_norms", "qjl_scale",
        "value_data", "value_scales", "value_biases",
        "bits_arr",
    ],
    output_names=["output_rot"],
    source=_FUSED_V2_ATTN_SOURCE,
    header=_FUSED_V2_HEADER,
)


def fused_v2_attention(
    q_rot: mx.array,
    q_sketch: mx.array,
    key_data: mx.array,
    key_scales: mx.array,
    key_biases: mx.array,
    sign_bits: mx.array,
    residual_norms: mx.array,
    value_data: mx.array,
    value_scales: mx.array,
    value_biases: mx.array,
    bits: int,
    qjl_scale: float,
    n_q_heads: int,
    D: int,
) -> mx.array:
    """Fully fused V2 attention: affine dequant + QJL + online softmax + value output.

    Single Metal dispatch. No intermediate score/weight matrices.
    Query must be rotated and scaled BEFORE the call.
    Output is in rotated space (caller applies inverse rotation).

    Args:
        q_rot: (n_q_heads, D) float32 — scaled, rotated queries
        q_sketch: (n_q_heads, D) float32 — JL sketch of queries
        key_data: (n_kv_heads, T_kv, D*bits/32) uint32
        key_scales: (n_kv_heads, T_kv, D/group_size) float
        key_biases: (n_kv_heads, T_kv, D/group_size) float
        sign_bits: (n_kv_heads, T_kv, D/32) uint32
        residual_norms: (n_kv_heads, T_kv) float
        value_data: (n_kv_heads, T_kv, D*bits/32) uint32
        value_scales: (n_kv_heads, T_kv, D/group_size) float
        value_biases: (n_kv_heads, T_kv, D/group_size) float
        bits: quantization bits (3 or 4)
        qjl_scale: sqrt(pi/2) / D
        n_q_heads: total query heads
        D: head dimension

    Returns:
        output_rot: (n_q_heads, D) float32 — in rotated space
    """
    scale_arr = mx.array([qjl_scale], dtype=mx.float32)
    bits_arr = mx.array([bits], dtype=mx.float32)

    outputs = _fused_v2_attn_kernel(
        inputs=[
            q_rot, q_sketch,
            key_data, key_scales, key_biases,
            sign_bits, residual_norms, scale_arr,
            value_data, value_scales, value_biases,
            bits_arr,
        ],
        grid=(n_q_heads * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(n_q_heads * D,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_q_heads, D)
