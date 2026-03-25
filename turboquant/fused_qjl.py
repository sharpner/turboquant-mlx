"""Fused QJL score kernel — computes dot(q_sketch, sign_bits) without unpacking.

The naive path materializes (B, H, T_kv, D) float32 from (B, H, T_kv, D//32) uint32
— a 32x memory blowup. This kernel reads packed uint32 directly and computes
the dot product in-register, reducing memory traffic by ~12x.
"""

import math

import mlx.core as mx

_FUSED_QJL_HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

# Each simdgroup (32 threads) handles one (b, h, r) combination.
# Each lane processes D/32 dimensions (e.g. 4 for D=128, 8 for D=256).
# simd_sum reduces partial dot products across all 32 lanes.
_FUSED_QJL_SCORES_SOURCE = """
    // Grid: (B * n_kv_heads * n_repeats * 32, T_kv, 1)
    // Threadgroup: (32, 1, 1) — one simdgroup
    // thread_position_in_grid.y selects key

    uint head_group = threadgroup_position_in_grid.x;  // which (b, h, r)
    uint key_idx = thread_position_in_grid.y;           // which key token
    uint lane = thread_index_in_simdgroup;              // 0..31

    // Derive dimensions from shapes
    uint total_bhr = q_sketch_shape[0];           // B * n_kv_heads * n_repeats
    uint D = q_sketch_shape[1];                    // head_dim
    uint total_bh = sign_bits_shape[0];            // B * n_kv_heads
    uint T_kv = sign_bits_shape[1];
    uint WORDS = sign_bits_shape[2];               // D / 32
    uint n_repeats = total_bhr / total_bh;

    if (head_group >= total_bhr) return;
    if (key_idx >= T_kv) return;

    uint bh = head_group / n_repeats;              // which (b, h)
    uint q_off = head_group * D;
    uint s_off = (bh * T_kv + key_idx) * WORDS;

    // Each lane handles D/32 consecutive dimensions.
    // For D=128: 4 dims/lane. For D=256: 8 dims/lane.
    uint dims_per_lane = D / 32;
    uint base_dim = lane * dims_per_lane;

    float partial = 0.0f;
    for (uint d = 0; d < dims_per_lane; d++) {
        uint dim = base_dim + d;
        uint word_idx = dim >> 5;       // dim / 32
        uint bit_idx = dim & 31;        // dim % 32

        float sign_val = ((sign_bits[s_off + word_idx] >> bit_idx) & 1u) ? 1.0f : -1.0f;
        partial += q_sketch[q_off + dim] * sign_val;
    }

    // Reduce across all 32 lanes to get full dot product
    float dot = simd_sum(partial);

    // Lane 0 writes the result
    if (lane == 0) {
        float norm = residual_norms[bh * T_kv + key_idx];
        output[head_group * T_kv + key_idx] = dot * qjl_scale[0] * norm;
    }
"""

_fused_qjl_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_qjl_scores",
    input_names=["q_sketch", "sign_bits", "residual_norms", "qjl_scale"],
    output_names=["output"],
    source=_FUSED_QJL_SCORES_SOURCE,
    header=_FUSED_QJL_HEADER,
)


def fused_qjl_scores(
    q_sketch: mx.array,
    sign_bits: mx.array,
    residual_norms: mx.array,
    D: int,
    qjl_scale,
) -> mx.array:
    """Fused QJL score computation — avoids 32x sign bit blowup.

    Computes dot(q_sketch, unpack(sign_bits)) * qjl_scale * residual_norms
    directly from packed uint32, without materializing the float sign array.

    Args:
        q_sketch: (B * n_kv_heads * n_repeats, D) float32
        sign_bits: (B * n_kv_heads, T_kv, D // 32) uint32
        residual_norms: (B * n_kv_heads, T_kv) float32
        D: head dimension
        qjl_scale: float or pre-computed mx.array([scale], float32)

    Returns:
        qjl_scores: (B * n_kv_heads * n_repeats, T_kv) float32
    """
    total_bhr = q_sketch.shape[0]
    T_kv = sign_bits.shape[1]

    if T_kv == 0:
        return mx.zeros((total_bhr, T_kv))

    if isinstance(qjl_scale, (int, float)):
        scale_arr = mx.array([qjl_scale], dtype=mx.float32)
    else:
        scale_arr = qjl_scale

    outputs = _fused_qjl_kernel(
        inputs=[q_sketch, sign_bits, residual_norms, scale_arr],
        grid=(total_bhr * 32, T_kv, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(total_bhr * T_kv,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(total_bhr, T_kv)
