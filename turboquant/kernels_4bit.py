"""4-bit Metal kernels for TurboQuant.

Extends the original 2-bit kernels to support 4-bit quantization with D=256.
Includes pack/unpack utilities and a fused attention kernel.

Uses 16 simdgroups (512 threads) per query head to stay within Apple Silicon's
32 KB threadgroup memory limit at D=256.
"""

import mlx.core as mx


# ===========================================================================
# 4-bit pack/unpack utilities
# ===========================================================================

_PACK_4BIT_SOURCE = """
    uint word_idx = thread_position_in_grid.x;
    uint base = word_idx * 8;
    uint total = indices_shape[0];

    uint32_t packed = 0;
    for (uint i = 0; i < 8; i++) {
        uint idx = base + i;
        if (idx < total) {
            packed |= (static_cast<uint32_t>(indices[idx]) & 0xFu) << (i * 4);
        }
    }
    packed_indices[word_idx] = packed;
"""

_pack_4bit_kernel = mx.fast.metal_kernel(
    name="turboquant_pack_4bit",
    input_names=["indices"],
    output_names=["packed_indices"],
    source=_PACK_4BIT_SOURCE,
)


def pack_4bit_indices(indices: mx.array) -> mx.array:
    """Packs 4-bit indices (0-15) into uint32 words (8 indices per word).

    Args:
        indices: (..., D) uint8 with values 0-15

    Returns:
        packed: (..., D // 8) uint32
    """
    original_shape = indices.shape
    D = original_shape[-1]
    if D % 8 != 0:
        raise ValueError(f"Last dimension must be divisible by 8, got {D}")

    flat = indices.reshape(-1)
    num_words = flat.size // 8
    outputs = _pack_4bit_kernel(
        inputs=[flat],
        grid=(num_words, 1, 1),
        threadgroup=(min(256, num_words), 1, 1),
        output_shapes=[(num_words,)],
        output_dtypes=[mx.uint32],
    )
    out_shape = original_shape[:-1] + (D // 8,)
    return outputs[0].reshape(out_shape)


_SHIFTS_4BIT = mx.array([i * 4 for i in range(8)], dtype=mx.uint32)


def unpack_4bit_indices(packed: mx.array, D: int) -> mx.array:
    """Unpacks uint32 to 4-bit indices (0-15).

    Args:
        packed: (..., D // 8) uint32
        D: Original dimension

    Returns:
        indices: (..., D) uint32 with values 0-15
    """
    expanded = (packed[..., None] >> _SHIFTS_4BIT) & 0xF
    return expanded.reshape(*packed.shape[:-1], D)


# ===========================================================================
# Fused attention kernel — 4-bit, D=256
# ===========================================================================

_FUSED_ATTN_4BIT_HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

_FUSED_ATTN_4BIT_SOURCE = """
    // Grid: total threads = n_q_heads * 512
    // Threadgroup: (512, 1, 1) = 16 Simdgroups x 32 Lanes
    // Each threadgroup processes one query head.
    // D=256 only. 4-bit quantization (8 values per uint32).
    //
    // Threadgroup memory budget:
    //   tg_max[16] = 64 B, tg_sum[16] = 64 B, tg_acc[16*256] = 16 KB
    //   Total = 16,128 B -- well within 32 KB limit.

    const uint N_SIMD = 16;
    const uint D = 256;
    const uint DIMS_PER_LANE = 8;  // D / 32
    const uint WORDS = 32;         // D / 8 (4-bit packing)

    uint head = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_id = tid >> 5;    // 0..15 (simdgroup index)
    uint lane_id = tid & 31;    // 0..31 (lane within simdgroup)

    // Dimensions from input shapes
    uint T_kv = key_norms_shape[1];
    uint n_kv_heads = key_norms_shape[0];
    uint n_q_heads = q_rot_shape[0];
    uint n_repeats = n_q_heads / n_kv_heads;
    uint kv_head = head / n_repeats;

    // Load query (already scaled and rotated) -- 8 values per lane
    float q[8];
    uint q_off = head * D;
    for (uint i = 0; i < DIMS_PER_LANE; i++)
        q[i] = q_rot[q_off + lane_id * DIMS_PER_LANE + i];

    // KV offsets
    uint kv_off_norms = kv_head * T_kv;
    uint kv_off_packed = kv_head * T_kv * WORDS;

    // Each lane reads exactly 1 uint32 word (8 x 4-bit values)
    uint word_idx = lane_id;

    // ===== Scoring + Online Softmax + Value Accumulation =====
    float local_max = -1e10f;
    float local_sum = 0.0f;
    float local_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint k = simd_id; k < T_kv; k += N_SIMD) {
        // Unpack key: 4-bit values from one uint32 word
        uint32_t k_packed = key_packed[kv_off_packed + k * WORDS + word_idx];
        float partial = 0.0f;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint idx = (k_packed >> (i * 4)) & 0xFu;
            partial += q[i] * centroids[idx];
        }
        float score = simd_sum(partial) * key_norms[kv_off_norms + k];

        // Online Softmax
        float new_max = max(local_max, score);
        float factor = metal::fast::exp(local_max - new_max);
        float exp_s = metal::fast::exp(score - new_max);
        local_max = new_max;
        local_sum = local_sum * factor + exp_s;

        // Value accumulation (in rotated space)
        uint32_t v_packed = value_packed[kv_off_packed + k * WORDS + word_idx];
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint v_idx = (v_packed >> (i * 4)) & 0xFu;
            float vc = centroids[v_idx] * value_norms[kv_off_norms + k];
            local_acc[i] = local_acc[i] * factor + exp_s * vc;
        }
    }

    // ===== Cross-Simdgroup Reduction via Threadgroup Memory =====
    threadgroup float tg_max[16];
    threadgroup float tg_sum[16];
    threadgroup float tg_acc[16 * 256];

    if (lane_id == 0) {
        tg_max[simd_id] = local_max;
        tg_sum[simd_id] = local_sum;
    }
    for (uint i = 0; i < DIMS_PER_LANE; i++)
        tg_acc[simd_id * D + lane_id * DIMS_PER_LANE + i] = local_acc[i];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Combine all simdgroups
    float global_max = -1e10f;
    for (uint s = 0; s < N_SIMD; s++)
        global_max = max(global_max, tg_max[s]);

    float global_sum = 0.0f;
    float result[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint s = 0; s < N_SIMD; s++) {
        float factor = metal::fast::exp(tg_max[s] - global_max);
        global_sum += tg_sum[s] * factor;
        for (uint i = 0; i < DIMS_PER_LANE; i++)
            result[i] += tg_acc[s * D + lane_id * DIMS_PER_LANE + i] * factor;
    }

    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    for (uint i = 0; i < DIMS_PER_LANE; i++)
        output_rot[head * D + lane_id * DIMS_PER_LANE + i] = result[i] * inv;
"""

_fused_attn_4bit_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_attention_4bit",
    input_names=[
        "q_rot", "key_packed", "centroids",
        "key_norms", "value_packed", "value_norms",
    ],
    output_names=["output_rot"],
    source=_FUSED_ATTN_4BIT_SOURCE,
    header=_FUSED_ATTN_4BIT_HEADER,
)


def fused_tq_attention_4bit(
    q_rot: mx.array,
    key_packed: mx.array,
    centroids: mx.array,
    key_norms: mx.array,
    value_packed: mx.array,
    value_norms: mx.array,
    n_q_heads: int,
    D: int,
) -> mx.array:
    """Fused 4-bit TurboQuant attention (16 simdgroups, D=256 only).

    Query must be pre-rotated. Output is in rotated space.

    Args:
        q_rot: (n_q_heads, D) float32 -- scaled, rotated queries
        key_packed: (n_kv_heads, T_kv, D//8) uint32 -- 4-bit packed
        centroids: (16,) float32 -- Lloyd-Max centroids
        key_norms: (n_kv_heads, T_kv) float32
        value_packed: (n_kv_heads, T_kv, D//8) uint32 -- 4-bit packed
        value_norms: (n_kv_heads, T_kv) float32
        n_q_heads: number of query heads
        D: head dimension (must be 256)

    Returns:
        output_rot: (n_q_heads, D) float32 -- in rotated space
    """
    if D != 256:
        raise ValueError(f"This kernel only supports D=256, got {D}")
    assert centroids.shape == (16,), f"4-bit requires 16 centroids, got {centroids.shape}"

    outputs = _fused_attn_4bit_kernel(
        inputs=[q_rot, key_packed, centroids, key_norms, value_packed, value_norms],
        grid=(n_q_heads * 512, 1, 1),
        threadgroup=(512, 1, 1),
        output_shapes=[(n_q_heads * D,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_q_heads, D)
