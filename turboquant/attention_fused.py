"""TurboQuant fused attention — EVERYTHING in one Metal dispatch.

Uses the fused kernel for T_q=1 (token-by-token generation).
Falls back to MLX ops for T_q>1 (prefill).
"""

import math

import mlx.core as mx

from turboquant.kernels import fused_tq_attention_norot
from turboquant.qjl import unpack_sign_bits


def turboquant_fused_sdpa(
    queries: mx.array,
    cache,
    scale: float,
    mask=None,
) -> mx.array:
    """TurboQuant attention — fused kernel for generation, MLX ops for prefill.

    Args:
        queries: (B, n_q_heads, T_q, D)
        cache: TurboQuantKVCache
        scale: 1/sqrt(D)
        mask: "causal", bool array, or None

    Returns:
        output: (B, n_q_heads, T_q, D)
    """
    B, n_q_heads, T_q, D = queries.shape
    T_kv = cache.offset

    # === FUSED PATH: T_q=1, B=1 (token-by-token generation) ===
    # Rotation via MLX GEMM (optimized), kernel only for quantized attention.
    if T_q == 1 and B == 1 and not cache.use_qjl:
        n_kv_heads = cache.key_packed.shape[1]

        # Pre-rotate query (MLX optimized GEMM — much faster than in-kernel)
        q_flat = queries.reshape(n_q_heads, D)
        q_rot = (q_flat * scale) @ cache.rotation_matrix.T

        # Fused quantized attention in rotated space (32 simdgroups)
        out_rot = fused_tq_attention_norot(
            q_rot,
            cache.key_packed.squeeze(0),
            cache.centroids,
            cache.key_norms.squeeze(0),
            cache.value_packed.squeeze(0),
            cache.value_norms.squeeze(0),
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            D=D,
        )

        # Inverse rotation (MLX optimized GEMM)
        output = out_rot @ cache.rotation_matrix
        return output.reshape(B, n_q_heads, T_q, D)

    # === FALLBACK: MLX ops for prefill (T_q>1) or QJL ===
    n_kv_heads = cache.key_packed.shape[1]
    n_repeats = n_q_heads // n_kv_heads

    q_scaled = queries * scale
    if cache.use_qjl:
        q_combined = q_scaled @ cache.combined_rot_jl.T
        q_rot = q_combined[..., :D]
        q_sketch = q_combined[..., D:]
    else:
        q_rot = q_scaled @ cache.rotation_matrix.T

    key_indices = cache.get_key_indices()[:, :, :T_kv, :]
    key_centroids = cache.centroids[key_indices]

    q_rot_grouped = q_rot.reshape(B, n_kv_heads, n_repeats, T_q, D)
    key_centroids_expanded = key_centroids[:, :, None, :, :]

    scores = q_rot_grouped @ key_centroids_expanded.transpose(0, 1, 2, 4, 3)
    scores = scores * cache.key_norms[:, :, :T_kv][:, :, None, None, :]

    if cache.use_qjl:
        q_sketch_grouped = q_sketch.reshape(B, n_kv_heads, n_repeats, T_q, D)
        k_signs_float = unpack_sign_bits(cache.key_sign_bits[:, :, :T_kv, :])
        k_signs_expanded = k_signs_float[:, :, None, :, :]
        qjl_scores = q_sketch_grouped @ k_signs_expanded.transpose(0, 1, 2, 4, 3)
        qjl_scale = math.sqrt(math.pi / 2.0) / D
        qjl_scores = qjl_scores * qjl_scale * cache.key_residual_norms[:, :, :T_kv][:, :, None, None, :]
        scores = scores + qjl_scores

    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            q_indices = mx.arange(T_kv - T_q, T_kv)
            k_indices = mx.arange(T_kv)
            causal_mask = q_indices[:, None] >= k_indices[None]
            scores = mx.where(causal_mask, scores, mx.finfo(scores.dtype).min)
        elif isinstance(mask, mx.array):
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask

    weights = mx.softmax(scores, axis=-1, precise=True)

    value_indices = cache.get_value_indices()[:, :, :T_kv, :]
    value_centroids = cache.centroids[value_indices]
    value_norms = cache.value_norms[:, :, :T_kv]
    weighted_norms = weights * value_norms[:, :, None, None, :]
    weighted_centroids = weighted_norms @ value_centroids[:, :, None, :, :]
    output = weighted_centroids @ cache.rotation_matrix
    output = output.reshape(B, n_q_heads, T_q, D)

    return output
