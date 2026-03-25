"""TurboQuant optimized attention — MLX-native, fully vectorized.

Optimizations vs. base version:
  1. Combined rotation+JL sketch in one matmul (1 dispatch instead of 2)
  2. Value associativity: (weights @ centroids) @ Pi instead of weights @ (centroids @ Pi)
     -> saves O(T_kv x D^2) per token, reduced to O(T_q x D^2)
  3. No Python loop, full MLX graph optimization
"""

import math

import mlx.core as mx

from turboquant.kernels import unpack_2bit_indices
from turboquant.qjl import unpack_sign_bits


def turboquant_scaled_dot_product_attention(
    queries: mx.array,
    cache,
    scale: float,
    mask=None,
) -> mx.array:
    """Optimized TurboQuant attention.

    Args:
        queries: (B, n_q_heads, T_q, D)
        cache: TurboQuantKVCache
        scale: Typically 1/sqrt(D)
        mask: "causal", bool array, or None

    Returns:
        output: (B, n_q_heads, T_q, D)
    """
    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = cache.key_packed.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    T_kv = cache.offset

    # --- 1. Rotation (+ JL sketch when QJL active) ---
    q_scaled = queries * scale
    if cache.use_qjl:
        q_combined = q_scaled @ cache.combined_rot_jl.T  # (B, n_q_heads, T_q, 2D)
        q_rot = q_combined[..., :D]
        q_sketch = q_combined[..., D:]
    else:
        q_rot = q_scaled @ cache.rotation_matrix.T

    # --- 2. MSE-Score ---
    key_indices = cache.get_key_indices()[:, :, :T_kv, :]
    key_centroids = cache.centroids[key_indices]

    q_rot_grouped = q_rot.reshape(B, n_kv_heads, n_repeats, T_q, D)
    key_centroids_expanded = key_centroids[:, :, None, :, :]

    scores = q_rot_grouped @ key_centroids_expanded.transpose(0, 1, 2, 4, 3)
    scores = scores * cache.key_norms[:, :, :T_kv][:, :, None, None, :]

    # --- 3. QJL-Score (optional) ---
    if cache.use_qjl:
        q_sketch_grouped = q_sketch.reshape(B, n_kv_heads, n_repeats, T_q, D)
        k_signs_float = unpack_sign_bits(cache.key_sign_bits[:, :, :T_kv, :])
        k_signs_expanded = k_signs_float[:, :, None, :, :]

        qjl_scores = q_sketch_grouped @ k_signs_expanded.transpose(0, 1, 2, 4, 3)
        qjl_scale = math.sqrt(math.pi / 2.0) / D
        qjl_scores = qjl_scores * qjl_scale * cache.key_residual_norms[:, :, :T_kv][:, :, None, None, :]
        scores = scores + qjl_scores

    # --- 5. Mask ---
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

    # --- 6. Softmax ---
    weights = mx.softmax(scores, axis=-1, precise=True)

    # --- 7. Value output via matrix associativity ---
    value_indices = cache.get_value_indices()[:, :, :T_kv, :]
    value_centroids = cache.centroids[value_indices]  # (B, n_kv_heads, T_kv, D)

    # Fold norms into weights: (B, kv, reps, T_q, T_kv) * (B, kv, 1, 1, T_kv)
    value_norms = cache.value_norms[:, :, :T_kv]
    weighted_norms = weights * value_norms[:, :, None, None, :]

    # Weighted centroid sum: (B, kv, reps, T_q, T_kv) @ (B, kv, 1, T_kv, D)
    value_centroids_expanded = value_centroids[:, :, None, :, :]
    weighted_centroids = weighted_norms @ value_centroids_expanded  # (B, kv, reps, T_q, D)

    # Single inverse rotation: (B, kv, reps, T_q, D) @ (D, D)
    output = weighted_centroids @ cache.rotation_matrix  # Pi^T @ c = c @ Pi
    output = output.reshape(B, n_q_heads, T_q, D)

    return output
