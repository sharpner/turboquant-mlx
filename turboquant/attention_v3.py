"""TurboQuant V3 attention — Lloyd-Max codebook with software dequantization.

Uses centroid lookup + regular mx.matmul instead of mx.quantized_matmul.

Supports:
  - TurboQuant_mse: b-bit Lloyd-Max codebook
  - TurboQuant_prod: (b-1)-bit key MSE + QJL score correction
  - Mixed bit allocation: outlier channels at higher precision

Matrix associativity optimization:
  output = weighted_centroids @ Pi
  saves O(T_kv * D^2) -> O(T_q * D^2) for inverse rotation.
"""

import math

import mlx.core as mx

from turboquant.qjl import unpack_sign_bits


def turboquant_v3_sdpa(
    queries: mx.array,
    cache,
    scale: float,
    mask=None,
) -> mx.array:
    """TurboQuant V3 SDPA with Lloyd-Max codebook dequantization."""
    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = cache.key_regular_packed.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    T_kv = cache.offset

    # --- Rotate query ---
    q_scaled = queries * scale
    if cache.use_qjl:
        q_combined = q_scaled @ cache.combined_rot_jl.T
        q_rot = q_combined[..., :D]
        q_sketch = q_combined[..., D:]
    else:
        q_rot = q_scaled @ cache.rotation_matrix.T

    # --- Dequant keys to centroids ---
    k_centroids = cache.get_key_centroids()  # (B, n_kv_heads, T_kv, D)

    # --- GQA reshape ---
    q_rot_grouped = q_rot.reshape(B, n_kv_heads, n_repeats, T_q, D)
    k_centroids_exp = k_centroids[:, :, None, :, :]

    # --- MSE Scores (before key norms) ---
    scores = q_rot_grouped @ k_centroids_exp.transpose(0, 1, 2, 4, 3)

    # --- QJL key correction ---
    if cache.use_qjl and cache.key_sign_bits is not None:
        q_sketch_grouped = q_sketch.reshape(B, n_kv_heads, n_repeats, T_q, D)
        k_signs = unpack_sign_bits(cache.key_sign_bits[:, :, :T_kv, :])
        k_signs_exp = k_signs[:, :, None, :, :]

        qjl_scores = q_sketch_grouped @ k_signs_exp.transpose(0, 1, 2, 4, 3)
        qjl_scale = math.sqrt(math.pi / 2.0) / D
        qjl_scores = qjl_scores * qjl_scale * cache.key_residual_norms[:, :, :T_kv][:, :, None, None, :]
        scores = scores + qjl_scores

    # --- Multiply by key norms AFTER QJL correction ---
    scores = scores * cache.key_norms[:, :, :T_kv][:, :, None, None, :]

    # --- Mask ---
    if mask is not None:
        if isinstance(mask, str):
            q_indices = mx.arange(T_kv - T_q, T_kv)
            k_indices_range = mx.arange(T_kv)
            causal_mask = q_indices[:, None] >= k_indices_range[None]
            scores = mx.where(causal_mask, scores, mx.finfo(scores.dtype).min)
        elif mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    # --- Softmax ---
    weights = mx.softmax(scores, axis=-1, precise=True)

    # --- Value output ---
    v_centroids = cache.get_value_centroids()  # (B, n_kv_heads, T_kv, D)
    v_centroids_exp = v_centroids[:, :, None, :, :]

    v_norms = cache.value_norms[:, :, :T_kv]
    weighted_norms = weights * v_norms[:, :, None, None, :]

    # Weighted centroid sum + inverse rotation
    weighted_centroids = weighted_norms @ v_centroids_exp
    output = weighted_centroids @ cache.rotation_matrix
    return output.reshape(B, n_q_heads, T_q, D)
