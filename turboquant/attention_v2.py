"""TurboQuant V2 attention — Uses mx.quantized_matmul for hardware affinity.

Identical to MLX's quantized_scaled_dot_product_attention,
extended with optional random QR rotation and QJL correction.

QJL uses a fused Metal kernel for sign-bit dot products (T_q=1),
avoiding the 32x memory blowup from unpacking sign bits to float.
"""

import mlx.core as mx
from mlx.utils import tree_map

from turboquant.fused_qjl import fused_qjl_scores
from turboquant.qjl import unpack_sign_bits


def turboquant_v2_sdpa(
    queries: mx.array,
    q_keys: tuple,
    q_values: tuple,
    cache,
    scale: float,
    mask=None,
) -> mx.array:
    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = q_keys[0].shape[1]
    n_repeats = n_q_heads // n_kv_heads
    T_kv = cache.offset

    # Rotate query (optional)
    q_scaled = queries * scale
    if cache.use_qjl:
        q_combined = q_scaled @ cache.combined_rot_jl.T
        q_rot = q_combined[..., :D]
        q_sketch = q_combined[..., D:]
    elif cache.use_rotation:
        q_rot = q_scaled @ cache.rotation_matrix.T
    else:
        q_rot = q_scaled

    # GQA: reshape + expand
    if n_repeats > 1:
        q_rot = q_rot.reshape(B, n_kv_heads, n_repeats, T_q, D)
    else:
        q_rot = q_rot[:, :, None, :, :]
    q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
    q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    # Scores via native quantized_matmul
    scores = mx.quantized_matmul(
        q_rot, *q_keys,
        transpose=True, group_size=cache.group_size, bits=cache.bits,
    )

    # QJL correction (optional)
    if cache.use_qjl and cache.key_sign_bits is not None:
        if T_q == 1:
            # Fused Metal kernel: reads packed sign bits directly
            q_sketch_flat = q_sketch.reshape(B * n_kv_heads * n_repeats, D)
            sign_bits_flat = cache.key_sign_bits[:, :, :T_kv, :].reshape(B * n_kv_heads, T_kv, -1)
            norms_flat = cache.key_residual_norms[:, :, :T_kv].reshape(B * n_kv_heads, T_kv)
            qjl_flat = fused_qjl_scores(q_sketch_flat, sign_bits_flat, norms_flat, D, cache.qjl_scale)
            qjl_scores = qjl_flat.reshape(B, n_kv_heads, n_repeats, 1, T_kv)
        else:
            # Prefill fallback: unpack + matmul
            q_sketch_grouped = q_sketch.reshape(B, n_kv_heads, n_repeats, T_q, D)
            k_signs = unpack_sign_bits(cache.key_sign_bits[:, :, :T_kv, :])
            k_signs_exp = k_signs[:, :, None, :, :]
            qjl_scores = q_sketch_grouped @ k_signs_exp.transpose(0, 1, 2, 4, 3)
            qjl_scores = qjl_scores * cache.qjl_scale * cache.key_residual_norms[:, :, :T_kv][:, :, None, None, :]
        scores = scores + qjl_scores

    # Mask
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    # Softmax + Value Output
    weights = mx.softmax(scores, axis=-1, precise=True)
    output = mx.quantized_matmul(
        weights, *q_values,
        transpose=False, group_size=cache.group_size, bits=cache.bits,
    )

    # Inverse rotation (optional)
    if cache.use_rotation:
        output = output @ cache.rotation_matrix

    return output.reshape(B, n_q_heads, T_q, D)
