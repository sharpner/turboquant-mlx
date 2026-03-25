"""Custom Metal kernels for TurboQuant.

Strategy: Matmul operations (rotation, centroid lookup) via mx.matmul/mx.take.
Only bit operations (quantization, sign packing, QJL scoring) as Metal kernels.
"""

import mlx.core as mx


# --- Kernel 1: Scalar Quantize (binary search on boundaries) ---

_QUANTIZE_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint num_boundaries = boundaries_shape[0];

    float val = rotated[elem];

    // Binary search: count how many boundaries are less than val
    uint idx = 0;
    for (uint b = 0; b < num_boundaries; b++) {
        idx += (val > boundaries[b]) ? 1 : 0;
    }
    indices[elem] = static_cast<uint8_t>(idx);
"""

_quantize_kernel = mx.fast.metal_kernel(
    name="turboquant_quantize",
    input_names=["rotated", "boundaries"],
    output_names=["indices"],
    source=_QUANTIZE_SOURCE,
)


def quantize_to_indices(rotated: mx.array, boundaries: mx.array) -> mx.array:
    """Quantizes rotated values to centroid indices via binary search.

    Args:
        rotated: Any shape, float32 — the rotated (and normalized) coordinates
        boundaries: (num_levels - 1,) float32 — sorted decision boundaries

    Returns:
        indices: Same shape as rotated, uint8
    """
    flat = rotated.reshape(-1)
    outputs = _quantize_kernel(
        inputs=[flat, boundaries],
        grid=(flat.size, 1, 1),
        threadgroup=(min(256, flat.size), 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[mx.uint8],
    )
    return outputs[0].reshape(rotated.shape)


# --- Kernel 2: Pack Sign Bits in uint32 ---

_PACK_SIGNS_SOURCE = """
    // One thread per uint32 word (packs 32 sign bits)
    uint word_idx = thread_position_in_grid.x;
    uint base = word_idx * 32;
    uint total = values_shape[0];

    uint32_t packed = 0;
    for (uint bit = 0; bit < 32; bit++) {
        uint idx = base + bit;
        if (idx < total) {
            packed |= ((values[idx] > 0.0f) ? 1u : 0u) << bit;
        }
    }
    sign_bits[word_idx] = packed;
"""

_pack_signs_kernel = mx.fast.metal_kernel(
    name="turboquant_pack_signs",
    input_names=["values"],
    output_names=["sign_bits"],
    source=_PACK_SIGNS_SOURCE,
)


def pack_sign_bits(values: mx.array) -> mx.array:
    """Packs the signs of values into uint32 words (32 bits per word).

    Args:
        values: (..., D) float32 — the projection values

    Returns:
        sign_bits: (..., D // 32) uint32 — packed sign bits
    """
    original_shape = values.shape
    D = original_shape[-1]
    if D % 32 != 0:
        raise ValueError(f"Last dimension must be divisible by 32, got {D}")

    flat = values.reshape(-1)
    num_words = flat.size // 32
    outputs = _pack_signs_kernel(
        inputs=[flat],
        grid=(num_words, 1, 1),
        threadgroup=(min(256, num_words), 1, 1),
        output_shapes=[(num_words,)],
        output_dtypes=[mx.uint32],
    )
    out_shape = original_shape[:-1] + (D // 32,)
    return outputs[0].reshape(out_shape)


# --- Kernel 3: Pack 2-bit Indices in uint32 (16 indices per word) ---

_PACK_2BIT_SOURCE = """
    // One thread per uint32 word (packs 16 2-bit indices)
    uint word_idx = thread_position_in_grid.x;
    uint base = word_idx * 16;
    uint total = indices_shape[0];

    uint32_t packed = 0;
    for (uint i = 0; i < 16; i++) {
        uint idx = base + i;
        if (idx < total) {
            packed |= (static_cast<uint32_t>(indices[idx]) & 0x3u) << (i * 2);
        }
    }
    packed_indices[word_idx] = packed;
"""

_pack_2bit_kernel = mx.fast.metal_kernel(
    name="turboquant_pack_2bit",
    input_names=["indices"],
    output_names=["packed_indices"],
    source=_PACK_2BIT_SOURCE,
)


def pack_2bit_indices(indices: mx.array) -> mx.array:
    """Packs 2-bit indices (0-3) into uint32 words (16 indices per word).

    Args:
        indices: (..., D) uint8 with values 0-3

    Returns:
        packed: (..., D // 16) uint32
    """
    original_shape = indices.shape
    D = original_shape[-1]
    if D % 16 != 0:
        raise ValueError(f"Last dimension must be divisible by 16, got {D}")

    flat = indices.reshape(-1)
    num_words = flat.size // 16
    outputs = _pack_2bit_kernel(
        inputs=[flat],
        grid=(num_words, 1, 1),
        threadgroup=(min(256, num_words), 1, 1),
        output_shapes=[(num_words,)],
        output_dtypes=[mx.uint32],
    )
    out_shape = original_shape[:-1] + (D // 16,)
    return outputs[0].reshape(out_shape)


# Bit masks for 2-bit unpacking (computed once)
_SHIFTS_2BIT = mx.array([i * 2 for i in range(16)], dtype=mx.uint32)


def unpack_2bit_indices(packed: mx.array, D: int) -> mx.array:
    """Unpacks uint32 to 2-bit indices (0-3).

    Args:
        packed: (..., D // 16) uint32
        D: Original dimension (e.g. 128)

    Returns:
        indices: (..., D) uint32 with values 0-3
    """
    # Expand each uint32 to 16 2-bit values
    expanded = (packed[..., None] >> _SHIFTS_2BIT) & 0x3  # (..., D//16, 16)
    return expanded.reshape(*packed.shape[:-1], D)


# --- Kernel 5: Pack 3-bit Indices in uint32 (10 indices per word, 2 bits unused) ---

_PACK_3BIT_SOURCE = """
    uint word_idx = thread_position_in_grid.x;
    uint base = word_idx * 10;
    uint total = indices_shape[0];

    uint32_t packed = 0;
    for (uint i = 0; i < 10; i++) {
        uint idx = base + i;
        if (idx < total) {
            packed |= (static_cast<uint32_t>(indices[idx]) & 0x7u) << (i * 3);
        }
    }
    packed_indices[word_idx] = packed;
"""

_pack_3bit_kernel = mx.fast.metal_kernel(
    name="turboquant_pack_3bit",
    input_names=["indices"],
    output_names=["packed_indices"],
    source=_PACK_3BIT_SOURCE,
)


def pack_3bit_indices(indices: mx.array) -> mx.array:
    """Packs 3-bit indices (0-7) into uint32 words (10 indices per word).

    Args:
        indices: (..., D) uint8 with values 0-7

    Returns:
        packed: (..., ceil(D / 10)) uint32
    """
    original_shape = indices.shape
    D = original_shape[-1]
    # Padding to multiple of 10
    num_words = (D + 9) // 10

    flat = indices.reshape(-1)
    total_words = flat.size // D * num_words
    # Pad to multiple of 10
    if D % 10 != 0:
        pad_size = num_words * 10 - D
        batch_size = flat.size // D
        flat_padded = flat.reshape(-1, D)
        flat_padded = mx.pad(flat_padded, [(0, 0), (0, pad_size)])
        flat = flat_padded.reshape(-1)

    total_elements = flat.size
    total_words = total_elements // 10
    outputs = _pack_3bit_kernel(
        inputs=[flat],
        grid=(total_words, 1, 1),
        threadgroup=(min(256, total_words), 1, 1),
        output_shapes=[(total_words,)],
        output_dtypes=[mx.uint32],
    )
    out_shape = original_shape[:-1] + (num_words,)
    return outputs[0].reshape(out_shape)


_SHIFTS_3BIT = mx.array([i * 3 for i in range(10)], dtype=mx.uint32)


def unpack_3bit_indices(packed: mx.array, D: int) -> mx.array:
    """Unpacks uint32 to 3-bit indices (0-7).

    Args:
        packed: (..., ceil(D/10)) uint32
        D: Original dimension

    Returns:
        indices: (..., D) uint32 with values 0-7
    """
    expanded = (packed[..., None] >> _SHIFTS_3BIT) & 0x7  # (..., words, 10)
    total = packed.shape[-1] * 10
    result = expanded.reshape(*packed.shape[:-1], total)
    # Trim to actual D
    return result[..., :D]


# --- High-level encode/decode functions (use kernels + MLX ops) ---

def turboquant_encode(
    vectors: mx.array,
    rotation_matrix: mx.array,
    boundaries: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """TurboQuant encoding: Normalize -> Rotate -> Scalar Quantize.

    Args:
        vectors: (..., D) float32 — input vectors
        rotation_matrix: (D, D) float32 — orthogonal matrix Pi
        boundaries: (num_levels - 1,) float32 — quantization boundaries (unscaled)

    Returns:
        indices: (..., D) uint8 — centroid indices
        norms: (...,) float32 — L2 norms of inputs
        rotated_normalized: (..., D) float32 — rotated normalized vectors (for residual)
    """
    # L2 norm per vector
    norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
    # Guard: handle zero vectors
    safe_norms = mx.where(norms < 1e-8, mx.ones_like(norms), norms)
    normalized = vectors / safe_norms

    # Rotate: x_rot = x_norm @ Pi^T
    rotated = normalized @ rotation_matrix.T

    # Scalar quantize via Metal kernel
    indices = quantize_to_indices(rotated, boundaries)

    return indices, norms.squeeze(-1), rotated


def turboquant_decode(
    indices: mx.array,
    rotation_matrix: mx.array,
    centroids: mx.array,
    norms: mx.array,
) -> mx.array:
    """TurboQuant decoding: Centroid lookup -> Inverse rotation -> Scaling.

    Args:
        indices: (..., D) uint8 — centroid indices
        rotation_matrix: (D, D) float32 — orthogonal matrix Pi
        centroids: (num_levels,) float32 — centroid values (unscaled)
        norms: (...,) float32 — L2 norms

    Returns:
        reconstructed: (..., D) float32 — reconstructed vectors
    """
    # Centroid lookup
    centroid_values = centroids[indices.astype(mx.uint32)]

    # Inverse rotation: x = c @ Pi (= Pi^T @ c since Pi is orthogonal -> Pi^(-1) = Pi^T)
    reconstructed = centroid_values @ rotation_matrix

    # Scale by original norm
    return reconstructed * norms[..., None]


def qjl_encode(
    residual: mx.array,
    jl_matrix: mx.array,
) -> tuple[mx.array, mx.array]:
    """QJL encoding: Project -> Pack sign bits.

    Args:
        residual: (..., D) float32 — residual vectors (x_norm - dequant_mse(x_norm))
        jl_matrix: (D, D) float32 — JL projection matrix S

    Returns:
        sign_bits: (..., D // 32) uint32 — packed sign bits
        residual_norms: (...,) float32 — ||residual||_2 (gamma)
    """
    # Residual norm (gamma in the paper)
    residual_norms = mx.linalg.norm(residual, axis=-1)

    # Projection: p = residual @ S^T
    projected = residual @ jl_matrix.T

    # Pack sign bits via Metal kernel
    sign_bits = pack_sign_bits(projected)

    return sign_bits, residual_norms


# ===========================================================================
# HIGH-OCCUPANCY FUSED KERNEL — 32 simdgroups, rotation outside
# ===========================================================================

_FUSED_SCORE_SOURCE_DEAD = """
    // Grid: (T_kv, n_repeats, 1) — one thread per (key, repeat) pair
    uint k = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;

    uint T_kv = key_norms_shape[0];
    uint D_DIV_16 = key_packed_shape[1];
    uint D_DIV_32 = key_sign_bits_shape[1];
    uint D = D_DIV_16 * 16;
    uint n_repeats = q_rot_shape[0];  // q_rot is (n_repeats, D)

    if (k >= T_kv) return;

    // q_rot and q_sketch are laid out as (n_repeats, D)
    uint q_base = r * D;

    // 1. MSE Score: Unpack 2-bit indices + Centroid lookup + Dot product
    float mse_score = 0.0f;
    for (uint w = 0; w < D_DIV_16; w++) {
        uint32_t packed = key_packed[k * D_DIV_16 + w];
        for (uint i = 0; i < 16; i++) {
            uint idx = (packed >> (i * 2)) & 0x3u;
            float c = centroids[idx];
            mse_score += q_rot[q_base + w * 16 + i] * c;
        }
    }
    mse_score *= key_norms[k];

    // 2. QJL Score: Unpack sign bits + Dot product with q_sketch
    float qjl_score = 0.0f;
    for (uint w = 0; w < D_DIV_32; w++) {
        uint32_t signs = key_sign_bits[k * D_DIV_32 + w];
        for (uint bit = 0; bit < 32; bit++) {
            float sign_val = ((signs >> bit) & 1u) ? 1.0f : -1.0f;
            qjl_score += q_sketch[q_base + w * 32 + bit] * sign_val;
        }
    }
    qjl_score *= qjl_scale[0] * key_residual_norms[k];

    scores[k * n_repeats + r] = mse_score + qjl_score;
"""

_fused_score_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_scores",
    input_names=["q_rot", "q_sketch", "key_packed", "centroids",
                 "key_norms", "key_sign_bits", "key_residual_norms", "qjl_scale"],
    output_names=["scores"],
    source=_FUSED_SCORE_SOURCE_DEAD,
)


def fused_tq_scores(
    q_rot: mx.array,
    q_sketch: mx.array,
    key_packed: mx.array,
    centroids: mx.array,
    key_norms: mx.array,
    key_sign_bits: mx.array,
    key_residual_norms: mx.array,
    qjl_scale: mx.array,
) -> mx.array:
    """Fused TurboQuant score: MSE + QJL in one kernel.

    Args:
        q_rot: (n_repeats, D) float32 — rotated queries for one KV head
        q_sketch: (n_repeats, D) float32 — JL sketches
        key_packed: (T_kv, D//16) uint32 — 2-bit packed keys
        centroids: (4,) float32
        key_norms: (T_kv,) float32
        key_sign_bits: (T_kv, D//32) uint32
        key_residual_norms: (T_kv,) float32
        qjl_scale: (1,) float32 — sqrt(pi/2)/D

    Returns:
        scores: (T_kv, n_repeats) float32
    """
    T_kv = key_packed.shape[0]
    n_repeats = q_rot.shape[0]

    if T_kv == 0:
        return mx.zeros((T_kv, n_repeats))

    outputs = _fused_score_kernel(
        inputs=[q_rot, q_sketch, key_packed, centroids,
                key_norms, key_sign_bits, key_residual_norms, qjl_scale],
        grid=(T_kv, n_repeats, 1),
        threadgroup=(min(256, T_kv), 1, 1),
        output_shapes=[(T_kv * n_repeats,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(T_kv, n_repeats)


# --- Fused Kernel 2: TurboQuant Value Output ---

_FUSED_VALUE_SOURCE = """
    // Grid: (D, n_repeats, 1) — one thread per (output_dim, repeat) pair
    uint d = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;

    uint T_kv = value_norms_shape[0];
    uint D_DIV_16 = value_packed_shape[1];
    uint D = D_DIV_16 * 16;
    uint n_repeats = weights_shape[1];

    if (d >= D) return;

    float result = 0.0f;

    for (uint k = 0; k < T_kv; k++) {
        float w = weights[k * n_repeats + r];
        if (w < 1e-8f) continue;

        // Decode value[k][d]: sum_j(centroid[idx_j] * Pi[j][d])
        float v_d = 0.0f;
        for (uint ww = 0; ww < D_DIV_16; ww++) {
            uint32_t packed = value_packed[k * D_DIV_16 + ww];
            for (uint i = 0; i < 16; i++) {
                uint idx = (packed >> (i * 2)) & 0x3u;
                uint j = ww * 16 + i;
                v_d += centroids[idx] * rotation_matrix[j * D + d];
            }
        }
        v_d *= value_norms[k];
        result += w * v_d;
    }

    output[r * D + d] = result;
"""

_fused_value_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_value",
    input_names=["weights", "value_packed", "centroids",
                 "value_norms", "rotation_matrix"],
    output_names=["output"],
    source=_FUSED_VALUE_SOURCE,
)


def fused_tq_value_output(
    weights: mx.array,
    value_packed: mx.array,
    centroids: mx.array,
    value_norms: mx.array,
    rotation_matrix: mx.array,
    D: int,
    n_repeats: int,
) -> mx.array:
    """Fused TurboQuant value output: Decode + weighted sum in one kernel.

    Args:
        weights: (T_kv, n_repeats) float32 — softmax weights
        value_packed: (T_kv, D//16) uint32 — 2-bit packed values
        centroids: (4,) float32
        value_norms: (T_kv,) float32
        rotation_matrix: (D, D) float32 — Pi

    Returns:
        output: (n_repeats, D) float32
    """
    outputs = _fused_value_kernel(
        inputs=[weights, value_packed, centroids, value_norms, rotation_matrix],
        grid=(D, n_repeats, 1),
        threadgroup=(min(128, D), 1, 1),
        output_shapes=[(n_repeats * D,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_repeats, D)


# ===========================================================================
# FUSED FULL ATTENTION KERNEL — Everything in one Metal dispatch
# ===========================================================================
# Rotation + Quantized Scoring + Online Softmax + Value Accumulation + Inverse Rotation
# Pattern: 1 simdgroup (32 threads) per query head, sequential over keys.

_FUSED_ATTN_HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

_FUSED_ATTN_SOURCE = """
    // Grid: (n_q_heads, 1, 1), Threadgroup: (32, 1, 1)
    // One simdgroup (32 threads) per query head
    // Thread lid handles dims [lid*4, lid*4+1, lid*4+2, lid*4+3]

    uint head = threadgroup_position_in_grid.x;
    uint lid = thread_index_in_simdgroup;

    // Derive dimensions from input shapes
    uint T_kv = key_norms_shape[1];        // key_norms: (n_kv_heads, T_kv)
    uint n_kv_heads = key_norms_shape[0];
    uint n_q_heads = queries_shape[0];      // queries: (n_q_heads, D)
    uint WORDS = key_packed_shape[2];       // key_packed: (n_kv_heads, T_kv, D/16)
    uint D = WORDS * 16;
    uint n_repeats = n_q_heads / n_kv_heads;
    uint kv_head = head / n_repeats;

    // Offsets
    uint q_off = head * D;
    uint kv_off_norms = kv_head * T_kv;
    uint kv_off_packed = kv_head * T_kv * WORDS;

    // ===== PHASE 1: Rotate Query =====
    // q_rot[d] = scale * sum_j(q[j] * Pi[d][j])
    // Load ALL query values into shared memory, then each thread
    // independently computes its 4 output dimensions

    threadgroup float shared_q[128];
    for (uint i = 0; i < 4; i++)
        shared_q[lid * 4 + i] = queries[q_off + lid * 4 + i] * scale[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes q_rot for dims [lid*4 .. lid*4+3]
    float q_rot[4];
    for (uint di = 0; di < 4; di++) {
        uint d = lid * 4 + di;
        float sum = 0.0f;
        for (uint j = 0; j < D; j++)
            sum += shared_q[j] * rotation_matrix[d * D + j];
        q_rot[di] = sum;
    }

    // ===== PHASE 2: Scoring + Online Softmax + Value Accumulation =====
    float max_score = -HUGE_VALF;
    float sum_exp = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Accumulator in rotated space

    uint word_idx = lid / 4;      // Which uint32 word (0..7)
    uint bit_base = (lid % 4) * 8; // Bit offset within word (0, 8, 16, 24)

    for (uint k = 0; k < T_kv; k++) {
        // -- Unpack key centroids for this thread's 4 dims --
        uint32_t k_packed = key_packed[kv_off_packed + k * WORDS + word_idx];
        float kc[4];
        for (uint i = 0; i < 4; i++) {
            uint idx = (k_packed >> (bit_base + i * 2)) & 0x3u;
            kc[i] = centroids[idx];
        }

        // -- MSE Score: dot(q_rot, key_centroids) via simd_sum --
        float partial = 0.0f;
        for (uint i = 0; i < 4; i++)
            partial += q_rot[i] * kc[i];
        float score = simd_sum(partial) * key_norms[kv_off_norms + k];

        // -- Causal mask: skip future keys --
        // offset = T_kv - 1 (current query position for T_q=1)
        if (k > T_kv - 1 + causal_offset[0])
            score = -HUGE_VALF;

        // -- Online Softmax --
        float new_max = max(max_score, score);
        float factor = metal::fast::exp(max_score - new_max);
        float exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        // -- Accumulate value centroids (in rotated space) --
        uint32_t v_packed = value_packed[kv_off_packed + k * WORDS + word_idx];
        for (uint i = 0; i < 4; i++) {
            uint v_idx = (v_packed >> (bit_base + i * 2)) & 0x3u;
            float vc = centroids[v_idx] * value_norms[kv_off_norms + k];
            acc[i] = acc[i] * factor + exp_score * vc;
        }
    }

    // Normalize by sum_exp
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (uint i = 0; i < 4; i++)
        acc[i] *= inv_sum;

    // ===== PHASE 3: Inverse Rotation =====
    // output[d] = sum_j(acc[j] * Pi[j][d])
    // acc is distributed: thread lid holds acc for dims [lid*4..lid*4+3]
    // Store acc to threadgroup memory so all threads can read all 128 values
    threadgroup float shared_acc[128];
    for (uint i = 0; i < 4; i++)
        shared_acc[lid * 4 + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes 4 output dimensions
    for (uint di = 0; di < 4; di++) {
        uint d = lid * 4 + di;
        float sum = 0.0f;
        for (uint j = 0; j < D; j++)
            sum += shared_acc[j] * rotation_matrix[j * D + d];
        output[head * D + d] = sum;
    }
"""

_fused_attn_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_attention",
    input_names=[
        "queries", "rotation_matrix", "key_packed", "centroids",
        "key_norms", "value_packed", "value_norms", "scale", "causal_offset",
    ],
    output_names=["output"],
    source=_FUSED_ATTN_SOURCE,
    header=_FUSED_ATTN_HEADER,
)


def fused_tq_attention(
    queries: mx.array,
    rotation_matrix: mx.array,
    key_packed: mx.array,
    centroids: mx.array,
    key_norms: mx.array,
    value_packed: mx.array,
    value_norms: mx.array,
    scale: float,
    n_q_heads: int,
    D: int,
    causal_offset: int = 0,
) -> mx.array:
    """Fully fused TurboQuant attention in ONE Metal dispatch.

    Rotation + Quantized Scoring + Online Softmax + Value Accumulation + Inverse Rotation.

    Args:
        queries: (n_q_heads, D) float32 — ALL query heads (flattened batch)
        rotation_matrix: (D, D) float32 — Pi
        key_packed: (n_kv_heads, T_kv, D//16) uint32 — 2-bit packed
        centroids: (4,) float32
        key_norms: (n_kv_heads, T_kv) float32
        value_packed: (n_kv_heads, T_kv, D//16) uint32
        value_norms: (n_kv_heads, T_kv) float32
        scale: Attention scale (1/sqrt(D))
        n_q_heads: Total query heads
        D: Head dimension
        causal_offset: 0 for generation (T_q=1)

    Returns:
        output: (n_q_heads, D) float32
    """
    scale_arr = mx.array([scale], dtype=mx.float32)
    offset_arr = mx.array([causal_offset], dtype=mx.int32)

    outputs = _fused_attn_kernel(
        inputs=[
            queries, rotation_matrix, key_packed, centroids,
            key_norms, value_packed, value_norms, scale_arr, offset_arr,
        ],
        grid=(n_q_heads * 32, 1, 1),  # Total threads = n_heads × 32
        threadgroup=(32, 1, 1),
        output_shapes=[(n_q_heads * D,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_q_heads, D)


# ===========================================================================
# HIGH-OCCUPANCY FUSED KERNEL — 32 simdgroups, rotation outside
# ===========================================================================
# Operates in rotated space: query is rotated BEFORE the kernel (MLX GEMM),
# output is rotated back AFTER the kernel (MLX GEMM).
# 32 simdgroups process keys IN PARALLEL with cross-simdgroup online softmax.
# Pattern: Like MLX's sdpa_vector.h — 1024 threads per head.

_FUSED_ATTN_NOROT_HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

_FUSED_ATTN_NOROT_SOURCE = """
    // Grid: total threads = n_q_heads * 1024
    // Threadgroup: (1024, 1, 1) = 32 Simdgroups × 32 Lanes
    // Each threadgroup processes one query head.
    // Simdgroups process keys with stride 32 (parallel).

    uint head = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_id = tid >> 5;   // 0..31 (Simdgroup-Index)
    uint lane_id = tid & 31;   // 0..31 (Lane innerhalb Simdgroup)

    // Dimensions
    uint T_kv = key_norms_shape[1];
    uint n_kv_heads = key_norms_shape[0];
    uint n_q_heads = q_rot_shape[0];
    uint WORDS = key_packed_shape[2];   // D/16 for 2-bit
    uint D = WORDS * 16;
    uint n_repeats = n_q_heads / n_kv_heads;
    uint kv_head = head / n_repeats;

    // Load query (already scaled and rotated)
    float q[4];
    uint q_off = head * D;
    for (uint i = 0; i < 4; i++)
        q[i] = q_rot[q_off + lane_id * 4 + i];

    // KV-Offsets
    uint kv_off_norms = kv_head * T_kv;
    uint kv_off_packed = kv_head * T_kv * WORDS;

    // Dim mapping: lane -> which uint32 word and bit offset
    uint word_idx = lane_id >> 2;        // 0..7
    uint bit_base = (lane_id & 3) << 3;  // 0, 8, 16, 24

    // ===== Scoring + Online Softmax + Value Accumulation (per simdgroup) =====
    float local_max = -1e10f;
    float local_sum = 0.0f;
    float local_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Simdgroups process keys with stride (32 simdgroups in parallel)
    for (uint k = simd_id; k < T_kv; k += 32) {
        // Unpack key centroids for this thread's 4 dims
        uint32_t k_packed = key_packed[kv_off_packed + k * WORDS + word_idx];
        float partial = 0.0f;
        for (uint i = 0; i < 4; i++) {
            uint idx = (k_packed >> (bit_base + i * 2)) & 0x3u;
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
        for (uint i = 0; i < 4; i++) {
            uint v_idx = (v_packed >> (bit_base + i * 2)) & 0x3u;
            float vc = centroids[v_idx] * value_norms[kv_off_norms + k];
            local_acc[i] = local_acc[i] * factor + exp_s * vc;
        }
    }

    // ===== Cross-Simdgroup Reduktion via Threadgroup Memory =====
    // 32 Simdgroups × (max + sum + 128 acc dims)
    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];
    threadgroup float tg_acc[32 * 128];

    // Lane 0 writes max/sum (identical after simd_sum)
    if (lane_id == 0) {
        tg_max[simd_id] = local_max;
        tg_sum[simd_id] = local_sum;
    }
    // All lanes write their 4 acc dims
    for (uint i = 0; i < 4; i++)
        tg_acc[simd_id * 128 + lane_id * 4 + i] = local_acc[i];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each lane computes 4 output dimensions by combining all simdgroups
    float global_max = -1e10f;
    for (uint s = 0; s < 32; s++)
        global_max = max(global_max, tg_max[s]);

    float global_sum = 0.0f;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint s = 0; s < 32; s++) {
        float factor = metal::fast::exp(tg_max[s] - global_max);
        global_sum += tg_sum[s] * factor;
        for (uint i = 0; i < 4; i++)
            result[i] += tg_acc[s * 128 + lane_id * 4 + i] * factor;
    }

    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    for (uint i = 0; i < 4; i++)
        output_rot[head * D + lane_id * 4 + i] = result[i] * inv;
"""

_fused_attn_norot_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_attention_norot",
    input_names=[
        "q_rot", "key_packed", "centroids",
        "key_norms", "value_packed", "value_norms",
    ],
    output_names=["output_rot"],
    source=_FUSED_ATTN_NOROT_SOURCE,
    header=_FUSED_ATTN_NOROT_HEADER,
)


def fused_tq_attention_norot(
    q_rot: mx.array,
    key_packed: mx.array,
    centroids: mx.array,
    key_norms: mx.array,
    value_packed: mx.array,
    value_norms: mx.array,
    n_q_heads: int,
    n_kv_heads: int,
    D: int,
) -> mx.array:
    """High-occupancy fused attention without rotation (32 simdgroups).

    Query must be rotated BEFORE the call, output is in rotated space.

    Args:
        q_rot: (n_q_heads, D) float32 — scaled, rotated queries
        key_packed: (n_kv_heads, T_kv, D//16) uint32 — 2-bit packed
        centroids: (4,) float32
        key_norms: (n_kv_heads, T_kv) float32
        value_packed: (n_kv_heads, T_kv, D//16) uint32
        value_norms: (n_kv_heads, T_kv) float32

    Returns:
        output_rot: (n_q_heads, D) float32 — in rotated space
    """
    outputs = _fused_attn_norot_kernel(
        inputs=[q_rot, key_packed, centroids, key_norms, value_packed, value_norms],
        grid=(n_q_heads * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(n_q_heads * D,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_q_heads, D)
