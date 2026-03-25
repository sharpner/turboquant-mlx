"""Pure MLX operations for Lloyd-Max codebook quantization.

No custom Metal kernels. All operations use MLX-native bit shifts,
comparisons, and indexing.
"""

import mlx.core as mx

# Pre-computed shift arrays (module-level constants)
_SHIFTS_2BIT = mx.array([i * 2 for i in range(16)], dtype=mx.uint32)
_SHIFTS_3BIT = mx.array([i * 3 for i in range(10)], dtype=mx.uint32)
_SHIFTS_4BIT = mx.array([i * 4 for i in range(8)], dtype=mx.uint32)


def quantize_to_indices(values: mx.array, boundaries: mx.array) -> mx.array:
    """Scalar quantization via cascaded boundary comparison.

    For each value, counts how many boundaries it exceeds.
    Uses accumulation loop to avoid O(N * D * num_boundaries) broadcast tensor.

    Args:
        values: (..., D) float32 — values to quantize
        boundaries: (num_levels - 1,) float32 — sorted decision boundaries

    Returns:
        indices: (..., D) uint8 — centroid indices (0 to 2^bits - 1)
    """
    # Accumulate boundary crossings one at a time.
    # For 1-4 bit (1-15 boundaries), this is 1-15 comparisons
    # without creating the huge (..., D, num_boundaries) intermediate.
    result = mx.zeros(values.shape, dtype=mx.uint8)
    for i in range(boundaries.size):
        result = result + (values > boundaries[i]).astype(mx.uint8)
    return result


def pack_2bit(indices: mx.array) -> mx.array:
    """Packs 2-bit indices into uint32 words (16 per word).

    Args:
        indices: (..., D) uint8 where D is divisible by 16

    Returns:
        packed: (..., D // 16) uint32
    """
    D = indices.shape[-1]
    if D % 16 != 0:
        raise ValueError(f"Last dimension must be divisible by 16, got {D}")

    reshaped = indices.reshape(*indices.shape[:-1], D // 16, 16).astype(mx.uint32)
    return mx.sum((reshaped & 0x3) << _SHIFTS_2BIT, axis=-1)


def unpack_2bit(packed: mx.array, D: int) -> mx.array:
    """Unpacks uint32 words to 2-bit indices.

    Args:
        packed: (..., D // 16) uint32
        D: original dimension

    Returns:
        indices: (..., D) uint32
    """
    expanded = (packed[..., None] >> _SHIFTS_2BIT) & 0x3
    return expanded.reshape(*packed.shape[:-1], D)


def pack_3bit(indices: mx.array) -> mx.array:
    """Packs 3-bit indices into uint32 words (10 per word, 2 bits unused).

    Args:
        indices: (..., D) uint8 where D is divisible by 10

    Returns:
        packed: (..., D // 10) uint32
    """
    D = indices.shape[-1]
    num_words = (D + 9) // 10
    pad_needed = num_words * 10 - D

    if pad_needed > 0:
        indices = mx.pad(indices, [(0, 0)] * (indices.ndim - 1) + [(0, pad_needed)])

    reshaped = indices.reshape(*indices.shape[:-1], num_words, 10).astype(mx.uint32)
    return mx.sum((reshaped & 0x7) << _SHIFTS_3BIT, axis=-1)


def unpack_3bit(packed: mx.array, D: int) -> mx.array:
    """Unpacks uint32 words to 3-bit indices.

    Args:
        packed: (..., D_packed) uint32
        D: original dimension

    Returns:
        indices: (..., D) uint32
    """
    expanded = (packed[..., None] >> _SHIFTS_3BIT) & 0x7
    total = packed.shape[-1] * 10
    flat = expanded.reshape(*packed.shape[:-1], total)
    return flat[..., :D]


def pack_4bit(indices: mx.array) -> mx.array:
    """Packs 4-bit indices into uint32 words (8 per word).

    Args:
        indices: (..., D) uint8 where D is divisible by 8

    Returns:
        packed: (..., D // 8) uint32
    """
    D = indices.shape[-1]
    if D % 8 != 0:
        raise ValueError(f"Last dimension must be divisible by 8, got {D}")

    reshaped = indices.reshape(*indices.shape[:-1], D // 8, 8).astype(mx.uint32)
    return mx.sum((reshaped & 0xF) << _SHIFTS_4BIT, axis=-1)


def unpack_4bit(packed: mx.array, D: int) -> mx.array:
    """Unpacks uint32 words to 4-bit indices.

    Args:
        packed: (..., D // 8) uint32
        D: original dimension

    Returns:
        indices: (..., D) uint32
    """
    expanded = (packed[..., None] >> _SHIFTS_4BIT) & 0xF
    return expanded.reshape(*packed.shape[:-1], D)
