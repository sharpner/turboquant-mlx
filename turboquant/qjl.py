"""QJL (Quantization via Johnson-Lindenstrauss) — Pure MLX implementation.

No custom Metal kernels. Uses MLX-native bit operations for sign packing.
"""

import mlx.core as mx

_BITS_32 = mx.arange(32, dtype=mx.uint32)


def pack_sign_bits(values: mx.array) -> mx.array:
    """Packs sign bits into uint32 words. Pure MLX, no custom kernel.

    For each group of 32 values, bit i = 1 if values[i] >= 0, else 0.

    Args:
        values: (..., D) float32 where D is divisible by 32.

    Returns:
        sign_bits: (..., D // 32) uint32
    """
    D = values.shape[-1]
    if D % 32 != 0:
        raise ValueError(f"Last dimension must be divisible by 32, got {D}")

    signs = (values >= 0).astype(mx.uint32)
    reshaped = signs.reshape(*values.shape[:-1], D // 32, 32)
    packed = mx.sum(reshaped << _BITS_32, axis=-1)
    return packed


def unpack_sign_bits(sign_bits: mx.array) -> mx.array:
    """Unpacks uint32 sign bits to float {-1, +1} values.

    Args:
        sign_bits: (..., D // 32) uint32

    Returns:
        signs: (..., D) float32 with values in {-1.0, +1.0}
    """
    expanded = (sign_bits[..., None] >> _BITS_32) & 1
    flat_D = sign_bits.shape[-1] * 32
    result = expanded.reshape(*sign_bits.shape[:-1], flat_D)
    return 2.0 * result.astype(mx.float32) - 1.0


def qjl_encode(
    residual: mx.array,
    jl_matrix: mx.array,
) -> tuple[mx.array, mx.array]:
    """QJL encoding: project residual with JL matrix, store sign bits.

    Args:
        residual: (..., D) float32 — quantization residual
        jl_matrix: (D, D) float32 — JL projection matrix

    Returns:
        sign_bits: (..., D // 32) uint32
        residual_norms: (...,) float32 — ||residual||_2
    """
    residual_norms = mx.linalg.norm(residual, axis=-1)
    projected = residual @ jl_matrix.T
    sign_bits = pack_sign_bits(projected)
    return sign_bits, residual_norms
