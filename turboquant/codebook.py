"""Lloyd-Max optimal scalar quantizer for the Gaussian N(0,1) distribution.

Hardcoded optimal centroids and boundaries for b=1,2,3,4 bits.
These minimize MSE under the normal distribution and are mathematical constants.
"""

import math

import mlx.core as mx


# Lloyd-Max centroids and inner boundaries for N(0,1), computed via
# iterative conditional-expectation (Lloyd's algorithm) with scipy.integrate.quad.
# These are well-known constants — no runtime computation needed.
_CODEBOOKS: dict[int, tuple[list[float], list[float]]] = {
    1: (
        [-0.7978845608028654, 0.7978845608028654],
        [0.0],
    ),
    2: (
        [-1.510417608611893, -0.4527800346911237,
         0.4527800346911237, 1.510417608611893],
        [-0.9815988216515084, 0.0, 0.9815988216515084],
    ),
    3: (
        [-2.151945705166112, -1.3439092791423422,
         -0.7560052816730181, -0.2450941791152904,
         0.2450941791152904, 0.7560052816730181,
         1.3439092791423422, 2.151945705166112],
        [-1.7479274921542272, -1.0499572804076802,
         -0.5005497303941542, 0.0,
         0.5005497303941542, 1.0499572804076802,
         1.7479274921542272],
    ),
    4: (
        [-2.732896755154294, -2.069364258154187,
         -1.618400443227723, -1.2565648452462146,
         -0.9426291036999694, -0.6569817464411519,
         -0.38818871416000605, -0.12844300124876415,
         0.12844300124876415, 0.38818871416000605,
         0.6569817464411519, 0.9426291036999694,
         1.2565648452462146, 1.618400443227723,
         2.069364258154187, 2.732896755154294],
        [-2.4011305066542405, -1.8438823506909552,
         -1.4374826442369688, -1.099596974473092,
         -0.7998054250705606, -0.522585230300579,
         -0.2583158577043851, 0.0,
         0.2583158577043851, 0.522585230300579,
         0.7998054250705606, 1.099596974473092,
         1.4374826442369688, 1.8438823506909552,
         2.4011305066542405],
    ),
}


def get_codebook(bits: int, head_dim: int) -> tuple[mx.array, mx.array]:
    """Returns (centroids, boundaries), scaled by 1/sqrt(head_dim).

    Args:
        bits: Number of bits per coordinate (1, 2, 3, or 4)
        head_dim: Attention head dimension (e.g. 128)

    Returns:
        centroids: mx.array shape (2^bits,) — the optimal centroid values
        boundaries: mx.array shape (2^bits - 1,) — the decision boundaries
    """
    if bits not in (1, 2, 3, 4):
        raise ValueError(f"Supported bits: 1, 2, 3, 4. Got: {bits}")

    centroids_list, boundaries_list = _CODEBOOKS[bits]
    scale = 1.0 / math.sqrt(head_dim)
    centroids = mx.array([c * scale for c in centroids_list], dtype=mx.float32)
    boundaries = mx.array([b * scale for b in boundaries_list], dtype=mx.float32)
    return centroids, boundaries


def get_codebook_unscaled(bits: int) -> tuple[mx.array, mx.array]:
    """Returns (centroids, boundaries) without scaling.

    Useful when scaling is applied separately (e.g. after normalization).
    """
    if bits not in (1, 2, 3, 4):
        raise ValueError(f"Supported bits: 1, 2, 3, 4. Got: {bits}")

    centroids_list, boundaries_list = _CODEBOOKS[bits]
    return mx.array(centroids_list, dtype=mx.float32), mx.array(boundaries_list, dtype=mx.float32)
