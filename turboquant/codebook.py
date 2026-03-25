"""Lloyd-Max optimal scalar quantizer for the Gaussian N(0,1) distribution.

Computes optimal centroids and boundaries for b=1,2,3 bits.
The centroids minimize MSE under the normal distribution.
"""

import math

import mlx.core as mx
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def _lloyd_max_gaussian(num_levels: int, max_iter: int = 200, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max iteration for N(0,1).

    Returns (centroids, boundaries) where boundaries are the decision boundaries.
    """
    # Initial boundaries: uniformly distributed over [-3, 3]
    boundaries = np.linspace(-3.0, 3.0, num_levels + 1)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    centroids = np.zeros(num_levels)

    for _ in range(max_iter):
        # Step 1: Compute centroids as conditional expected value
        # c_i = E[X | b_{i-1} < X <= b_i]
        old_centroids = centroids.copy()
        for i in range(num_levels):
            lo, hi = boundaries[i], boundaries[i + 1]

            numerator, _ = quad(lambda x: x * norm.pdf(x), lo, hi)
            denominator, _ = quad(norm.pdf, lo, hi)

            if denominator < 1e-15:
                centroids[i] = (lo + hi) / 2.0 if np.isfinite(lo) and np.isfinite(hi) else old_centroids[i]
                continue
            centroids[i] = numerator / denominator

        # Step 2: Boundaries as midpoint between centroids
        for i in range(1, num_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        if np.max(np.abs(centroids - old_centroids)) < tol:
            break

    # Return inner boundaries (without -inf/+inf)
    inner_boundaries = boundaries[1:-1]
    return centroids, inner_boundaries


# Precomputed codebooks for b=1,2,3,4
_CODEBOOKS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _ensure_codebooks():
    if _CODEBOOKS:
        return
    for bits in (1, 2, 3, 4):
        num_levels = 2**bits
        centroids, boundaries = _lloyd_max_gaussian(num_levels)
        _CODEBOOKS[bits] = (centroids, boundaries)


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

    _ensure_codebooks()
    centroids_np, boundaries_np = _CODEBOOKS[bits]

    scale = 1.0 / math.sqrt(head_dim)
    centroids = mx.array(centroids_np * scale, dtype=mx.float32)
    boundaries = mx.array(boundaries_np * scale, dtype=mx.float32)
    return centroids, boundaries


def get_codebook_unscaled(bits: int) -> tuple[mx.array, mx.array]:
    """Returns (centroids, boundaries) without scaling.

    Useful when scaling is applied separately (e.g. after normalization).
    """
    if bits not in (1, 2, 3):
        raise ValueError(f"Supported bits: 1, 2, 3. Got: {bits}")

    _ensure_codebooks()
    centroids_np, boundaries_np = _CODEBOOKS[bits]
    return mx.array(centroids_np, dtype=mx.float32), mx.array(boundaries_np, dtype=mx.float32)
