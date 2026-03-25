"""Random rotation matrix (TurboQuant) and JL matrix (QJL).

The rotation transforms arbitrary vectors into a near-uniform
distribution via QR decomposition, which optimizes scalar quantization.
The JL matrix is used for 1-bit residual compression.
"""

import mlx.core as mx


def generate_rotation_matrix(head_dim: int, seed: int = 42) -> mx.array:
    """Generates an orthogonal rotation matrix Pi via QR decomposition.

    Args:
        head_dim: Dimension (e.g. 128)
        seed: Seed for reproducibility

    Returns:
        Pi: mx.array shape (head_dim, head_dim), orthogonal, float32
    """
    mx.random.seed(seed)
    G = mx.random.normal((head_dim, head_dim))
    mx.eval(G)

    # QR only possible on CPU
    Q, R = mx.linalg.qr(G, stream=mx.cpu)
    mx.eval(Q)

    # Sign correction: ensure det(Q) = +1
    # (QR can have negative diagonal in R -> Q reflects instead of rotates)
    diag_sign = mx.sign(mx.diag(R))
    Q = Q * diag_sign[None, :]
    mx.eval(Q)

    return Q


def generate_jl_matrix(head_dim: int, seed: int = 137) -> mx.array:
    """Generates the Johnson-Lindenstrauss projection matrix S.

    S has i.i.d. N(0,1) entries. No orthogonalization required —
    the JL property holds for arbitrary Gaussian matrices.

    Args:
        head_dim: Dimension (e.g. 128)
        seed: Seed (different default than rotation to ensure independence)

    Returns:
        S: mx.array shape (head_dim, head_dim), float32
    """
    mx.random.seed(seed)
    S = mx.random.normal((head_dim, head_dim))
    mx.eval(S)
    return S
