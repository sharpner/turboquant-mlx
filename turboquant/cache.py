"""TurboQuantKVCache — Drop-in replacement for mlx-lm's KV cache.

Stores keys and values quantized (TurboQuant rotation + QJL for keys).
Indices are 2-bit packed in uint32 (16 per word).

Memory layout per token (head_dim=128, mse_bits=2):
  - Key packed indices: 128/16 * 4 = 32 bytes (uint32, 2-bit packed)
  - Key sign bits:      128/32 * 4 = 16 bytes (uint32)
  - Key norm:           4 bytes (float32)
  - Key residual norm:  4 bytes (float32)
  - Value packed indices: 32 bytes (uint32, 2-bit packed)
  - Value norm:         4 bytes (float32)
  Total: 92 bytes vs 512 bytes (float16) = 5.6x compression
"""

import mlx.core as mx

from turboquant.codebook import get_codebook
from turboquant.rotation import generate_rotation_matrix, generate_jl_matrix, build_combined_rot_jl
from turboquant.kernels import (
    turboquant_encode,
    qjl_encode,
    pack_2bit_indices,
    unpack_2bit_indices,
    pack_3bit_indices,
    unpack_3bit_indices,
)


def make_causal_mask(offset: int, N: int, return_array: bool = False, window_size=None):
    """Creates attention mask compatible with mlx-lm.

    Args:
        offset: Current cache offset (total tokens seen)
        N: Number of new query tokens
        return_array: Force array mask instead of string
        window_size: Sliding window attention size

    Returns:
        None (single token), "causal" (string), or mx.array mask
    """
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        from mlx_lm.models.base import create_causal_mask
        return create_causal_mask(N, offset=offset - N, window_size=window_size)
    return "causal"


class TurboQuantKVCache:
    """TurboQuant KV cache with random QR rotation + QJL compression.

    Compatible with mlx-lm's cache interface (update_and_fetch, offset, etc.).
    Indices are 2-bit packed in uint32 for maximum compression.
    """

    def __init__(self, head_dim: int = 128, mse_bits: int = 2, use_qjl: bool = True, seed: int = 42):
        self.head_dim = head_dim
        self.mse_bits = mse_bits
        self.use_qjl = use_qjl
        self.offset = 0

        # Codebook (scaled for normalized vectors)
        self.centroids, self.boundaries = get_codebook(mse_bits, head_dim)

        # Rotation + JL matrices (compute once)
        self.rotation_matrix = generate_rotation_matrix(head_dim, seed=seed)
        self.jl_matrix = generate_jl_matrix(head_dim, seed=seed + 95)

        # Precompute: Combined matrix for rotation + JL sketch in one matmul
        self.combined_rot_jl = build_combined_rot_jl(self.rotation_matrix, self.jl_matrix)

        # Quantized storage (packed) — initialized on first update_and_fetch
        self.key_packed = None      # uint32, 2-bit packed indices
        self.key_norms = None       # float32, L2 norms
        self.key_sign_bits = None   # uint32, QJL sign bits
        self.key_residual_norms = None  # float32, gamma
        self.value_packed = None    # uint32, 2-bit packed indices
        self.value_norms = None     # float32, L2 norms

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Quantizes new KV pairs, stores packed, returns dequantized.

        Args:
            keys: (B, n_kv_heads, T_new, D) — new keys (after RoPE)
            values: (B, n_kv_heads, T_new, D) — new values

        Returns:
            (all_keys, all_values): Dequantized tensors (for standard SDPA fallback)
        """
        # --- Quantize keys ---
        k_indices, k_norms, k_rotated = turboquant_encode(
            keys, self.rotation_matrix, self.boundaries
        )

        # Residual for QJL (optional)
        if self.use_qjl:
            k_reconstructed_rotated = self.centroids[k_indices.astype(mx.uint32)]
            k_residual = k_rotated - k_reconstructed_rotated
            k_sign_bits, k_residual_norms = qjl_encode(k_residual, self.jl_matrix)
        else:
            k_sign_bits = None
            k_residual_norms = None

        # Pack indices
        if self.mse_bits == 2:
            k_packed = pack_2bit_indices(k_indices)
        else:
            k_packed = pack_3bit_indices(k_indices)

        # --- Quantize values (MSE only, no QJL) ---
        v_indices, v_norms, _ = turboquant_encode(
            values, self.rotation_matrix, self.boundaries
        )
        if self.mse_bits == 2:
            v_packed = pack_2bit_indices(v_indices)
        else:
            v_packed = pack_3bit_indices(v_indices)

        # --- Append to cache ---
        if self.key_packed is None:
            self.key_packed = k_packed
            self.key_norms = k_norms
            self.key_sign_bits = k_sign_bits
            self.key_residual_norms = k_residual_norms
            self.value_packed = v_packed
            self.value_norms = v_norms
        else:
            self.key_packed = mx.concatenate([self.key_packed, k_packed], axis=2)
            self.key_norms = mx.concatenate([self.key_norms, k_norms], axis=2)
            if self.use_qjl:
                self.key_sign_bits = mx.concatenate(
                    [self.key_sign_bits, k_sign_bits], axis=2
                )
                self.key_residual_norms = mx.concatenate(
                    [self.key_residual_norms, k_residual_norms], axis=2
                )
            self.value_packed = mx.concatenate(
                [self.value_packed, v_packed], axis=2
            )
            self.value_norms = mx.concatenate([self.value_norms, v_norms], axis=2)

        self.offset += keys.shape[2]

        # mx.eval required: Metal barrier between concat and custom kernel
        # only guarantees threadgroup-level ordering, not memory-write
        # completion. Without eval the fused kernel reads stale GPU buffers.
        # -> V2 (mx.quantized_matmul) does not have this problem.
        state = [self.key_packed, self.key_norms, self.value_packed, self.value_norms]
        if self.use_qjl and self.key_sign_bits is not None:
            state.extend([self.key_sign_bits, self.key_residual_norms])
        mx.eval(*state)

        return keys, values

    def get_key_indices(self) -> mx.array:
        """Unpacks key indices. Returns (..., D) uint32."""
        if self.mse_bits == 2:
            return unpack_2bit_indices(self.key_packed, self.head_dim)
        return unpack_3bit_indices(self.key_packed, self.head_dim)

    def get_value_indices(self) -> mx.array:
        """Unpacks value indices. Returns (..., D) uint32."""
        if self.mse_bits == 2:
            return unpack_2bit_indices(self.value_packed, self.head_dim)
        return unpack_3bit_indices(self.value_packed, self.head_dim)

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        return make_causal_mask(self.offset, N, return_array, window_size)

    @property
    def state(self):
        """Cache state for mx.eval() compatibility with mlx-lm."""
        if self.key_packed is None:
            return []
        parts = [self.key_packed, self.key_norms, self.value_packed, self.value_norms]
        if self.use_qjl and self.key_sign_bits is not None:
            parts.extend([self.key_sign_bits, self.key_residual_norms])
        return parts

    @state.setter
    def state(self, v):
        if not v:
            return
        self.key_packed = v[0]
        self.key_norms = v[1]
        self.value_packed = v[2]
        self.value_norms = v[3]
        if self.use_qjl and len(v) > 4:
            self.key_sign_bits = v[4]
            self.key_residual_norms = v[5]

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass

    def is_trimmable(self):
        return False

    def empty(self):
        return self.key_packed is None

    @property
    def nbytes(self):
        if self.key_packed is None:
            return 0
        total = self.key_packed.nbytes + self.key_norms.nbytes
        total += self.value_packed.nbytes + self.value_norms.nbytes
        if self.use_qjl and self.key_sign_bits is not None:
            total += self.key_sign_bits.nbytes + self.key_residual_norms.nbytes
        return total

    @property
    def nbytes_equivalent_fp16(self):
        """Equivalent memory if everything were in float16."""
        if self.key_packed is None:
            return 0
        B, n_kv_heads, T, _ = self.key_packed.shape
        D = self.head_dim
        return B * n_kv_heads * T * D * 2 * 2  # 2 bytes fp16 * 2 (keys + values)
