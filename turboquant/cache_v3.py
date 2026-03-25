"""TurboQuantKVCache V3 — Lloyd-Max codebook quantization (paper-correct).

Implements the actual TurboQuant algorithm from the paper:
  - TurboQuant_mse: Random rotation + Lloyd-Max scalar quantization
  - TurboQuant_prod: Keys at (b-1)-bit MSE + QJL, values at b-bit MSE
  - Outlier channel splitting: mixed bit allocation across channels
    e.g. 2.5-bit = 32 channels @ 3-bit + 96 channels @ 2-bit

After random rotation, all channels are ~iid N(0, 1/sqrt(D)), so a fixed
channel split works as well as dynamic outlier detection.

Uses pure MLX operations — no custom Metal kernels, no mx.quantized_matmul.
Pre-allocation with step=256 for minimal allocation overhead.

Performance: Dequantized centroids are cached incrementally. Only newly
added slices are dequantized — get_key_centroids() / get_value_centroids()
return pre-computed buffers directly.
"""

import mlx.core as mx

from turboquant.cache import make_causal_mask
from turboquant.codebook import get_codebook
from turboquant.codebook_ops import quantize_to_indices, pack_2bit, pack_3bit, pack_4bit, unpack_2bit, unpack_3bit, unpack_4bit
from turboquant.qjl import qjl_encode
from turboquant.rotation import (
    generate_rotation_matrix,
    generate_jl_matrix,
    build_combined_rot_jl,
    safe_normalize,
)


def _pack(indices: mx.array, bits: int) -> mx.array:
    if bits <= 2:
        return pack_2bit(indices)
    if bits == 3:
        return pack_3bit(indices)
    return pack_4bit(indices)


def _unpack(packed: mx.array, D: int, bits: int) -> mx.array:
    if bits <= 2:
        return unpack_2bit(packed, D)
    if bits == 3:
        return unpack_3bit(packed, D)
    return unpack_4bit(packed, D)


def _els_per_word(bits: int) -> int:
    if bits <= 2:
        return 16
    if bits == 3:
        return 10
    return 8


class TurboQuantKVCacheV3:
    """TurboQuant V3 — Lloyd-Max codebook with optional QJL and channel splitting.

    Modes:
      - Uniform: all channels at same bit width
      - Mixed (n_outlier > 0): first n_outlier channels at outlier_bits,
        rest at base bits. After rotation all channels are equivalent,
        so fixed split is as good as dynamic outlier detection.
      - QJL (use_qjl=True): keys at (b-1)-bit MSE + 1-bit QJL,
        values at b-bit MSE.
    """

    step = 256

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 2,
        use_qjl: bool = False,
        n_outlier: int = 0,
        outlier_bits: int = 3,
        seed: int = 42,
    ):
        if n_outlier > head_dim:
            raise ValueError(f"n_outlier ({n_outlier}) must be <= head_dim ({head_dim})")

        self.head_dim = head_dim
        self.bits = bits
        self.use_qjl = use_qjl
        self.offset = 0

        # --- Channel splitting ---
        self.n_outlier = n_outlier
        self.n_regular = head_dim - n_outlier
        self.mixed = n_outlier > 0

        if self.mixed:
            self.outlier_bits = outlier_bits
            self.regular_bits = bits
            self.effective_bits = (n_outlier * outlier_bits + self.n_regular * bits) / head_dim
        else:
            self.outlier_bits = bits
            self.regular_bits = bits
            self.effective_bits = float(bits)

        # --- Key MSE bits (QJL reduces by 1) ---
        self.key_regular_bits = self.regular_bits - 1 if use_qjl else self.regular_bits
        self.key_outlier_bits = self.outlier_bits  # outlier channels always at full bits

        if self.key_regular_bits < 1:
            raise ValueError(f"Need at least 1 MSE bit for keys. Got bits={bits}, use_qjl={use_qjl}")

        # --- Codebooks ---
        if self.mixed:
            self.outlier_centroids, self.outlier_boundaries = get_codebook(self.outlier_bits, head_dim)
            self.regular_centroids, self.regular_boundaries = get_codebook(self.regular_bits, head_dim)
            self.key_regular_centroids, self.key_regular_boundaries = get_codebook(self.key_regular_bits, head_dim)
            mx.eval(self.outlier_centroids, self.outlier_boundaries,
                    self.regular_centroids, self.regular_boundaries,
                    self.key_regular_centroids, self.key_regular_boundaries)
        else:
            self.key_centroids, self.key_boundaries = get_codebook(
                self.key_regular_bits if use_qjl else bits, head_dim)
            self.value_centroids, self.value_boundaries = get_codebook(bits, head_dim)
            mx.eval(self.key_centroids, self.key_boundaries,
                    self.value_centroids, self.value_boundaries)

        # --- Rotation matrix ---
        self.rotation_matrix = generate_rotation_matrix(head_dim, seed=seed)
        mx.eval(self.rotation_matrix)

        # --- QJL (keys only) ---
        if use_qjl:
            self.jl_matrix = generate_jl_matrix(head_dim, seed=seed + 95)
            mx.eval(self.jl_matrix)
            self.combined_rot_jl = build_combined_rot_jl(self.rotation_matrix, self.jl_matrix)

        # --- Storage (pre-allocated buffers, access via offset slicing) ---
        self.key_outlier_packed = None
        self.key_regular_packed = None
        self.key_norms = None
        self.value_outlier_packed = None
        self.value_regular_packed = None
        self.value_norms = None
        self._key_sign_bits_buf = None
        self._key_residual_norms_buf = None

        # --- Dequantized centroid caches (P1: incremental) ---
        self._key_centroids_cache = None
        self._value_centroids_cache = None

    @property
    def key_sign_bits(self):
        """Returns QJL sign bits sliced to valid offset."""
        if self._key_sign_bits_buf is None:
            return None
        return self._key_sign_bits_buf[:, :, :self.offset, :]

    @property
    def key_residual_norms(self):
        """Returns QJL residual norms sliced to valid offset."""
        if self._key_residual_norms_buf is None:
            return None
        return self._key_residual_norms_buf[:, :, :self.offset]

    def _ensure_capacity(self, B, n_kv_heads, num_steps):
        """Pre-allocate or expand buffers."""
        prev = self.offset

        if self.key_regular_packed is not None and (prev + num_steps) <= self.key_regular_packed.shape[2]:
            return

        new_steps = (self.step + num_steps - 1) // self.step * self.step

        def _alloc_or_grow(existing, shape):
            if existing is not None:
                old = existing if prev % self.step == 0 else existing[:, :, :prev, :]
                return mx.concatenate([old, mx.zeros(shape, dtype=mx.uint32)], axis=2)
            return mx.zeros(shape, dtype=mx.uint32)

        def _alloc_or_grow_1d(existing, shape):
            if existing is not None:
                old = existing if prev % self.step == 0 else existing[:, :, :prev]
                return mx.concatenate([old, mx.zeros(shape, dtype=mx.float32)], axis=2)
            return mx.zeros(shape, dtype=mx.float32)

        # Regular channels
        if self.mixed:
            reg_key_dim = (self.n_regular + _els_per_word(self.key_regular_bits) - 1) // _els_per_word(self.key_regular_bits) if self.n_regular > 0 else 0
            reg_val_dim = (self.n_regular + _els_per_word(self.regular_bits) - 1) // _els_per_word(self.regular_bits) if self.n_regular > 0 else 0
        else:
            reg_key_dim = (self.head_dim + _els_per_word(self.key_regular_bits) - 1) // _els_per_word(self.key_regular_bits)
            reg_val_dim = (self.head_dim + _els_per_word(self.regular_bits) - 1) // _els_per_word(self.regular_bits)

        self.key_regular_packed = _alloc_or_grow(self.key_regular_packed, (B, n_kv_heads, new_steps, reg_key_dim))
        self.value_regular_packed = _alloc_or_grow(self.value_regular_packed, (B, n_kv_heads, new_steps, reg_val_dim))

        # Outlier channels
        if self.mixed:
            out_dim = (self.n_outlier + _els_per_word(self.outlier_bits) - 1) // _els_per_word(self.outlier_bits)
            self.key_outlier_packed = _alloc_or_grow(self.key_outlier_packed, (B, n_kv_heads, new_steps, out_dim))
            self.value_outlier_packed = _alloc_or_grow(self.value_outlier_packed, (B, n_kv_heads, new_steps, out_dim))

        # Norms
        self.key_norms = _alloc_or_grow_1d(self.key_norms, (B, n_kv_heads, new_steps))
        self.value_norms = _alloc_or_grow_1d(self.value_norms, (B, n_kv_heads, new_steps))

        # QJL pre-allocated buffers
        if self.use_qjl:
            n_proj_words = self.head_dim // 32
            if self._key_sign_bits_buf is not None:
                sb_old = self._key_sign_bits_buf if prev % self.step == 0 else self._key_sign_bits_buf[:, :, :prev, :]
                rn_old = self._key_residual_norms_buf if prev % self.step == 0 else self._key_residual_norms_buf[:, :, :prev]
                self._key_sign_bits_buf = mx.concatenate([sb_old, mx.zeros((B, n_kv_heads, new_steps, n_proj_words), dtype=mx.uint32)], axis=2)
                self._key_residual_norms_buf = mx.concatenate([rn_old, mx.zeros((B, n_kv_heads, new_steps), dtype=mx.float32)], axis=2)
            else:
                total_steps = new_steps
                self._key_sign_bits_buf = mx.zeros((B, n_kv_heads, total_steps, n_proj_words), dtype=mx.uint32)
                self._key_residual_norms_buf = mx.zeros((B, n_kv_heads, total_steps), dtype=mx.float32)

        # Dequant cache buffers
        if self._key_centroids_cache is not None:
            kc_old = self._key_centroids_cache if prev % self.step == 0 else self._key_centroids_cache[:, :, :prev, :]
            vc_old = self._value_centroids_cache if prev % self.step == 0 else self._value_centroids_cache[:, :, :prev, :]
            self._key_centroids_cache = mx.concatenate([kc_old, mx.zeros((B, n_kv_heads, new_steps, self.head_dim), dtype=mx.float32)], axis=2)
            self._value_centroids_cache = mx.concatenate([vc_old, mx.zeros((B, n_kv_heads, new_steps, self.head_dim), dtype=mx.float32)], axis=2)
        else:
            total_steps = new_steps
            if self.key_regular_packed is not None:
                total_steps = self.key_regular_packed.shape[2]
            self._key_centroids_cache = mx.zeros((B, n_kv_heads, total_steps, self.head_dim), dtype=mx.float32)
            self._value_centroids_cache = mx.zeros((B, n_kv_heads, total_steps, self.head_dim), dtype=mx.float32)

    def _dequant_slice(self, indices_or_packed, is_key, is_outlier=False):
        """Dequantize a slice of indices to centroid values.

        Args:
            indices_or_packed: Already-unpacked indices (uint8/uint32)
            is_key: True for keys, False for values
            is_outlier: True for outlier channel indices

        Returns:
            Centroid values looked up from the appropriate codebook
        """
        if self.mixed:
            if is_key:
                if is_outlier:
                    return self.outlier_centroids[indices_or_packed]
                return self.key_regular_centroids[indices_or_packed]
            if is_outlier:
                return self.outlier_centroids[indices_or_packed]
            return self.regular_centroids[indices_or_packed]

        if is_key:
            return self.key_centroids[indices_or_packed]
        return self.value_centroids[indices_or_packed]

    def _quantize_and_pack(self, rotated, is_key=True):
        """Quantize rotated vector and pack indices. Returns (out_packed, reg_packed, centroid_values)."""
        if self.mixed:
            outlier = rotated[..., :self.n_outlier]
            regular = rotated[..., self.n_outlier:]

            if is_key:
                out_idx = quantize_to_indices(outlier, self.outlier_boundaries)
                reg_idx = quantize_to_indices(regular, self.key_regular_boundaries)
                out_packed = _pack(out_idx, self.key_outlier_bits)
                reg_packed = _pack(reg_idx, self.key_regular_bits)
                out_vals = self.outlier_centroids[out_idx.astype(mx.uint32)]
                reg_vals = self.key_regular_centroids[reg_idx.astype(mx.uint32)]
            else:
                out_idx = quantize_to_indices(outlier, self.outlier_boundaries)
                reg_idx = quantize_to_indices(regular, self.regular_boundaries)
                out_packed = _pack(out_idx, self.outlier_bits)
                reg_packed = _pack(reg_idx, self.regular_bits)
                out_vals = self.outlier_centroids[out_idx.astype(mx.uint32)]
                reg_vals = self.regular_centroids[reg_idx.astype(mx.uint32)]

            centroid_vals = mx.concatenate([out_vals, reg_vals], axis=-1)
            return out_packed, reg_packed, centroid_vals

        if is_key:
            idx = quantize_to_indices(rotated, self.key_boundaries)
            packed = _pack(idx, self.key_regular_bits)
            centroid_vals = self.key_centroids[idx.astype(mx.uint32)]
        else:
            idx = quantize_to_indices(rotated, self.value_boundaries)
            packed = _pack(idx, self.regular_bits)
            centroid_vals = self.value_centroids[idx.astype(mx.uint32)]

        return None, packed, centroid_vals

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Quantizes new KV pairs with Lloyd-Max codebook and stores packed."""
        B, n_kv_heads, num_steps, D = keys.shape
        prev = self.offset

        self._ensure_capacity(B, n_kv_heads, num_steps)

        # Normalize
        k_normalized, k_norms = safe_normalize(keys)
        v_normalized, v_norms = safe_normalize(values)

        # Rotate
        k_rotated = k_normalized @ self.rotation_matrix.T
        v_rotated = v_normalized @ self.rotation_matrix.T

        # Quantize, pack, and get centroid values in one pass
        k_out_packed, k_reg_packed, k_centroid_vals = self._quantize_and_pack(k_rotated, is_key=True)
        v_out_packed, v_reg_packed, v_centroid_vals = self._quantize_and_pack(v_rotated, is_key=False)

        # QJL on key residual
        if self.use_qjl:
            k_residual = k_rotated - k_centroid_vals
            k_sign_bits, k_residual_norms = qjl_encode(k_residual, self.jl_matrix)

        # Store
        self.offset += num_steps
        self.key_regular_packed[:, :, prev:self.offset, :] = k_reg_packed
        self.value_regular_packed[:, :, prev:self.offset, :] = v_reg_packed
        self.key_norms[:, :, prev:self.offset] = k_norms.squeeze(-1)
        self.value_norms[:, :, prev:self.offset] = v_norms.squeeze(-1)

        if self.mixed:
            self.key_outlier_packed[:, :, prev:self.offset, :] = k_out_packed
            self.value_outlier_packed[:, :, prev:self.offset, :] = v_out_packed

        if self.use_qjl:
            self._key_sign_bits_buf[:, :, prev:self.offset, :] = k_sign_bits
            self._key_residual_norms_buf[:, :, prev:self.offset] = k_residual_norms

        # Incrementally update dequant caches (P1)
        self._key_centroids_cache[:, :, prev:self.offset, :] = k_centroid_vals
        self._value_centroids_cache[:, :, prev:self.offset, :] = v_centroid_vals

        return keys, values

    def get_key_centroids(self) -> mx.array:
        """Returns cached dequantized key centroids. O(1) — no re-dequantization."""
        return self._key_centroids_cache[:, :, :self.offset, :]

    def get_value_centroids(self) -> mx.array:
        """Returns cached dequantized value centroids. O(1) — no re-dequantization."""
        return self._value_centroids_cache[:, :, :self.offset, :]

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        return make_causal_mask(self.offset, N, return_array, window_size)

    @property
    def state(self):
        if self.key_regular_packed is None:
            return []
        parts = [
            self.key_regular_packed[:, :, :self.offset, :],
            self.value_regular_packed[:, :, :self.offset, :],
            self.key_norms[:, :, :self.offset],
            self.value_norms[:, :, :self.offset],
        ]
        if self.mixed:
            parts += [
                self.key_outlier_packed[:, :, :self.offset, :],
                self.value_outlier_packed[:, :, :self.offset, :],
            ]
        if self.use_qjl and self.key_sign_bits is not None:
            parts += [self.key_sign_bits, self.key_residual_norms]
        # Include dequant caches
        parts += [
            self._key_centroids_cache[:, :, :self.offset, :],
            self._value_centroids_cache[:, :, :self.offset, :],
        ]
        return parts

    @state.setter
    def state(self, v):
        pass

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def empty(self):
        return self.key_regular_packed is None

    @property
    def nbytes(self):
        if self.key_regular_packed is None:
            return 0
        T = self.offset
        B, n_kv_heads = self.key_regular_packed.shape[:2]
        # Regular packed indices
        total = B * n_kv_heads * T * self.key_regular_packed.shape[-1] * 4
        total += B * n_kv_heads * T * self.value_regular_packed.shape[-1] * 4
        # Outlier packed indices
        if self.mixed and self.key_outlier_packed is not None:
            total += B * n_kv_heads * T * self.key_outlier_packed.shape[-1] * 4
            total += B * n_kv_heads * T * self.value_outlier_packed.shape[-1] * 4
        # Norms
        total += 2 * B * n_kv_heads * T * 4
        # QJL
        if self.use_qjl and self._key_sign_bits_buf is not None:
            total += B * n_kv_heads * T * self._key_sign_bits_buf.shape[-1] * 4
            total += B * n_kv_heads * T * 4
        # NOTE: dequant caches (_key_centroids_cache, _value_centroids_cache) are
        # a speed optimization, not compressed storage. Excluded from nbytes.
        return total

    @property
    def nbytes_equivalent_fp16(self):
        if self.key_regular_packed is None:
            return 0
        B, n_kv_heads = self.key_regular_packed.shape[:2]
        T = self.offset
        D = self.head_dim
        return B * n_kv_heads * T * D * 2 * 2
