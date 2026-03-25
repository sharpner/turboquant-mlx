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
"""

import mlx.core as mx

from turboquant.codebook import get_codebook
from turboquant.codebook_ops import quantize_to_indices, pack_2bit, pack_3bit, pack_4bit, unpack_2bit, unpack_3bit, unpack_4bit
from turboquant.qjl import qjl_encode
from turboquant.rotation import generate_rotation_matrix, generate_jl_matrix


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

    is_turboquant_v3 = True
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
            # Effective bits per dimension
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
            # Separate codebooks for outlier and regular channels
            self.outlier_centroids, self.outlier_boundaries = get_codebook(self.outlier_bits, head_dim)
            self.regular_centroids, self.regular_boundaries = get_codebook(self.regular_bits, head_dim)
            self.key_outlier_centroids = self.outlier_centroids
            self.key_outlier_boundaries = self.outlier_boundaries
            self.key_regular_centroids, self.key_regular_boundaries = get_codebook(self.key_regular_bits, head_dim)
            mx.eval(self.outlier_centroids, self.outlier_boundaries,
                    self.regular_centroids, self.regular_boundaries,
                    self.key_regular_centroids, self.key_regular_boundaries)
        else:
            # Single codebook
            self.key_centroids, self.key_boundaries = get_codebook(
                self.key_regular_bits if use_qjl else bits, head_dim)
            self.value_centroids, self.value_boundaries = get_codebook(bits, head_dim)
            mx.eval(self.key_centroids, self.key_boundaries,
                    self.value_centroids, self.value_boundaries)
            # Alias for attention
            self.centroids = self.key_centroids

        # --- Rotation matrix ---
        self.rotation_matrix = generate_rotation_matrix(head_dim, seed=seed)
        mx.eval(self.rotation_matrix)

        # --- QJL (keys only) ---
        if use_qjl:
            self.jl_matrix = generate_jl_matrix(head_dim, seed=seed + 95)
            mx.eval(self.jl_matrix)
            self.combined_rot_jl = mx.concatenate(
                [self.rotation_matrix, self.jl_matrix @ self.rotation_matrix], axis=0
            )
            mx.eval(self.combined_rot_jl)

        # --- Storage ---
        self.key_outlier_packed = None
        self.key_regular_packed = None
        self.key_norms = None
        self.value_outlier_packed = None
        self.value_regular_packed = None
        self.value_norms = None
        self.key_sign_bits = None
        self.key_residual_norms = None

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
        reg_key_dim = (self.n_regular + _els_per_word(self.key_regular_bits) - 1) // _els_per_word(self.key_regular_bits) if self.n_regular > 0 else 0
        reg_val_dim = (self.n_regular + _els_per_word(self.regular_bits) - 1) // _els_per_word(self.regular_bits) if self.n_regular > 0 else 0

        if not self.mixed:
            # Uniform mode: regular = full vector
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

    def _quantize_and_pack(self, rotated, is_key=True):
        """Quantize rotated vector and pack indices."""
        if self.mixed:
            # Split into outlier and regular channels
            outlier = rotated[..., :self.n_outlier]
            regular = rotated[..., self.n_outlier:]

            if is_key:
                out_idx = quantize_to_indices(outlier, self.key_outlier_boundaries)
                reg_idx = quantize_to_indices(regular, self.key_regular_boundaries)
                out_packed = _pack(out_idx, self.key_outlier_bits)
                reg_packed = _pack(reg_idx, self.key_regular_bits)
            else:
                out_idx = quantize_to_indices(outlier, self.outlier_boundaries)
                reg_idx = quantize_to_indices(regular, self.regular_boundaries)
                out_packed = _pack(out_idx, self.outlier_bits)
                reg_packed = _pack(reg_idx, self.regular_bits)

            return out_packed, reg_packed, (out_idx, reg_idx)
        else:
            if is_key:
                idx = quantize_to_indices(rotated, self.key_boundaries)
                packed = _pack(idx, self.key_regular_bits)
            else:
                idx = quantize_to_indices(rotated, self.value_boundaries)
                packed = _pack(idx, self.regular_bits)
            return None, packed, (None, idx)

    def _unpack_and_dequant(self, is_key=True):
        """Unpack indices and lookup centroids for full vector."""
        T = self.offset
        if self.mixed:
            if is_key:
                out_packed = self.key_outlier_packed[:, :, :T, :]
                reg_packed = self.key_regular_packed[:, :, :T, :]
                out_idx = _unpack(out_packed, self.n_outlier, self.key_outlier_bits)
                reg_idx = _unpack(reg_packed, self.n_regular, self.key_regular_bits)
                out_vals = self.key_outlier_centroids[out_idx]
                reg_vals = self.key_regular_centroids[reg_idx]
            else:
                out_packed = self.value_outlier_packed[:, :, :T, :]
                reg_packed = self.value_regular_packed[:, :, :T, :]
                out_idx = _unpack(out_packed, self.n_outlier, self.outlier_bits)
                reg_idx = _unpack(reg_packed, self.n_regular, self.regular_bits)
                out_vals = self.outlier_centroids[out_idx]
                reg_vals = self.regular_centroids[reg_idx]
            return mx.concatenate([out_vals, reg_vals], axis=-1)
        else:
            if is_key:
                packed = self.key_regular_packed[:, :, :T, :]
                idx = _unpack(packed, self.head_dim, self.key_regular_bits)
                return self.key_centroids[idx]
            else:
                packed = self.value_regular_packed[:, :, :T, :]
                idx = _unpack(packed, self.head_dim, self.regular_bits)
                return self.value_centroids[idx]

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Quantizes new KV pairs with Lloyd-Max codebook and stores packed."""
        B, n_kv_heads, num_steps, D = keys.shape
        prev = self.offset

        self._ensure_capacity(B, n_kv_heads, num_steps)

        # Normalize
        k_norms = mx.linalg.norm(keys, axis=-1, keepdims=True)
        v_norms = mx.linalg.norm(values, axis=-1, keepdims=True)
        safe_k = mx.where(k_norms < 1e-8, mx.ones_like(k_norms), k_norms)
        safe_v = mx.where(v_norms < 1e-8, mx.ones_like(v_norms), v_norms)

        k_normalized = keys / safe_k
        v_normalized = values / safe_v

        # Rotate
        k_rotated = k_normalized @ self.rotation_matrix.T
        v_rotated = v_normalized @ self.rotation_matrix.T

        # Quantize and pack
        k_out_packed, k_reg_packed, (k_out_idx, k_reg_idx) = self._quantize_and_pack(k_rotated, is_key=True)
        v_out_packed, v_reg_packed, _ = self._quantize_and_pack(v_rotated, is_key=False)

        # QJL on key residual (regular channels only, in rotated space)
        if self.use_qjl:
            if self.mixed:
                # Reconstruct key vector for residual computation
                k_out_recon = self.key_outlier_centroids[k_out_idx.astype(mx.uint32)]
                k_reg_recon = self.key_regular_centroids[k_reg_idx.astype(mx.uint32)]
                k_reconstructed = mx.concatenate([k_out_recon, k_reg_recon], axis=-1)
            else:
                k_reconstructed = self.key_centroids[k_reg_idx.astype(mx.uint32)]
            k_residual = k_rotated - k_reconstructed
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
            if self.key_sign_bits is None:
                self.key_sign_bits = k_sign_bits
                self.key_residual_norms = k_residual_norms
            else:
                self.key_sign_bits = mx.concatenate([self.key_sign_bits, k_sign_bits], axis=2)
                self.key_residual_norms = mx.concatenate([self.key_residual_norms, k_residual_norms], axis=2)

        return keys, values

    def get_key_centroids(self) -> mx.array:
        """Dequantize keys to centroid values."""
        return self._unpack_and_dequant(is_key=True)

    def get_value_centroids(self) -> mx.array:
        """Dequantize values to centroid values."""
        return self._unpack_and_dequant(is_key=False)

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        from mlx_lm.models.base import create_causal_mask
        if N == 1:
            return None
        if return_array or (window_size and N > window_size):
            return create_causal_mask(N, offset=self.offset - N, window_size=window_size)
        return "causal"

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
        if self.use_qjl and self.key_sign_bits is not None:
            total += self.key_sign_bits.nbytes + self.key_residual_norms.nbytes
        return total

    @property
    def nbytes_equivalent_fp16(self):
        if self.key_regular_packed is None:
            return 0
        B, n_kv_heads = self.key_regular_packed.shape[:2]
        T = self.offset
        D = self.head_dim
        return B * n_kv_heads * T * D * 2 * 2
