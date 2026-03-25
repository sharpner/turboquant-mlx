"""Monkey-patch for mlx-lm's SDPA dispatch.

Supports TurboQuant V1 (fused Metal kernel), V2 (mx.quantized_matmul),
V3 (Lloyd-Max codebook with software dequant).
"""

import mlx.core as mx
import mlx_lm.models.base as _base

from turboquant.attention_fused import turboquant_fused_sdpa
from turboquant.attention_v2 import turboquant_v2_sdpa
from turboquant.attention_v3 import turboquant_v3_sdpa
from turboquant.cache import TurboQuantKVCache
from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v3 import TurboQuantKVCacheV3

_original_sdpa = _base.scaled_dot_product_attention
_patched = False


def _patched_sdpa(queries, keys, values, cache, scale, mask, **kwargs):
    if isinstance(cache, TurboQuantKVCacheV3):
        return turboquant_v3_sdpa(queries, cache, scale, mask)
    if isinstance(cache, TurboQuantKVCacheV2):
        return turboquant_v2_sdpa(queries, keys, values, cache, scale, mask)
    if isinstance(cache, TurboQuantKVCache):
        return turboquant_fused_sdpa(queries, cache, scale, mask)
    return _original_sdpa(queries, keys, values, cache, scale, mask, **kwargs)


def apply():
    """Activates the TurboQuant SDPA patch. Idempotent."""
    global _patched
    if _patched:
        return
    _base.scaled_dot_product_attention = _patched_sdpa
    _patched = True


def revert():
    """Removes the patch."""
    global _patched
    _base.scaled_dot_product_attention = _original_sdpa
    _patched = False
