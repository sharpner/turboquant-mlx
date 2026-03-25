"""Shared utilities for TurboQuant benchmarks and experiments.

Provides: EVAL_TEXT, compute_perplexity, cache_nbytes, make_cache.
"""

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache

from turboquant.cache import TurboQuantKVCache
from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v3 import TurboQuantKVCacheV3

EVAL_TEXT = (
    "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
    "artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of "
    "modern AI were planted by philosophers who attempted to describe the process of human thinking "
    "as the mechanical manipulation of symbols. This work culminated in the invention of the "
    "programmable digital computer in the 1940s, a machine based on the abstract essence of "
    "mathematical reasoning. This device and the ideas behind it inspired a handful of scientists "
    "to begin seriously discussing the possibility of building an electronic brain. The field of AI "
    "research was founded at a workshop held on the campus of Dartmouth College during the summer "
    "of 1956. Those who attended would become the leaders of AI research for decades. Many of them "
    "predicted that a machine as intelligent as a human being would exist in no more than a "
    "generation, and they were given millions of dollars to make this vision come true. Eventually, "
    "it became obvious that commercial developers and researchers had grossly underestimated the "
    "difficulty of the project. In 1974, in response to the criticism from James Lighthill and "
    "ongoing pressure from congress, the U.S. and British governments cut off exploratory research "
    "in AI. The next few years would later be called an AI winter, a period when obtaining funding "
    "for AI projects was difficult."
)


def compute_perplexity(model, tokenizer, text, cache):
    """Computes perplexity on an evaluation text."""
    input_ids = mx.array(tokenizer.encode(text))[None]  # (1, T)
    T = input_ids.shape[1]

    if T < 2:
        return float("inf")

    logits = model(input_ids, cache=cache)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)
    token_log_probs = mx.take_along_axis(
        log_probs, shift_labels[:, :, None], axis=-1
    ).squeeze(-1)

    avg_nll = -mx.mean(token_log_probs).item()
    return float(mx.exp(mx.array(avg_nll)).item())


def cache_nbytes(cache_layer) -> int:
    """Computes cache memory in bytes for a single layer.

    Workaround for mlx-lm bug where QuantizedKVCache.nbytes crashes
    due to missing tree_reduce import.
    """
    if isinstance(cache_layer, (TurboQuantKVCache, TurboQuantKVCacheV2, TurboQuantKVCacheV3)):
        return cache_layer.nbytes
    if isinstance(cache_layer, KVCache):
        if cache_layer.keys is None:
            return 0
        return cache_layer.keys.nbytes + cache_layer.values.nbytes
    if isinstance(cache_layer, QuantizedKVCache):
        if cache_layer.keys is None:
            return 0
        total = 0
        for tensor in (*cache_layer.keys, *cache_layer.values):
            total += tensor.nbytes
        return total
    return 0


# ---------------------------------------------------------------------------
# Strategy registry for make_cache
# ---------------------------------------------------------------------------
# Each entry maps strategy_name -> lambda(n_layers, head_dim) -> list[cache]
# "fp16" is handled specially via make_prompt_cache(model).

_STRATEGIES = {
    # --- MLX built-in ---
    "quant4": lambda nl, hd: [
        QuantizedKVCache(group_size=64, bits=4) for _ in range(nl)
    ],
    "quant8": lambda nl, hd: [
        QuantizedKVCache(group_size=64, bits=8) for _ in range(nl)
    ],

    # --- V1: TurboQuantKVCache (custom codebook path) ---
    "turboquant2": lambda nl, hd: [
        TurboQuantKVCache(head_dim=hd, mse_bits=2, seed=42 + i)
        for i in range(nl)
    ],
    "turboquant3": lambda nl, hd: [
        TurboQuantKVCache(head_dim=hd, mse_bits=3, use_qjl=True, seed=42 + i)
        for i in range(nl)
    ],
    "turboquant3_noqjl": lambda nl, hd: [
        TurboQuantKVCache(head_dim=hd, mse_bits=3, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "tq_fused_2bit": lambda nl, hd: [
        TurboQuantKVCache(head_dim=hd, mse_bits=2, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],

    # --- V2: MLX-native quantized_matmul ---
    "tqv2_2bit": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=2, group_size=64, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "tqv2_3bit": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=3, group_size=64, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "tqv2_4bit": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=4, group_size=64, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "tqv2_3bit_norot": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=3, group_size=64, use_qjl=False, use_rotation=False, seed=42 + i)
        for i in range(nl)
    ],
    "tqv2_4bit_norot": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=4, group_size=64, use_qjl=False, use_rotation=False, seed=42 + i)
        for i in range(nl)
    ],
    "tqv2_4bit_lean": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=4, group_size=64, use_qjl=False, use_rotation=False, use_normalization=False, seed=42 + i)
        for i in range(nl)
    ],
    "tqv2_3bit_lean": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=3, group_size=64, use_qjl=False, use_rotation=False, use_normalization=False, seed=42 + i)
        for i in range(nl)
    ],
    "tqv2_3bit_rot_qjl": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=3, group_size=64, use_rotation=True, use_normalization=True, use_qjl=True, seed=42 + i)
        for i in range(nl)
    ],

    # --- V2 from benchmark_models (different naming convention) ---
    "v2_4bit_lean": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=4, group_size=64, use_rotation=False, use_normalization=False, seed=42 + i)
        for i in range(nl)
    ],
    "v2_4bit_rot": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=4, group_size=64, use_rotation=True, use_normalization=True, seed=42 + i)
        for i in range(nl)
    ],
    "v2_3bit_lean": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=3, group_size=64, use_rotation=False, use_normalization=False, seed=42 + i)
        for i in range(nl)
    ],
    "v2_3bit_rot": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=3, group_size=64, use_rotation=True, use_normalization=True, use_qjl=True, seed=42 + i)
        for i in range(nl)
    ],
    "v2_2bit_rot": lambda nl, hd: [
        TurboQuantKVCacheV2(head_dim=hd, bits=2, group_size=32, use_rotation=True, use_normalization=True, seed=42 + i)
        for i in range(nl)
    ],

    # --- V3: Lloyd-Max codebook (paper-correct) ---
    "tqv3_2bit": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=2, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "tqv3_2bit_prod": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=2, use_qjl=True, seed=42 + i)
        for i in range(nl)
    ],
    "tqv3_3bit": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=3, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "tqv3_3bit_prod": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=3, use_qjl=True, seed=42 + i)
        for i in range(nl)
    ],

    # --- V3 from benchmark_models (different naming convention) ---
    "v3_2bit": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=2, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "v3_2bit_prod": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=2, use_qjl=True, seed=42 + i)
        for i in range(nl)
    ],
    "v3_3bit": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=3, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "v3_3bit_prod": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=3, use_qjl=True, seed=42 + i)
        for i in range(nl)
    ],

    # --- V3: Mixed-bit strategies ---
    "tqv3_3.5bit": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=3, n_outlier=hd // 2, outlier_bits=4, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "tqv3_2.5bit": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=2, n_outlier=hd // 2, outlier_bits=3, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "v3_3.5bit": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=3, n_outlier=hd // 2, outlier_bits=4, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "v3_3.25bit": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=3, n_outlier=hd // 4, outlier_bits=4, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "v3_2.5bit": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=2, n_outlier=hd // 4, outlier_bits=3, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "v3_2.5bit_b": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=2, n_outlier=hd // 2, outlier_bits=3, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
    "v3_2.75bit": lambda nl, hd: [
        TurboQuantKVCacheV3(head_dim=hd, bits=2, n_outlier=3 * hd // 4, outlier_bits=3, use_qjl=False, seed=42 + i)
        for i in range(nl)
    ],
}


def make_cache(model, strategy):
    """Creates a KV cache list for the given model and strategy name."""
    if strategy == "fp16":
        return make_prompt_cache(model)

    factory = _STRATEGIES.get(strategy)
    if factory is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    n_layers = len(model.layers)
    head_dim = model.layers[0].self_attn.head_dim
    return factory(n_layers, head_dim)
