"""Experiment: 2-bit + QJL residual correction.

The paper's key claim: TurboQuant 2-bit + QJL 1-bit residual achieves
near-lossless quality. Let's test this on MLX.

Configurations:
1. fp16 baseline
2. 4-bit rotated (our best so far)
3. 2-bit rotated (no QJL) — currently broken
4. 2-bit rotated + QJL — the paper's approach
5. 3-bit rotated + QJL — bonus
"""

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

from benchmark_common import EVAL_TEXT, compute_perplexity
from turboquant.cache_v2 import TurboQuantKVCacheV2
import turboquant.patch as tq_patch

tq_patch.apply()

MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = mlx_lm.load(MODEL_NAME)
    n_layers = len(model.layers)
    head_dim = model.layers[0].self_attn.head_dim
    print(f"Model loaded: {n_layers} layers, head_dim={head_dim}\n")

    configs = [
        ("fp16", {}),
        ("4-bit rotated", dict(bits=4, group_size=64, use_rotation=True, use_normalization=True, use_qjl=False)),
        ("4-bit LEAN", dict(bits=4, group_size=64, use_rotation=False, use_normalization=False, use_qjl=False)),
        ("3-bit rotated", dict(bits=3, group_size=64, use_rotation=True, use_normalization=True, use_qjl=False)),
        ("3-bit rotated + QJL", dict(bits=3, group_size=64, use_rotation=True, use_normalization=True, use_qjl=True)),
        ("2-bit rotated", dict(bits=2, group_size=64, use_rotation=True, use_normalization=True, use_qjl=False)),
        ("2-bit rotated + QJL", dict(bits=2, group_size=64, use_rotation=True, use_normalization=True, use_qjl=True)),
        ("2-bit rot gs=32", dict(bits=2, group_size=32, use_rotation=True, use_normalization=True, use_qjl=False)),
        ("2-bit rot gs=32 + QJL", dict(bits=2, group_size=32, use_rotation=True, use_normalization=True, use_qjl=True)),
    ]

    print(f"{'Config':30s}  {'PPL':>8s}  {'vs fp16':>10s}")
    print("-" * 55)

    fp16_ppl = None
    for label, kwargs in configs:
        if label == "fp16":
            cache = make_prompt_cache(model)
        else:
            cache = [
                TurboQuantKVCacheV2(head_dim=head_dim, seed=42 + i, **kwargs)
                for i in range(n_layers)
            ]

        ppl = compute_perplexity(model, tokenizer, EVAL_TEXT, cache)

        if fp16_ppl is None:
            fp16_ppl = ppl
            print(f"  {label:28s}  {ppl:8.2f}  {'baseline':>10s}")
        else:
            delta = ((ppl / fp16_ppl) - 1) * 100
            sign = "+" if delta >= 0 else ""
            print(f"  {label:28s}  {ppl:8.2f}  {sign}{delta:.1f}%")

        # Print cache size for non-fp16
        if label != "fp16":
            total_bytes = sum(c.nbytes for c in cache)
            fp16_equiv = sum(c.nbytes_equivalent_fp16 for c in cache)
            if fp16_equiv > 0:
                ratio = fp16_equiv / total_bytes
                print(f"  {'':28s}  cache: {total_bytes/1024/1024:.1f} MB ({ratio:.1f}x compression)")


if __name__ == "__main__":
    main()
