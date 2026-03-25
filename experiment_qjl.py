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

from turboquant.cache_v2 import TurboQuantKVCacheV2
import turboquant.patch as tq_patch

tq_patch.apply()

MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"

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
    input_ids = mx.array(tokenizer.encode(text))[None]
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
