"""Multi-model benchmark: PPL comparison across different LLMs.

Tests the key strategies on multiple models to validate
that results generalize beyond Llama 3.2 3B.
"""

import time

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v3 import TurboQuantKVCacheV3
import turboquant.patch as tq_patch

tq_patch.apply()

MODELS = [
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
]

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


def make_cache(model, strategy):
    n_layers = len(model.layers)
    head_dim = model.layers[0].self_attn.head_dim

    if strategy == "fp16":
        return make_prompt_cache(model)
    if strategy == "v2_4bit_lean":
        return [TurboQuantKVCacheV2(head_dim=head_dim, bits=4, group_size=64,
                use_rotation=False, use_normalization=False, seed=42+i) for i in range(n_layers)]
    if strategy == "v2_4bit_rot":
        return [TurboQuantKVCacheV2(head_dim=head_dim, bits=4, group_size=64,
                use_rotation=True, use_normalization=True, seed=42+i) for i in range(n_layers)]
    if strategy == "v2_3bit_lean":
        return [TurboQuantKVCacheV2(head_dim=head_dim, bits=3, group_size=64,
                use_rotation=False, use_normalization=False, seed=42+i) for i in range(n_layers)]
    if strategy == "v3_3bit":
        return [TurboQuantKVCacheV3(head_dim=head_dim, bits=3, use_qjl=False, seed=42+i) for i in range(n_layers)]
    if strategy == "v3_2bit":
        return [TurboQuantKVCacheV3(head_dim=head_dim, bits=2, use_qjl=False, seed=42+i) for i in range(n_layers)]
    if strategy == "v3_3bit_prod":
        return [TurboQuantKVCacheV3(head_dim=head_dim, bits=3, use_qjl=True, seed=42+i) for i in range(n_layers)]
    if strategy == "v3_2bit_prod":
        return [TurboQuantKVCacheV3(head_dim=head_dim, bits=2, use_qjl=True, seed=42+i) for i in range(n_layers)]
    # Mixed-bit: n_outlier proportional to head_dim for consistent bit rates
    # 3.5-bit: half@4 + half@3 = (D/2*4 + D/2*3) / D = 3.5
    # 3.25-bit: quarter@4 + 3quarter@3 = (D/4*4 + 3D/4*3) / D = 3.25
    # 2.75-bit: 3quarter@3 + quarter@2 = (3D/4*3 + D/4*2) / D = 2.75
    # 2.5-bit: half@3 + half@2 = (D/2*3 + D/2*2) / D = 2.5
    if strategy == "v3_2.5bit":
        return [TurboQuantKVCacheV3(head_dim=head_dim, bits=2, n_outlier=head_dim // 4, outlier_bits=3,
                use_qjl=False, seed=42+i) for i in range(n_layers)]
    if strategy == "v3_2.5bit_b":
        return [TurboQuantKVCacheV3(head_dim=head_dim, bits=2, n_outlier=head_dim // 2, outlier_bits=3,
                use_qjl=False, seed=42+i) for i in range(n_layers)]
    if strategy == "v3_2.75bit":
        return [TurboQuantKVCacheV3(head_dim=head_dim, bits=2, n_outlier=3 * head_dim // 4, outlier_bits=3,
                use_qjl=False, seed=42+i) for i in range(n_layers)]
    if strategy == "v3_3.25bit":
        return [TurboQuantKVCacheV3(head_dim=head_dim, bits=3, n_outlier=head_dim // 4, outlier_bits=4,
                use_qjl=False, seed=42+i) for i in range(n_layers)]
    if strategy == "v3_3.5bit":
        return [TurboQuantKVCacheV3(head_dim=head_dim, bits=3, n_outlier=head_dim // 2, outlier_bits=4,
                use_qjl=False, seed=42+i) for i in range(n_layers)]
    if strategy == "v2_3bit_rot":
        return [TurboQuantKVCacheV2(head_dim=head_dim, bits=3, group_size=64,
                use_rotation=True, use_normalization=True, use_qjl=True, seed=42+i) for i in range(n_layers)]
    if strategy == "v2_2bit_rot":
        return [TurboQuantKVCacheV2(head_dim=head_dim, bits=2, group_size=32,
                use_rotation=True, use_normalization=True, seed=42+i) for i in range(n_layers)]
    raise ValueError(f"Unknown strategy: {strategy}")


STRATEGIES = [
    ("fp16", "fp16 baseline"),
    ("v2_4bit_rot", "V2 4bit rotated"),
    ("v2_4bit_lean", "V2 4bit LEAN"),
    ("v3_3.5bit", "V3 3.5bit mixed"),
    ("v3_3.25bit", "V3 3.25bit mixed"),
    ("v3_3bit", "V3 3bit Lloyd-Max"),
    ("v2_3bit_rot", "V2 3bit rot+QJL"),
    ("v3_2.75bit", "V3 2.75bit mixed"),
    ("v3_2.5bit_b", "V3 2.5bit mixed"),
    ("v3_2.5bit", "V3 2.25bit mixed"),
    ("v3_2bit", "V3 2bit Lloyd-Max"),
]


def main():
    all_results = {}

    for model_name in MODELS:
        short_name = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"Model: {short_name}")
        print(f"{'='*70}")

        model, tokenizer = mlx_lm.load(model_name)
        n_layers = len(model.layers)
        head_dim = model.layers[0].self_attn.head_dim
        print(f"  {n_layers} layers, head_dim={head_dim}\n")

        fp16_ppl = None
        for strategy, label in STRATEGIES:
            cache = make_cache(model, strategy)
            ppl = compute_perplexity(model, tokenizer, EVAL_TEXT, cache)

            if fp16_ppl is None:
                fp16_ppl = ppl

            delta = ((ppl / fp16_ppl) - 1) * 100
            sign = "+" if delta >= 0 else ""
            print(f"  {label:25s}  PPL: {ppl:6.2f}  ({sign}{delta:.1f}%)")

            all_results[(short_name, strategy)] = ppl

        # Free model memory
        del model, tokenizer
        mx.metal.clear_cache()

    # --- Summary table ---
    print(f"\n{'='*70}")
    print("Summary: PPL across models")
    print(f"{'='*70}")

    header = f"{'Strategy':25s}"
    for model_name in MODELS:
        short = model_name.split("/")[-1][:20]
        header += f"  {short:>20s}"
    print(header)
    print("-" * len(header))

    for strategy, label in STRATEGIES:
        row = f"{label:25s}"
        for model_name in MODELS:
            short = model_name.split("/")[-1]
            ppl = all_results.get((short, strategy), float("nan"))
            row += f"  {ppl:>20.2f}"
        print(row)


if __name__ == "__main__":
    main()
