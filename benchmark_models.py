"""Multi-model benchmark: PPL comparison across different LLMs.

Tests the key strategies on multiple models to validate
that results generalize beyond Llama 3.2 3B.
"""

import mlx.core as mx
import mlx_lm

from benchmark_common import EVAL_TEXT, compute_perplexity, make_cache
import turboquant.patch as tq_patch

tq_patch.apply()

MODELS = [
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
]


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
