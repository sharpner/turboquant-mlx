"""TurboQuant Long-Sequence Benchmark.

Measures throughput at various KV-Cache sizes.
For long sequences, cache bandwidth dominates → TurboQuant compression pays off.

Apple M3: ~100 GB/s Bandwidth
- fp16 @ T=1000: ~196 MB cache reads per forward → 1.96 ms
- V2 3-bit @ T=1000: ~41 MB cache reads → 0.41 ms (4.7x less)
"""

import time

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step

from benchmark_common import cache_nbytes, make_cache
import turboquant.patch as tq_patch

tq_patch.apply()

MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
GENERATE_TOKENS = 50  # Tokens to measure after prefill


def build_long_prompt(tokenizer, target_tokens):
    """Builds a prompt that is ~target_tokens long."""
    base = (
        "The history of humanity is long and diverse. "
        "From the first civilizations in Mesopotamia through the Roman Empire "
        "to the industrial revolution, the world has steadily changed. "
        "Technology, science and culture have continued to evolve. "
    )
    # Repeat until we have enough tokens
    text = base
    while True:
        tokens = tokenizer.encode(text)
        if len(tokens) >= target_tokens:
            return tokens[:target_tokens]
        text = text + base


def measure_generation_speed(model, tokenizer, cache, prompt_tokens, n_generate):
    """Prefill prompt, then measure generation speed."""
    input_ids = mx.array(prompt_tokens)

    tokens = []
    gen_start = None
    prefill_done = False

    for i, (token, logprobs) in enumerate(generate_step(
        prompt=input_ids,
        model=model,
        max_tokens=n_generate + 10,
        prompt_cache=cache,
    )):
        if not prefill_done:
            # First token = prefill done
            prefill_done = True
            gen_start = time.perf_counter()

        tok = token.item() if hasattr(token, "item") else int(token)
        if tok == tokenizer.eos_token_id:
            break
        tokens.append(tok)
        if len(tokens) >= n_generate:
            break

    elapsed = time.perf_counter() - gen_start
    tok_per_sec = len(tokens) / elapsed if elapsed > 0 else 0

    cache_bytes = 0
    for c in cache:
        cache_bytes += cache_nbytes(c)

    return {
        "n_tokens": len(tokens),
        "elapsed": elapsed,
        "tok_per_sec": tok_per_sec,
        "cache_bytes": cache_bytes,
        "prompt_tokens": len(prompt_tokens),
    }


def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = mlx_lm.load(MODEL_NAME)
    print(f"Model loaded: {len(model.layers)} layers\n")

    strategies = [
        ("fp16", "Standard fp16"),
        ("quant4", "MLX 4-bit Quant"),
        ("tqv2_4bit_lean", "V2 4bit LEAN"),
        ("tqv2_4bit", "V2 4bit (rotated)"),
        ("tqv2_3bit_rot_qjl", "V2 3bit rot+QJL"),
        ("tqv3_3.5bit", "V3 3.5bit mixed"),
        ("tqv3_3bit", "V3 3bit Lloyd-Max"),
        ("tqv3_2.5bit", "V3 2.5bit mixed"),
    ]

    context_lengths = [512, 1024, 2048, 4096, 8192]

    # Collect results
    all_results = {}

    for ctx_len in context_lengths:
        print(f"\n{'='*70}")
        print(f"Context Length: {ctx_len} tokens (+ {GENERATE_TOKENS} generated)")
        print(f"{'='*70}")

        prompt_tokens = build_long_prompt(tokenizer, ctx_len)

        for strategy, label in strategies:
            cache = make_cache(model, strategy)
            result = measure_generation_speed(
                model, tokenizer, cache, prompt_tokens, GENERATE_TOKENS
            )
            key = (strategy, ctx_len)
            all_results[key] = result
            print(f"  {label:20s}  {result['tok_per_sec']:>7.1f} tok/s  Cache: {result['cache_bytes']:>12,} B")

    # --- Summary ---
    print(f"\n\n{'='*70}")
    print("Summary: Tok/s at various context lengths")
    print(f"{'='*70}")

    header = f"{'Strategy':20s}"
    for ctx_len in context_lengths:
        header += f" {'T=' + str(ctx_len):>10s}"
    print(header)
    print("-" * (20 + 11 * len(context_lengths)))

    for strategy, label in strategies:
        row = f"{label:20s}"
        for ctx_len in context_lengths:
            result = all_results[(strategy, ctx_len)]
            row += f" {result['tok_per_sec']:>10.1f}"
        print(row)

    # Relative performance
    print(f"\n{'Strategy':20s}", end="")
    for ctx_len in context_lengths:
        print(f" {'T=' + str(ctx_len):>10s}", end="")
    print()
    print("-" * (20 + 11 * len(context_lengths)))

    for strategy, label in strategies:
        if strategy == "fp16":
            continue
        row = f"{label:20s}"
        for ctx_len in context_lengths:
            fp16_speed = all_results[("fp16", ctx_len)]["tok_per_sec"]
            this_speed = all_results[(strategy, ctx_len)]["tok_per_sec"]
            pct = (this_speed / fp16_speed * 100) if fp16_speed > 0 else 0
            row += f" {pct:>9.1f}%"
        print(row)

    # Cache sizes
    max_ctx = context_lengths[-1]
    print(f"\nCache size at T={max_ctx}:")
    for strategy, label in strategies:
        result = all_results[(strategy, max_ctx)]
        print(f"  {label:20s}  {result['cache_bytes']:>12,} B")
    fp16_bytes = all_results[("fp16", max_ctx)]["cache_bytes"]
    if fp16_bytes > 0:
        print(f"\nCompression vs fp16 at T={max_ctx}:")
        for strategy, label in strategies:
            if strategy == "fp16":
                continue
            cb = all_results[(strategy, max_ctx)]["cache_bytes"]
            ratio = fp16_bytes / cb if cb > 0 else 0
            print(f"  {label:20s}  {ratio:.1f}x")


if __name__ == "__main__":
    main()
