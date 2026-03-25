"""TurboQuant Benchmark — Comparison of different KV-Cache strategies.

Measures memory, tokens/second and perplexity for:
  1. Standard fp16 KVCache
  2. MLX QuantizedKVCache (4-bit)
  3. MLX QuantizedKVCache (8-bit)
  4. TurboQuant (2-bit MSE + 1-bit QJL)
"""

import time

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step

from benchmark_common import EVAL_TEXT, compute_perplexity, cache_nbytes, make_cache
import turboquant.patch as tq_patch
tq_patch.apply()


MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
PROMPT = "Write a short story about a robot learning to cook."
MAX_TOKENS = 150


def benchmark_generation(model, tokenizer, cache, max_tokens=MAX_TOKENS):
    """Generates text and measures performance."""
    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = mx.array(tokenizer.encode(formatted))

    tokens = []
    start = time.perf_counter()

    for token, logprobs in generate_step(
        prompt=input_ids,
        model=model,
        max_tokens=max_tokens,
        prompt_cache=cache,
    ):
        tok = token.item() if hasattr(token, "item") else int(token)
        if tok == tokenizer.eos_token_id:
            break
        tokens.append(tok)

    elapsed = time.perf_counter() - start
    text = tokenizer.decode(tokens)

    cache_bytes = 0
    for c in cache:
        cache_bytes += cache_nbytes(c)

    return {
        "text": text,
        "n_tokens": len(tokens),
        "elapsed": elapsed,
        "tok_per_sec": len(tokens) / elapsed if elapsed > 0 else 0,
        "cache_bytes": cache_bytes,
    }


def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = mlx_lm.load(MODEL_NAME)
    print(f"Model loaded: {len(model.layers)} layers\n")

    strategies = [
        ("fp16", "Standard fp16"),
        ("quant4", "MLX 4-bit Quant"),
        ("tqv2_4bit_lean", "V2 4bit LEAN"),
        ("tqv2_3bit_lean", "V2 3bit LEAN"),
        ("tqv2_4bit_norot", "V2 4bit NO-ROT"),
        ("tqv2_3bit_norot", "V2 3bit NO-ROT"),
        ("tqv2_4bit", "V2 4bit (rotated)"),
        # V3: Lloyd-Max codebook (paper-correct)
        ("tqv3_3bit", "V3 3bit (Lloyd-Max)"),
        ("tqv3_3bit_prod", "V3 3bit prod (2b+QJL)"),
        ("tqv3_2bit", "V3 2bit (Lloyd-Max)"),
        ("tqv3_2bit_prod", "V3 2bit prod (1b+QJL)"),
    ]

    results = {}
    for strategy, label in strategies:
        print(f"{'='*60}")
        print(f"Benchmark: {label}")
        print(f"{'='*60}")

        cache = make_cache(model, strategy)
        result = benchmark_generation(model, tokenizer, cache)
        results[strategy] = result

        print(f"  Tokens:    {result['n_tokens']}")
        print(f"  Time:      {result['elapsed']:.2f}s")
        print(f"  Tok/s:     {result['tok_per_sec']:.1f}")
        print(f"  Cache:     {result['cache_bytes']:,} bytes")
        print(f"  Response:  {result['text'][:120]}...")
        print()

    # --- Perplexity ---
    print(f"\n{'='*60}")
    print("Perplexity Comparison")
    print(f"{'='*60}")
    print(f"Eval-Text: \"{EVAL_TEXT[:60]}...\"")
    print()

    for strategy, label in strategies:
        cache = make_cache(model, strategy)
        ppl = compute_perplexity(model, tokenizer, EVAL_TEXT, cache)
        results[strategy]["perplexity"] = ppl
        print(f"  {label:25s}  PPL: {ppl:.2f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Strategy':25s} {'Tok/s':>8s} {'Cache':>12s} {'PPL':>8s}")
    print("-" * 55)
    for strategy, label in strategies:
        r = results[strategy]
        ppl_str = f"{r.get('perplexity', 0):.2f}"
        print(f"{label:25s} {r['tok_per_sec']:>8.1f} {r['cache_bytes']:>10,} B {ppl_str:>8s}")

    # Compression
    fp16_bytes = results["fp16"]["cache_bytes"]
    if fp16_bytes > 0:
        print(f"\nCompression vs fp16:")
        for strategy, label in strategies:
            if strategy == "fp16":
                continue
            ratio = fp16_bytes / results[strategy]["cache_bytes"] if results[strategy]["cache_bytes"] > 0 else 0
            print(f"  {label:25s}  {ratio:.1f}x")


if __name__ == "__main__":
    main()
