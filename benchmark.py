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
from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache

from turboquant.cache import TurboQuantKVCache
from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v3 import TurboQuantKVCacheV3
import turboquant.patch as tq_patch
tq_patch.apply()

def _cache_nbytes(cache_layer) -> int:
    """Computes cache memory in bytes. Workaround for mlx-lm bug where
    QuantizedKVCache.nbytes crashes due to missing tree_reduce import."""
    # TurboQuant caches have working .nbytes
    if hasattr(cache_layer, 'is_turboquant') or hasattr(cache_layer, 'is_turboquant_v2') or hasattr(cache_layer, 'is_turboquant_v3'):
        return cache_layer.nbytes
    # KVCache (fp16)
    if isinstance(cache_layer, KVCache):
        if cache_layer.keys is None:
            return 0
        return cache_layer.keys.nbytes + cache_layer.values.nbytes
    # QuantizedKVCache — tree_reduce is broken in mlx-lm, sum manually
    if isinstance(cache_layer, QuantizedKVCache):
        if cache_layer.keys is None:
            return 0
        total = 0
        for tensor in (*cache_layer.keys, *cache_layer.values):
            total += tensor.nbytes
        return total
    return 0


MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
PROMPT = "Write a short story about a robot learning to cook."
MAX_TOKENS = 150
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


def make_cache(model, strategy):
    """Creates cache based on strategy."""
    n_layers = len(model.layers)
    head_dim = model.layers[0].self_attn.head_dim

    if strategy == "fp16":
        return make_prompt_cache(model)
    if strategy == "quant4":
        return [QuantizedKVCache(group_size=64, bits=4) for _ in range(n_layers)]
    if strategy == "quant8":
        return [QuantizedKVCache(group_size=64, bits=8) for _ in range(n_layers)]
    if strategy == "turboquant2":
        return [
            TurboQuantKVCache(head_dim=head_dim, mse_bits=2, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "turboquant3":
        return [
            TurboQuantKVCache(head_dim=head_dim, mse_bits=3, use_qjl=True, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "turboquant3_noqjl":
        return [
            TurboQuantKVCache(head_dim=head_dim, mse_bits=3, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tq_fused_2bit":
        return [
            TurboQuantKVCache(head_dim=head_dim, mse_bits=2, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_2bit":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=2, group_size=64, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_3bit_norot":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=3, group_size=64, use_qjl=False, use_rotation=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_4bit_norot":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=4, group_size=64, use_qjl=False, use_rotation=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_4bit_lean":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=4, group_size=64, use_qjl=False, use_rotation=False, use_normalization=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_3bit_lean":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=3, group_size=64, use_qjl=False, use_rotation=False, use_normalization=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_3bit":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=3, group_size=64, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_4bit":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=4, group_size=64, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    # --- V3: Lloyd-Max Codebook (paper-correct) ---
    if strategy == "tqv3_2bit":
        return [
            TurboQuantKVCacheV3(head_dim=head_dim, bits=2, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv3_2bit_prod":
        return [
            TurboQuantKVCacheV3(head_dim=head_dim, bits=2, use_qjl=True, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv3_3bit":
        return [
            TurboQuantKVCacheV3(head_dim=head_dim, bits=3, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv3_3bit_prod":
        return [
            TurboQuantKVCacheV3(head_dim=head_dim, bits=3, use_qjl=True, seed=42 + i)
            for i in range(n_layers)
        ]
    raise ValueError(f"Unknown strategy: {strategy}")


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
        cache_bytes += _cache_nbytes(c)

    return {
        "text": text,
        "n_tokens": len(tokens),
        "elapsed": elapsed,
        "tok_per_sec": len(tokens) / elapsed if elapsed > 0 else 0,
        "cache_bytes": cache_bytes,
    }


def compute_perplexity(model, tokenizer, text, cache):
    """Computes perplexity on an evaluation text."""
    input_ids = mx.array(tokenizer.encode(text))[None]  # (1, T)
    T = input_ids.shape[1]

    if T < 2:
        return float("inf")

    logits = model(input_ids, cache=cache)
    # Shift: logits[:-1] predicts tokens[1:]
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
