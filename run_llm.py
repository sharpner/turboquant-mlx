"""TurboQuant LLM Demo — Llama 3.2 3B with TurboQuant V2 KV-Cache.

V2 uses random QR rotation + MLX-native mx.quantized_matmul for
maximum hardware affinity on Apple Silicon.
"""

import time

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step

from turboquant.cache_v2 import TurboQuantKVCacheV2
import turboquant.patch as tq_patch
tq_patch.apply()

MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
PROMPT = "Explain to a child why the sky is blue."
MAX_TOKENS = 100


def make_turboquant_cache(model, bits=3, group_size=64, use_qjl=False):
    """Creates TurboQuant V2 KV-Caches for all layers."""
    head_dim = model.layers[0].self_attn.head_dim
    return [
        TurboQuantKVCacheV2(
            head_dim=head_dim, bits=bits, group_size=group_size,
            use_qjl=use_qjl, seed=42 + i,
        )
        for i in range(len(model.layers))
    ]


def generate_with_cache(model, tokenizer, prompt, cache, max_tokens=100):
    """Generates text with custom cache."""
    messages = [{"role": "user", "content": prompt}]
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
    return text, len(tokens), elapsed


def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = mlx_lm.load(MODEL_NAME)

    head_dim = model.layers[0].self_attn.head_dim
    n_layers = len(model.layers)
    print(f"Model loaded: {n_layers} layers, head_dim={head_dim}")

    configs = [
        ("TQ-V2 3-bit (rotation + mx.quantized_matmul)", dict(bits=3)),
        ("TQ-V2 4-bit (rotation + mx.quantized_matmul)", dict(bits=4)),
    ]

    for label, kwargs in configs:
        print(f"\n{'='*60}")
        print(label)
        print(f"{'='*60}")

        tq_cache = make_turboquant_cache(model, **kwargs)
        text, n_tokens, elapsed = generate_with_cache(
            model, tokenizer, PROMPT, tq_cache, MAX_TOKENS
        )

        tq_nbytes = sum(c.nbytes for c in tq_cache)
        tq_fp16_equiv = sum(c.nbytes_equivalent_fp16 for c in tq_cache)

        print(f"\nPrompt: {PROMPT}")
        print(f"Response ({n_tokens} tokens, {elapsed:.2f}s, {n_tokens/elapsed:.1f} tok/s):")
        print(f"  {text}")
        print(f"\nCache: {tq_nbytes:,} bytes ({tq_fp16_equiv/tq_nbytes:.1f}x compression vs fp16)")

    # Standard for comparison
    print(f"\n{'='*60}")
    print("Standard KV-Cache (float16)")
    print(f"{'='*60}")

    from mlx_lm.models.cache import make_prompt_cache
    std_cache = make_prompt_cache(model)
    text2, n_tokens2, elapsed2 = generate_with_cache(
        model, tokenizer, PROMPT, std_cache, MAX_TOKENS
    )
    std_nbytes = sum(c.nbytes for c in std_cache)
    print(f"\nResponse ({n_tokens2} tokens, {elapsed2:.2f}s, {n_tokens2/elapsed2:.1f} tok/s):")
    print(f"  {text2}")
    print(f"\nCache: {std_nbytes:,} bytes")


if __name__ == "__main__":
    main()
