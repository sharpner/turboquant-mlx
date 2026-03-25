"""Experiment: Can we make 2-bit work on MLX?

The paper uses Lloyd-Max codebook quantization for 2-bit.
MLX's mx.quantize uses affine (uniform) quantization which collapses at 2-bit.

This script tests:
1. 2-bit affine with different group sizes (32, 64, 128)
2. 2-bit affine with random QR rotation
3. Lloyd-Max codebook quantization (software dequant path)
4. Quality comparison on actual attention patterns
"""

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

from benchmark_common import compute_perplexity
from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.codebook import get_codebook_unscaled
from turboquant.rotation import generate_rotation_matrix
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
    "of 1956."
)


def analyze_quantization_error(bits, group_size, use_rotation, head_dim=128):
    """Measures quantization error on random attention-like vectors."""
    mx.random.seed(42)
    # Simulate KV vectors (B=1, heads=8, T=64, D=128)
    data = mx.random.normal((1, 8, 64, head_dim))
    mx.eval(data)

    if use_rotation:
        R = generate_rotation_matrix(head_dim, seed=42)
        mx.eval(R)
        rotated = data @ R.T
    else:
        rotated = data

    # Quantize
    q_data, q_scales, q_biases = mx.quantize(rotated, group_size=group_size, bits=bits)
    mx.eval(q_data, q_scales, q_biases)

    # Dequantize
    reconstructed = mx.dequantize(q_data, q_scales, q_biases, group_size=group_size, bits=bits)

    if use_rotation:
        reconstructed = reconstructed @ R
    mx.eval(reconstructed)

    # Error metrics
    diff = data - reconstructed
    mse = mx.mean(diff * diff).item()
    rel_error = mx.mean(mx.abs(diff) / (mx.abs(data) + 1e-8)).item()

    # Cosine similarity (per-vector)
    cos_sim = mx.sum(data * reconstructed, axis=-1) / (
        mx.linalg.norm(data, axis=-1) * mx.linalg.norm(reconstructed, axis=-1) + 1e-8
    )
    avg_cos = mx.mean(cos_sim).item()

    return mse, rel_error, avg_cos


def analyze_lloyd_max_error(bits, head_dim=128):
    """Measures quantization error using Lloyd-Max codebook (software path)."""
    mx.random.seed(42)
    data = mx.random.normal((1, 8, 64, head_dim))
    mx.eval(data)

    R = generate_rotation_matrix(head_dim, seed=42)
    mx.eval(R)

    # Normalize + rotate
    norms = mx.linalg.norm(data, axis=-1, keepdims=True)
    safe_norms = mx.where(norms < 1e-8, mx.ones_like(norms), norms)
    normalized = data / safe_norms
    rotated = normalized @ R.T

    # Lloyd-Max quantization
    centroids, boundaries = get_codebook_unscaled(bits)
    mx.eval(centroids, boundaries)

    # Quantize: find nearest centroid for each value
    # For 2-bit: 4 centroids, 3 boundaries
    n_levels = 2 ** bits
    # Broadcast: rotated (..., D) vs boundaries (n_levels-1,)
    expanded = rotated[..., None]  # (..., D, 1)
    bounds = boundaries.reshape(1, 1, 1, 1, -1)  # (1, 1, 1, 1, n_levels-1)

    # Count how many boundaries each value exceeds
    indices = mx.sum(expanded > bounds, axis=-1).astype(mx.int32)  # (..., D)
    mx.eval(indices)

    # Dequantize: look up centroids
    reconstructed_rot = centroids[indices]

    # Inverse rotation + denormalize
    reconstructed_norm = reconstructed_rot @ R
    reconstructed = reconstructed_norm * safe_norms
    mx.eval(reconstructed)

    diff = data - reconstructed
    mse = mx.mean(diff * diff).item()
    rel_error = mx.mean(mx.abs(diff) / (mx.abs(data) + 1e-8)).item()

    cos_sim = mx.sum(data * reconstructed, axis=-1) / (
        mx.linalg.norm(data, axis=-1) * mx.linalg.norm(reconstructed, axis=-1) + 1e-8
    )
    avg_cos = mx.mean(cos_sim).item()

    return mse, rel_error, avg_cos


def main():
    print("=" * 70)
    print("Experiment: 2-bit Quantization Quality on Apple Silicon")
    print("=" * 70)

    # --- Part 1: Quantization error analysis ---
    print("\n--- Part 1: Quantization Error (random vectors, D=128) ---\n")
    print(f"{'Config':40s} {'MSE':>10s} {'Rel.Err':>10s} {'CosSim':>10s}")
    print("-" * 72)

    configs = [
        (4, 64, False, "4-bit affine gs=64"),
        (4, 64, True, "4-bit affine gs=64 + rotation"),
        (3, 64, False, "3-bit affine gs=64"),
        (3, 64, True, "3-bit affine gs=64 + rotation"),
        (2, 64, False, "2-bit affine gs=64"),
        (2, 64, True, "2-bit affine gs=64 + rotation"),
        (2, 32, False, "2-bit affine gs=32"),
        (2, 32, True, "2-bit affine gs=32 + rotation"),
    ]

    for bits, gs, rot, label in configs:
        mse, rel_err, cos_sim = analyze_quantization_error(bits, gs, rot)
        print(f"  {label:38s} {mse:10.6f} {rel_err:10.4f} {cos_sim:10.6f}")

    # Lloyd-Max comparison
    print()
    for bits in (2, 3):
        mse, rel_err, cos_sim = analyze_lloyd_max_error(bits)
        print(f"  {'Lloyd-Max ' + str(bits) + '-bit + rotation':38s} {mse:10.6f} {rel_err:10.4f} {cos_sim:10.6f}")

    # --- Part 2: Perplexity on actual model ---
    print(f"\n--- Part 2: Perplexity (Llama 3.2 3B, ~170 token eval) ---\n")
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = mlx_lm.load(MODEL_NAME)
    n_layers = len(model.layers)
    head_dim = model.layers[0].self_attn.head_dim
    print(f"Model loaded: {n_layers} layers, head_dim={head_dim}\n")

    ppl_configs = [
        ("fp16", "Standard fp16"),
        ("4bit_lean", "4-bit LEAN (gs=64)"),
        ("4bit_rot", "4-bit rotated (gs=64)"),
        ("3bit_lean", "3-bit LEAN (gs=64)"),
        ("2bit_lean_gs64", "2-bit LEAN (gs=64)"),
        ("2bit_lean_gs32", "2-bit LEAN (gs=32)"),
        ("2bit_rot_gs64", "2-bit rotated (gs=64)"),
        ("2bit_rot_gs32", "2-bit rotated (gs=32)"),
    ]

    for strategy, label in ppl_configs:
        if strategy == "fp16":
            cache = make_prompt_cache(model)
        elif strategy == "4bit_lean":
            cache = [TurboQuantKVCacheV2(head_dim=head_dim, bits=4, group_size=64,
                     use_rotation=False, use_normalization=False, seed=42+i) for i in range(n_layers)]
        elif strategy == "4bit_rot":
            cache = [TurboQuantKVCacheV2(head_dim=head_dim, bits=4, group_size=64,
                     use_rotation=True, use_normalization=True, seed=42+i) for i in range(n_layers)]
        elif strategy == "3bit_lean":
            cache = [TurboQuantKVCacheV2(head_dim=head_dim, bits=3, group_size=64,
                     use_rotation=False, use_normalization=False, seed=42+i) for i in range(n_layers)]
        elif strategy == "2bit_lean_gs64":
            cache = [TurboQuantKVCacheV2(head_dim=head_dim, bits=2, group_size=64,
                     use_rotation=False, use_normalization=False, seed=42+i) for i in range(n_layers)]
        elif strategy == "2bit_lean_gs32":
            cache = [TurboQuantKVCacheV2(head_dim=head_dim, bits=2, group_size=32,
                     use_rotation=False, use_normalization=False, seed=42+i) for i in range(n_layers)]
        elif strategy == "2bit_rot_gs64":
            cache = [TurboQuantKVCacheV2(head_dim=head_dim, bits=2, group_size=64,
                     use_rotation=True, use_normalization=True, seed=42+i) for i in range(n_layers)]
        elif strategy == "2bit_rot_gs32":
            cache = [TurboQuantKVCacheV2(head_dim=head_dim, bits=2, group_size=32,
                     use_rotation=True, use_normalization=True, seed=42+i) for i in range(n_layers)]

        ppl = compute_perplexity(model, tokenizer, EVAL_TEXT, cache)
        delta = ((ppl / 12.94) - 1) * 100 if strategy != "fp16" else 0
        sign = "+" if delta >= 0 else ""
        print(f"  {label:30s}  PPL: {ppl:6.2f}  ({sign}{delta:.1f}%)")


if __name__ == "__main__":
    main()
