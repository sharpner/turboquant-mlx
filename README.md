# TurboQuant MLX — KV-Cache Compression on Apple Silicon

Reproduction of KV-Cache quantization from [TurboQuant (Google, 2025)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) ([Paper](https://arxiv.org/abs/2504.19874)) on Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

**Result:** Up to 5.5x KV-Cache compression. Two paths: V2 (hardware-accelerated, `mx.quantized_matmul`) for speed, V3 (Lloyd-Max codebook, paper-correct) for maximum quality. Mostly MLX-native ops, with a custom Metal kernel for fused QJL sign-bit scoring.

## Benchmark Results

Tested on Apple M4 Max (64 GB), models from `mlx-community` (4-bit weight quantized).

### Multi-Model Quality (Perplexity, lower is better)

| Strategy | bits/dim | Llama 3.2 3B | Llama 3.1 8B | Mistral 7B | Gemma 3 4B |
|----------|:---:|:---:|:---:|:---:|:---:|
| | | D=128 | D=128 | D=128 | D=256 |
| fp16 baseline | 16 | 12.94 | 9.47 | 6.79 | 12.18 |
| **V2 3-bit rot+QJL** | 3 | 13.63 (+5.3%) | 10.21 (+7.8%) | 7.14 (+5.1%) | **12.05 (-1.1%)** |
| V2 4-bit rotated | 4 | 12.84 (-0.8%) | 9.61 (+1.4%) | 6.89 (+1.4%) | 12.53 (+2.9%) |
| V2 4-bit LEAN | 4 | 13.02 (+0.6%) | 9.85 (+4.0%) | 6.87 (+1.2%) | 12.37 (+1.6%) |
| **V3 3.5-bit mixed** | 3.5 | **12.98 (+0.3%)** | 10.10 (+6.7%) | 7.06 (+4.0%) | 12.44 (+2.1%) |
| V3 3.25-bit mixed | 3.25 | 13.57 (+4.8%) | 10.25 (+8.3%) | 7.17 (+5.6%) | 12.74 (+4.6%) |
| V3 3-bit Lloyd-Max | 3 | 13.60 (+5.1%) | 10.28 (+8.6%) | 7.27 (+7.0%) | 12.93 (+6.2%) |
| V3 2.75-bit mixed | 2.75 | 14.95 (+15.5%) | 11.21 (+18.4%) | 7.33 (+7.9%) | 13.88 (+14.0%) |
| V3 2.5-bit mixed | 2.5 | 16.44 (+27.0%) | 12.80 (+35.2%) | 7.53 (+10.8%) | 13.04 (+7.0%) |
| V3 2-bit Lloyd-Max | 2 | 21.27 (+64.3%) | 15.67 (+65.5%) | 8.10 (+19.3%) | 14.64 (+20.2%) |

**Key finding:** V2 3-bit rot+QJL beats fp16 on Gemma 3 (D=256) — the rotation + QJL correction acts as a regularizer at larger head dimensions. V3 2.5-bit on Gemma (+7.0%) is dramatically better than on Llama 3B (+27.0%), confirming that larger head_dim improves quantization quality.

### Throughput (Llama 3.2 3B, tok/s)

```
Strategy              T=512   T=1024   T=2048   T=4096   T=8192
──────────────────────────────────────────────────────────────────
Standard fp16          208      199      191      175      148
MLX 4-bit Quant        188      188      184      174      156
V2 4-bit LEAN          188      188      184      174      156
V2 4-bit (rotated)     135      133      131      124      115
V2 3-bit rot+QJL       101       96       84       65       45
V3 3.5-bit mixed        82       74       59       42       24
V3 3-bit Lloyd-Max      98       86       70       47       27
V3 2.5-bit mixed        83       75       59       42       24
```

V2 uses `mx.quantized_matmul` (Metal kernel) — near-native speed.
V3 uses software dequant (centroid lookup + `mx.matmul`) — slower but paper-correct quality.

### KV-Cache Compression at T=8192

| Strategy | Cache Size | Compression |
|----------|------------|-------------|
| fp16 | 969 MB | 1x |
| V2 4-bit LEAN | 266 MB | 3.6x |
| V3 3.5-bit mixed | 236 MB | 4.1x |
| V3 3-bit Lloyd-Max | 207 MB | 4.7x |
| V3 2.5-bit mixed | 177 MB | 5.5x |

### Recommendation

| Use Case | Strategy | Quality (D=128) | Quality (D=256) | Speed |
|----------|----------|---------|---------|-------|
| Maximum speed | V2 4-bit LEAN | +0.6-4% PPL | +1.6% PPL | ~105% of fp16 at 8K |
| Best quality at 4-bit | V2 4-bit rotated | -0.8 to +1.4% | +2.9% | ~78% of fp16 |
| Best 3-bit (D=256) | V2 3-bit rot+QJL | +5-8% | **-1.1%** | ~30% of fp16 at 8K |
| Near-lossless compression | V3 3.5-bit mixed | +0.3-7% | +2.1% | ~16% of fp16 |
| Balanced | V3 3-bit Lloyd-Max | +5-9% | +6.2% | ~18% of fp16 |
| Aggressive compression | V3 2.5-bit mixed | +11-35% | +7.0% | ~16% of fp16 |

## Architecture

```
┌─────────────────────────────────────────────┐
│  mlx-lm (Llama, Mistral, ...)               │
│    ↓ SDPA dispatch (monkey-patch)           │
├─────────────────────────────────────────────┤
│  turboquant.patch                            │
│    → Detects TurboQuant cache objects       │
│    → Routes to V2 or V3 attention           │
├─────────────────────────────────────────────┤
│                                             │
│  V2 Path (Speed)         V3 Path (Quality)  │
│  ┌───────────────┐       ┌───────────────┐  │
│  │ attention_v2   │       │ attention_v3   │  │
│  │ mx.quantized_  │       │ Centroid lookup│  │
│  │ matmul (Metal) │       │ + mx.matmul    │  │
│  ├───────────────┤       ├───────────────┤  │
│  │ cache_v2       │       │ cache_v3       │  │
│  │ mx.quantize    │       │ Lloyd-Max      │  │
│  │ affine quant   │       │ codebook quant │  │
│  │ ± rotation     │       │ + rotation     │  │
│  │ ± QJL          │       │ ± channel split│  │
│  └───────────────┘       └───────────────┘  │
│                                             │
├─────────────────────────────────────────────┤
│  Shared: codebook.py, codebook_ops.py,      │
│  qjl.py, rotation.py                        │
├─────────────────────────────────────────────┤
│  MLX Metal Backend                           │
│    → quantized_matmul (V2 only)             │
│    → All ops are MLX-native                 │
└─────────────────────────────────────────────┘
```

### V2 Variants (Affine Quantization, Hardware-Accelerated)

| Variant | Rotation | Norm-Baking | QJL | Speed | Description |
|---------|:---:|:---:|:---:|:---:|---|
| **LEAN** | — | — | — | Fastest | `mx.quantize` directly. Matches MLX built-in `QuantizedKVCache`. |
| **rotated** | ✓ | ✓ | — | ~70% | Random QR rotation + norm-baking. Best 4-bit quality. |
| **rotated+QJL** | ✓ | ✓ | ✓ | ~30% | +1-bit residual correction. Fused Metal kernel for sign-bit scoring. |

### V3 Variants (Lloyd-Max Codebook, Paper-Correct)

| Variant | Channels | Description |
|---------|----------|-------------|
| **uniform** | all @ b-bit | Lloyd-Max codebook at b bits. Best quality per bit. |
| **mixed** | n@(b+1) + rest@b | Outlier channel splitting. Fractional bit rates (2.5, 3.5). |

## Paper Reproduction

### What was confirmed

1. **Quality-neutral at 4-bit** — PPL 13.02 vs 12.94 fp16 (+0.6%). With rotation: 12.84 (-0.8%)
2. **3.6-5.5x cache compression** depending on bit width
3. **Bandwidth crossover** — V2 compressed cache overtakes fp16 at T~4K
4. **Random rotation (QR) improves quality** — distributes outlier channels evenly
5. **Lloyd-Max codebook beats affine at 3-bit** — PPL +5-9% vs +9-23% (V3 vs V2)
6. **Outlier channel splitting enables fractional bit rates** — V3 3.5-bit mixed: +0.3% PPL
7. **QJL improves V2 3-bit** — from +6.6% to +5.3% as additional correction
8. **Results generalize** across Llama 3.2 3B, Llama 3.1 8B, Mistral 7B, Gemma 3 4B
9. **Larger head_dim improves quantization** — Gemma (D=256) shows dramatically better quality at low bits than Llama (D=128). V3 2.5-bit: +7% (Gemma) vs +27% (Llama 3B)
10. **V2 3-bit rot+QJL beats fp16 on Gemma** — PPL 12.05 vs 12.18 (-1.1%). Rotation + QJL acts as regularizer at D=256

### What differs

- **Hardware:** Paper tests on A100 (80 GB HBM2e, 2.0 TB/s). We test on M4 Max (Unified Memory, ~400 GB/s).
- **Weight precision:** Paper tests full-precision (bfloat16) models. We test 4-bit weight quantized models, which compounds KV cache quantization error.
- **Kernels:** Paper uses custom CUDA kernels for codebook dequant. We use MLX-native ops. V2 uses `mx.quantized_matmul` (Metal kernel, fast). V3 uses software dequant via centroid lookup (correct, slow).
- **TurboQuant_prod:** The paper's (b-1)-bit MSE + 1-bit QJL scheme doesn't improve quality at D=128 or D=256 in our tests. QJL works as an *additional* correction (V2 3-bit rot+QJL) but not as a *replacement* for MSE bits. See analysis below.
- **2-bit quality:** Both V3 Lloyd-Max and V2 affine degrade ~60% at 2-bit (D=128). With channel splitting (2.5-bit mixed), quality improves to +7-35% depending on model and head_dim. Gemma (D=256) achieves +7% vs Llama 3B (D=128) at +27%.
- **V3 throughput:** Without custom Metal kernels for codebook dequant+matmul, V3 runs ~5-6x slower than V2. On A100 with custom CUDA kernels, the paper avoids this penalty.

### Why TurboQuant_prod doesn't help

The paper's TurboQuant_prod uses (b-1)-bit MSE + 1-bit QJL for inner-product-optimal quantization. The QJL correction estimates `<q, residual>` via the Johnson-Lindenstrauss sign projection.

In our tests, TurboQuant_prod consistently degrades quality at **both D=128 and D=256**:
- V3 3-bit prod (2-bit MSE + QJL): PPL 19.48 vs V3 3-bit MSE: 13.60 (D=128)
- At D=256 (Gemma head_dim), the gap does NOT shrink — prod remains worse

**Root cause: centroid resolution loss through softmax amplification.**

The JL estimator variance scales correctly as O(π/(2d)) for unit-norm queries (verified in tests). But the real bottleneck is not JL variance — it's the **catastrophic centroid resolution drop** from b-bit to (b-1)-bit:
- 3-bit (8 centroids): MSE distortion 0.034σ²
- 2-bit (4 centroids): MSE distortion 0.120σ² — **3.5x worse**

The QJL correction applies a **linear** correction to attention scores, but softmax amplifies score errors **exponentially**. Having 4 centroids instead of 8 creates coarser score quantization that softmax magnifies into attention weight errors far exceeding what the QJL correction can recover.

QJL *does* work when added as extra information (V2 3-bit rot+QJL: +5.3% vs +6.6% without QJL), but not when it replaces MSE bits (TurboQuant_prod). This holds across all tested dimensions and models.

**Note:** The paper may achieve different results with custom CUDA kernels, full-precision weight models, and potentially different QJL scaling. Our models use 4-bit weight quantization, which compounds KV cache quantization error.

## Quickstart

```bash
# Requirements: Apple Silicon Mac with Python 3.10+
pip install mlx mlx-lm

# Demo: text generation with compressed KV cache
python run_llm.py

# Benchmark: speed + quality
python benchmark.py

# Long-context benchmark: throughput at 512-8192 tokens
python benchmark_longseq.py

# Multi-model benchmark: PPL across 4 models (incl. Gemma D=256)
python benchmark_models.py
```

### Custom Models

```python
import mlx_lm
from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v3 import TurboQuantKVCacheV3
import turboquant.patch as tq_patch

tq_patch.apply()  # Monkey-patch SDPA dispatch

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
head_dim = model.layers[0].self_attn.head_dim
n_layers = len(model.layers)

# Option A: V2 4-bit (fast, hardware-accelerated)
cache = [
    TurboQuantKVCacheV2(
        head_dim=head_dim, bits=4, group_size=64,
        use_rotation=True, use_normalization=True,
    )
    for _ in range(n_layers)
]

# Option B: V3 3.5-bit mixed (near-lossless, 4.1x compression)
cache = [
    TurboQuantKVCacheV3(
        head_dim=head_dim, bits=3,
        n_outlier=64, outlier_bits=4,  # 64 channels @ 4-bit, 64 @ 3-bit
    )
    for _ in range(n_layers)
]
```

## Project Structure

```
turboquant/
├── cache_v2.py          # V2: KV cache with mx.quantize (affine, fast)
├── cache_v3.py          # V3: Lloyd-Max codebook + channel splitting
├── attention_v2.py      # V2: SDPA with mx.quantized_matmul
├── attention_v3.py      # V3: SDPA with software dequant
├── codebook.py          # Lloyd-Max optimal centroids (1-4 bit)
├── codebook_ops.py      # Pure MLX pack/unpack for 2/3/4-bit indices
├── qjl.py               # Pure MLX QJL encoding (sign-bit packing)
├── fused_qjl.py         # Fused Metal kernel for QJL sign-bit dot products
├── patch.py             # Monkey-patch for mlx-lm SDPA dispatch
├── rotation.py          # Random rotation (QR) + JL matrix generation
├── kernels.py           # V1: Metal kernels + packing (legacy)
├── cache.py             # V1: cache (legacy)
├── attention.py         # V1: attention (legacy)
└── attention_fused.py   # V1: fused attention (legacy)

benchmark.py             # Speed + quality benchmark
benchmark_common.py      # Shared eval text and perplexity computation
benchmark_longseq.py     # Long-context throughput benchmark
benchmark_models.py      # Multi-model PPL comparison
run_llm.py               # Interactive demo
tests/
├── test_turboquant.py   # 58 unit tests (core components)
└── test_metal_barrier.py # Metal kernel barrier reproduction test
```

## Technical Details

### Pre-allocation (step=256)

Both V2 and V3 use pre-allocated buffers with slice assignment instead of per-token concatenation. Reduces allocations from O(T) to O(T/256).

### Norm-Baking (V2)

For the rotated variant, L2 norms are baked into quantized scales/biases:
```
dequant(data, norm*scale, norm*bias) = norm * dequant(data, scale, bias)
```
Eliminates 2 element-wise operations from the SDPA hot path.

### Lloyd-Max Codebook (V3)

After random rotation, each coordinate is ~N(0, 1/sqrt(D)). Lloyd-Max gives optimal centroids for this distribution:
- **4-bit** (16 levels): Nearly identical to affine. Both work well.
- **3-bit** (8 levels): Lloyd-Max significantly better. Non-uniform spacing matches Gaussian tails.
- **2-bit** (4 levels): Both degrade substantially. Need channel splitting for usable quality.

### Outlier Channel Splitting (V3)

After rotation, all channels are statistically equivalent (iid Gaussian). A fixed channel split achieves fractional bit rates:
- **3.5-bit:** 64 channels @ 4-bit + 64 @ 3-bit = (64*4+64*3)/128 = 3.5 bits/dim
- **2.5-bit:** 64 channels @ 3-bit + 64 @ 2-bit = (64*3+64*2)/128 = 2.5 bits/dim

The split is fixed (no per-token overhead) because rotation eliminates channel-dependent outliers.

### QJL Residual Correction (V2)

The residual (original - dequantized) is projected through a random matrix and stored as 1-bit sign bits. During attention, this corrects key score estimation via the JL inner product estimator.

Works as an *additional* correction on V2 affine quantization (3-bit: +6.6% -> +5.3%). Does NOT work as a bit replacement (TurboQuant_prod) because the (b-1)-bit centroid resolution loss is amplified exponentially by softmax, overwhelming the linear QJL correction.

### MLX-LM Bug: QuantizedKVCache.nbytes

MLX-LM's `QuantizedKVCache.nbytes` property crashes with `NameError: name 'tree_reduce' is not defined` because `tree_reduce` is used but not imported in `cache.py`. Our benchmarks work around this by manually summing tensor sizes.

## References

- [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research Blog
- [TurboQuant Paper](https://arxiv.org/abs/2504.19874) — arXiv, 2025
- [MLX](https://github.com/ml-explore/mlx) — Apple Machine Learning Framework
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) — Language Models for MLX

## License

MIT
