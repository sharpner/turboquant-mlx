"""Metal Kernel Barrier — Reproduction tests.

Tests whether mx.fast.metal_kernel correctly sees lazily-computed inputs.

Background: During TurboQuant V1 development, we observed that custom Metal
kernels produced garbage output when the KV cache was updated via lazy ops
(concatenation) without an explicit mx.eval() in between. The fused attention
kernel read stale GPU buffers.

Analysis of MLX source (device.cpp:323-352) showed that maybeInsertBarrier()
only inserts threadgroup-level barriers, not memory-write-completion barriers.
Under heavy compute graphs with async_eval (as used by mlx-lm's generate_step),
this can cause custom kernels to read stale data.

The fix for V1 was adding mx.eval() after every cache update.
V2 avoids the issue entirely by using only MLX-native ops (mx.quantized_matmul).

Note: Simple test cases with small arrays may not trigger the bug. The issue
is timing-dependent and manifests under load (large buffers, async_eval,
complex compute graphs).
"""

import mlx.core as mx
import pytest


def _has_metal_kernel():
    return hasattr(mx.fast, "metal_kernel")


IDENTITY_KERNEL = """
uint idx = thread_position_in_grid.x;
if (idx < inp_shape[0]) {
    out[idx] = inp[idx];
}
"""


@pytest.mark.skipif(not _has_metal_kernel(), reason="mx.fast.metal_kernel not available")
class TestMetalBarrier:

    def test_native_ops_after_lazy_concat(self):
        """MLX native ops correctly see concatenated data without eval."""
        a = mx.ones(64)
        b = mx.full(64, 2.0)
        combined = mx.concatenate([a, b])  # lazy
        result = combined * 3.0  # lazy
        mx.eval(result)

        assert abs(result[0].item() - 3.0) < 1e-6
        assert abs(result[64].item() - 6.0) < 1e-6

    def test_metal_kernel_with_eval_is_correct(self):
        """Custom Metal kernel works correctly WITH explicit mx.eval()."""
        a = mx.ones(128)
        b = a + 1.0
        mx.eval(b)

        out = mx.fast.metal_kernel(
            name="identity_eval",
            input_names=["inp"],
            output_names=["out"],
            source=IDENTITY_KERNEL,
        )(inputs=[b], output_shapes=[(128,)], output_dtypes=[mx.float32], grid=(128, 1, 1), threadgroup=(128, 1, 1))

        mx.eval(out)
        assert abs(out[0][0].item() - 2.0) < 1e-6

    def test_metal_kernel_without_eval_simple(self):
        """Simple custom kernel test without eval — may or may not trigger bug.

        The barrier issue is timing-dependent and typically requires:
        - Large buffers (MBs, not bytes)
        - async_eval (as used by mlx-lm generation)
        - Complex preceding compute graph

        This simple test serves as a canary — if it fails, the bug is
        definitely present. If it passes, the bug may still exist under load.
        """
        a = mx.ones(128)
        b = a + 1.0  # lazy

        out = mx.fast.metal_kernel(
            name="identity_noval",
            input_names=["inp"],
            output_names=["out"],
            source=IDENTITY_KERNEL,
        )(inputs=[b], output_shapes=[(128,)], output_dtypes=[mx.float32], grid=(128, 1, 1), threadgroup=(128, 1, 1))

        mx.eval(out)
        value = out[0][0].item()
        # May read 2.0 (correct) or stale value depending on timing
        # We log the result either way
        if abs(value - 2.0) > 1e-4:
            pytest.fail(
                f"Metal barrier bug triggered in simple test: "
                f"expected 2.0, got {value}"
            )

    def test_metal_kernel_large_buffer_after_concat(self):
        """Larger buffer test — closer to real KV cache patterns.

        Uses 1MB+ buffers and multiple lazy concatenations to increase
        the chance of triggering the barrier issue.
        """
        # Simulate growing KV cache: multiple concat steps without eval
        size = 256 * 1024  # 256K floats = 1 MB
        cache = mx.zeros(size)
        mx.eval(cache)

        for i in range(4):
            new_data = mx.full(size // 4, float(i + 1))
            cache = mx.concatenate([cache, new_data])
            # NO mx.eval() — all lazy

        # Total size: 5 * 64K = 320K floats
        expected_size = size + 4 * (size // 4)

        out = mx.fast.metal_kernel(
            name="large_concat_test",
            input_names=["inp"],
            output_names=["out"],
            source=IDENTITY_KERNEL,
        )(inputs=[cache], output_shapes=[(expected_size,)], output_dtypes=[mx.float32],
          grid=(expected_size, 1, 1), threadgroup=(256, 1, 1))

        mx.eval(out)

        # Check last chunk should be 4.0
        last_val = out[0][-1].item()
        if abs(last_val - 4.0) > 1e-4:
            pytest.fail(
                f"Metal barrier bug with large buffers: "
                f"expected 4.0 at end, got {last_val}"
            )

    def test_quantized_matmul_after_lazy_concat(self):
        """MLX-native quantized_matmul correctly reads lazy-concatenated data.

        This is the V2 approach and must always work correctly.
        """
        keys_a = mx.random.normal((1, 8, 4, 128))
        mx.eval(keys_a)
        q_a = mx.quantize(keys_a, group_size=64, bits=4)
        mx.eval(*q_a)

        keys_b = mx.random.normal((1, 8, 4, 128))
        mx.eval(keys_b)
        q_b = mx.quantize(keys_b, group_size=64, bits=4)
        # q_b is lazy — NOT evaluated

        # Concatenate (lazy)
        q_combined = tuple(
            mx.concatenate([a, b], axis=2)
            for a, b in zip(q_a, q_b)
        )

        # quantized_matmul on lazy-concatenated data
        queries = mx.random.normal((1, 8, 1, 128))
        mx.eval(queries)

        scores = mx.quantized_matmul(
            queries, *q_combined,
            transpose=True, group_size=64, bits=4,
        )
        mx.eval(scores)

        assert scores.shape == (1, 8, 1, 8)
        assert mx.isfinite(scores).all().item(), "Scores contain NaN/Inf"
        assert mx.var(scores).item() > 1e-6, "Scores have no variance"

    def test_async_eval_with_metal_kernel(self):
        """Test with mx.async_eval — closer to mlx-lm's generation loop.

        mlx-lm uses async_eval for overlapping compute and Python execution.
        This is the context where the barrier bug was originally observed.
        """
        size = 64 * 1024  # 64K floats

        a = mx.random.normal((size,))
        mx.async_eval(a)  # Async — may not be done when kernel runs

        b = a * 2.0 + 1.0  # Lazy on top of async

        out = mx.fast.metal_kernel(
            name="async_test",
            input_names=["inp"],
            output_names=["out"],
            source=IDENTITY_KERNEL,
        )(inputs=[b], output_shapes=[(size,)], output_dtypes=[mx.float32],
          grid=(size, 1, 1), threadgroup=(256, 1, 1))

        mx.eval(out)

        # Verify against reference
        mx.eval(b)
        ref_val = b[0].item()
        got_val = out[0][0].item()

        if abs(ref_val - got_val) > 1e-3:
            pytest.fail(
                f"Metal barrier bug with async_eval: "
                f"expected {ref_val}, got {got_val}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
