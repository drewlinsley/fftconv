#!/usr/bin/env python3
"""
Benchmark: Sequential vs Parallel ConvSSM in Spatial vs Fourier domains.

Compares:
1. Spatial Sequential:  h_t = IFFT(FFT(h_{t-1}) * A_f) + IFFT(FFT(x_t) * B_f)
2. Spatial Parallel:    FFT all inputs → parallel scan → IFFT
3. Fourier Sequential:  Pre-FFT inputs, sequential scan (lax.scan)
4. Fourier Parallel:    Pre-FFT inputs, parallel scan (lax.associative_scan)

Usage:
    python tests/benchmark_sequential_vs_parallel.py
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from flashfftconv.conv_nd_jax import (
    convssm_sequential_3d_jit,
    convssm_parallel_3d_jit,
    to_fourier_3d_jit,
    from_fourier_3d_jit,
    kernel_to_fourier_3d_jit,
    convssm_fourier_scan_sequential_jit,
    convssm_fourier_scan_parallel_jit,
)


def create_data(key, T, B, C, D, H, W, K):
    """Create test data with decaying kernels for stability."""
    keys = random.split(key, 3)

    x_seq = random.normal(keys[0], (T, B, C, D, H, W)) * 0.1
    A_base = random.normal(keys[1], (C, K, K, K)) * 0.1
    B_base = random.normal(keys[2], (C, K, K, K)) * 0.1

    decay = jnp.exp(-0.3 * jnp.arange(K))
    decay_3d = decay[:, None, None] * decay[None, :, None] * decay[None, None, :]
    A_kernel = A_base * decay_3d
    B_kernel = B_base * decay_3d

    return x_seq, A_kernel, B_kernel


def benchmark(fn, warmup=5, rep=20):
    """Generic benchmark function."""
    for _ in range(warmup):
        result = fn()
        result.block_until_ready()

    start = time.perf_counter()
    for _ in range(rep):
        result = fn()
        result.block_until_ready()
    return (time.perf_counter() - start) / rep * 1000


def main():
    print("=" * 80)
    print("Sequential vs Parallel ConvSSM: Spatial vs Fourier Domain")
    print("=" * 80)
    print(f"\nJAX devices: {jax.devices()}")

    print("\n" + "=" * 80)
    print("METHODS:")
    print("-" * 80)
    print("1. Spatial Sequential:   FFT + conv + IFFT at EVERY timestep")
    print("2. Spatial Parallel:     FFT once → parallel scan → IFFT once")
    print("3. Fourier Sequential:   Pre-FFT → lax.scan (O(T) work, O(T) depth)")
    print("4. Fourier Parallel:     Pre-FFT → associative_scan (O(T logT) work, O(logT) depth)")
    print("=" * 80)

    configs = [
        # (T, B, C, D, H, W, K, description)
        (100, 2, 8, 4, 4, 4, 3, "Small spatial (4x4x4)"),
        (100, 2, 32, 8, 16, 16, 3, "Medium spatial (8x16x16)"),
        (100, 2, 32, 16, 32, 32, 3, "Large spatial (16x32x32)"),
        (500, 2, 8, 4, 4, 4, 3, "Small spatial, long T=500"),
    ]

    for T, B, C, D, H, W, K, desc in configs:
        print(f"\n{'='*80}")
        print(f"{desc}: T={T}, B={B}, C={C}, D={D}×{H}×{W}")
        print(f"{'='*80}")

        try:
            key = random.PRNGKey(42)
            spatial_size = (D, H, W)
            x_seq, A_kernel, B_kernel = create_data(key, T, B, C, D, H, W, K)

            # Pre-FFT for Fourier methods
            x_seq_f = to_fourier_3d_jit(x_seq, spatial_size)
            A_f = kernel_to_fourier_3d_jit(A_kernel, spatial_size)
            B_f = kernel_to_fourier_3d_jit(B_kernel, spatial_size)
            x_seq_f.block_until_ready()

            # Benchmark all methods
            spatial_seq = benchmark(lambda: convssm_sequential_3d_jit(x_seq, A_kernel, B_kernel, spatial_size))
            spatial_par = benchmark(lambda: convssm_parallel_3d_jit(x_seq, A_kernel, B_kernel, spatial_size, True))
            fourier_seq = benchmark(lambda: convssm_fourier_scan_sequential_jit(A_f, B_f, x_seq_f))
            fourier_par = benchmark(lambda: convssm_fourier_scan_parallel_jit(A_f, B_f, x_seq_f))

            # With IFFT
            fourier_seq_ifft = benchmark(lambda: from_fourier_3d_jit(
                convssm_fourier_scan_sequential_jit(A_f, B_f, x_seq_f), spatial_size))
            fourier_par_ifft = benchmark(lambda: from_fourier_3d_jit(
                convssm_fourier_scan_parallel_jit(A_f, B_f, x_seq_f), spatial_size))

            print(f"\n{'Method':<35} {'Time':>10} {'vs Best':>10}")
            print("-" * 58)

            results = [
                ("Spatial Sequential", spatial_seq),
                ("Spatial Parallel", spatial_par),
                ("Fourier Sequential (no IFFT)", fourier_seq),
                ("Fourier Parallel (no IFFT)", fourier_par),
                ("Fourier Sequential (+IFFT)", fourier_seq_ifft),
                ("Fourier Parallel (+IFFT)", fourier_par_ifft),
            ]

            best = min(r[1] for r in results)
            for name, t in results:
                marker = " ★" if t == best else ""
                print(f"{name:<35} {t:>8.2f}ms {t/best:>8.2f}x{marker}")

            # Summary
            print(f"\n  Winner (no IFFT): {'Fourier Sequential' if fourier_seq < fourier_par else 'Fourier Parallel'}")
            print(f"  Fourier speedup vs Spatial: {spatial_seq/fourier_seq:.1f}x (seq), {spatial_par/fourier_par:.1f}x (par)")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
When to use which:

  Fourier Sequential (lax.scan):
    - Best for LARGE spatial sizes (GPU saturated)
    - O(T) total work beats O(T log T)
    - Better memory locality

  Fourier Parallel (associative_scan):
    - Best for SMALL spatial sizes + LARGE T
    - O(log T) depth matters when GPU has spare capacity
    - Can be 10-75x faster than sequential for small tensors!

The 'auto' mode in convssm_fourier_scan() selects based on tensor size.
""")


if __name__ == '__main__':
    main()
