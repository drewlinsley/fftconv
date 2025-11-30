#!/usr/bin/env python3
"""
Benchmark Fourier-space ConvSSM vs Regular ConvSSM.

This compares:
1. Regular parallel scan (FFT every forward pass)
2. Fourier-space parallel scan (FFT only once at data loading)

The Fourier-space version should be significantly faster since it
eliminates FFT/IFFT operations during the forward pass.

Usage:
    python tests/benchmark_fourier_space.py
"""

import time
import numpy as np

import jax
import jax.numpy as jnp
from jax import random

from flashfftconv.conv_nd_jax import (
    convssm_parallel_3d_jit,
    to_fourier_3d_jit,
    from_fourier_3d_jit,
    kernel_to_fourier_3d_jit,
    convssm_fourier_scan_jit,
    FourierConvSSM3D,
)


def benchmark_regular(T, B, C, D, H, W, K, warmup=5, rep=20):
    """Benchmark regular parallel scan (FFT every forward)."""
    key = random.PRNGKey(42)
    keys = random.split(key, 3)

    x_seq = random.normal(keys[0], (T, B, C, D, H, W)) * 0.1
    A_base = random.normal(keys[1], (C, K, K, K)) * 0.1
    B_base = random.normal(keys[2], (C, K, K, K)) * 0.1

    # Apply decay for stability
    decay = jnp.exp(-0.3 * jnp.arange(K))
    decay_3d = decay[:, None, None] * decay[None, :, None] * decay[None, None, :]
    A_kernel = A_base * decay_3d
    B_kernel = B_base * decay_3d

    spatial_size = (D, H, W)

    # Warmup (also compiles)
    for _ in range(warmup):
        h = convssm_parallel_3d_jit(x_seq, A_kernel, B_kernel, spatial_size, True)
        h.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(rep):
        h = convssm_parallel_3d_jit(x_seq, A_kernel, B_kernel, spatial_size, True)
        h.block_until_ready()
    elapsed = (time.perf_counter() - start) / rep * 1000

    return elapsed


def benchmark_fourier_space(T, B, C, D, H, W, K, warmup=5, rep=20):
    """Benchmark Fourier-space parallel scan (no FFT in forward)."""
    key = random.PRNGKey(42)
    keys = random.split(key, 3)

    x_seq = random.normal(keys[0], (T, B, C, D, H, W)) * 0.1
    A_base = random.normal(keys[1], (C, K, K, K)) * 0.1
    B_base = random.normal(keys[2], (C, K, K, K)) * 0.1

    # Apply decay for stability
    decay = jnp.exp(-0.3 * jnp.arange(K))
    decay_3d = decay[:, None, None] * decay[None, :, None] * decay[None, None, :]
    A_kernel = A_base * decay_3d
    B_kernel = B_base * decay_3d

    spatial_size = (D, H, W)

    # Pre-FFT data (done ONCE at data loading time, not counted in benchmark)
    print("    (Pre-FFT data...)")
    x_seq_f = to_fourier_3d_jit(x_seq, spatial_size)
    A_f = kernel_to_fourier_3d_jit(A_kernel, spatial_size)
    B_f = kernel_to_fourier_3d_jit(B_kernel, spatial_size)
    x_seq_f.block_until_ready()

    results = {}

    # --- Forward only (no IFFT) ---
    # Warmup
    for _ in range(warmup):
        h_f = convssm_fourier_scan_jit(A_f, B_f, x_seq_f)
        h_f.block_until_ready()

    # Benchmark forward only
    start = time.perf_counter()
    for _ in range(rep):
        h_f = convssm_fourier_scan_jit(A_f, B_f, x_seq_f)
        h_f.block_until_ready()
    results['forward_only'] = (time.perf_counter() - start) / rep * 1000

    # --- Forward + IFFT (for loss/output) ---
    # Warmup
    for _ in range(warmup):
        h_f = convssm_fourier_scan_jit(A_f, B_f, x_seq_f)
        h = from_fourier_3d_jit(h_f, spatial_size)
        h.block_until_ready()

    # Benchmark forward + IFFT
    start = time.perf_counter()
    for _ in range(rep):
        h_f = convssm_fourier_scan_jit(A_f, B_f, x_seq_f)
        h = from_fourier_3d_jit(h_f, spatial_size)
        h.block_until_ready()
    results['forward_plus_ifft'] = (time.perf_counter() - start) / rep * 1000

    return results


def benchmark_pre_fft_time(T, B, C, D, H, W, K, warmup=2, rep=5):
    """Benchmark the one-time FFT preprocessing cost."""
    key = random.PRNGKey(42)
    x_seq = random.normal(key, (T, B, C, D, H, W)) * 0.1
    spatial_size = (D, H, W)

    # Warmup
    for _ in range(warmup):
        x_f = to_fourier_3d_jit(x_seq, spatial_size)
        x_f.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(rep):
        x_f = to_fourier_3d_jit(x_seq, spatial_size)
        x_f.block_until_ready()
    elapsed = (time.perf_counter() - start) / rep * 1000

    return elapsed


def main():
    print("=" * 70)
    print("Fourier-Space ConvSSM Benchmark")
    print("=" * 70)
    print(f"\nJAX devices: {jax.devices()}")

    # Test configurations
    configs = [
        # (T, B, C, D, H, W, K)
        (10, 2, 32, 8, 16, 16, 3),    # Small
        (50, 2, 32, 8, 16, 16, 3),    # Medium T
        (100, 2, 32, 8, 16, 16, 3),   # Large T
        (100, 2, 32, 16, 32, 32, 3),  # Large spatial
        (100, 4, 64, 8, 16, 16, 3),   # Larger batch/channels
    ]

    print("\n" + "=" * 70)
    print("LEGEND:")
    print("  Regular:      FFT input + parallel scan + IFFT (every forward pass)")
    print("  Fourier FWD:  Pre-FFT input once, forward is element-wise only (no FFT!)")
    print("  Fourier+IFFT: Forward + final IFFT to get spatial output")
    print("=" * 70)

    for T, B, C, D, H, W, K in configs:
        print(f"\n{'='*70}")
        print(f"Config: T={T}, B={B}, C={C}, D={D}, H={H}, W={W}, K={K}")
        print(f"{'='*70}")

        try:
            # Pre-FFT time
            pre_fft_time = benchmark_pre_fft_time(T, B, C, D, H, W, K)
            print(f"\nPre-FFT time (one-time cost): {pre_fft_time:.2f} ms")

            # Regular benchmark
            regular_time = benchmark_regular(T, B, C, D, H, W, K)
            print(f"\nRegular parallel scan: {regular_time:.2f} ms")

            # Fourier-space benchmark
            fourier_results = benchmark_fourier_space(T, B, C, D, H, W, K)
            print(f"Fourier forward only:  {fourier_results['forward_only']:.2f} ms")
            print(f"Fourier + IFFT:        {fourier_results['forward_plus_ifft']:.2f} ms")

            # Speedup analysis
            print(f"\nSpeedup Analysis:")
            fwd_speedup = regular_time / fourier_results['forward_only']
            print(f"  Forward-only speedup:  {fwd_speedup:.2f}x faster")

            full_speedup = regular_time / fourier_results['forward_plus_ifft']
            print(f"  Full (incl IFFT):      {full_speedup:.2f}x faster")

            # Break-even point
            if fourier_results['forward_only'] < regular_time:
                break_even = pre_fft_time / (regular_time - fourier_results['forward_only'])
                print(f"\n  Break-even: Pre-FFT pays off after {break_even:.1f} forward passes")
            else:
                print(f"\n  Warning: Fourier version not faster for this config")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key observations:
- Fourier-space forward pass eliminates all FFT operations
- Only element-wise multiplications in the forward pass
- Pre-FFT cost amortizes over many forward passes (training epochs)
- For training: Pre-FFT once per epoch, all forward/backward passes benefit
- For inference: Pre-FFT in data loading, then fast inference

Recommended workflow:
1. Pre-FFT your dataset once (can save to disk)
2. Store kernels in Fourier domain
3. Forward pass: pure element-wise operations
4. IFFT only at the very end for loss/visualization
""")


if __name__ == '__main__':
    main()
