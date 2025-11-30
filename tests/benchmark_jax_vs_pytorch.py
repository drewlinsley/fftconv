#!/usr/bin/env python3
"""
Benchmark JAX vs PyTorch for 3D ConvSSM parallel scan.

This compares:
1. PyTorch sequential ConvSSM
2. PyTorch parallel scan (our implementation)
3. JAX sequential ConvSSM
4. JAX parallel scan (using lax.associative_scan)

Usage:
    python tests/benchmark_jax_vs_pytorch.py
"""

import time
import numpy as np

# =============================================================================
# PyTorch setup
# =============================================================================
import torch

from flashfftconv import FlashFFTConv3D, ConvSSMParallelScan3D

# =============================================================================
# JAX setup
# =============================================================================
import jax
import jax.numpy as jnp
from jax import random

from flashfftconv.conv_nd_jax import (
    FlashFFTConv3DJAX,
    ConvSSMParallelScan3DJAX,
    convssm_sequential_3d_jit,
    convssm_parallel_3d_jit,
)


def benchmark_pytorch(T, B, C, D, H, W, K, warmup=5, rep=20):
    """Benchmark PyTorch implementations."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create data
    torch.manual_seed(42)
    x_seq = torch.randn(T, B, C, D, H, W, device=device) * 0.1
    A_kernel = torch.randn(C, K, K, K, device=device) * 0.1
    B_kernel = torch.randn(C, K, K, K, device=device) * 0.1

    # Apply decay for stability
    decay = torch.exp(-0.3 * torch.arange(K, device=device))
    decay_3d = decay[:, None, None] * decay[None, :, None] * decay[None, None, :]
    A_kernel = A_kernel * decay_3d
    B_kernel = B_kernel * decay_3d

    results = {}

    # --- PyTorch Sequential ---
    fftconv = FlashFFTConv3D(D, H, W).to(device)

    def run_pytorch_sequential():
        h = torch.zeros(B, C, D, H, W, device=device)
        for t in range(T):
            h = fftconv(h, A_kernel) + fftconv(x_seq[t], B_kernel)
        return h

    # Warmup
    for _ in range(warmup):
        run_pytorch_sequential()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(rep):
        h = run_pytorch_sequential()
    torch.cuda.synchronize()
    results['pytorch_sequential'] = (time.perf_counter() - start) / rep * 1000

    # --- PyTorch Parallel Scan ---
    scanner = ConvSSMParallelScan3D(D, H, W).to(device)

    def run_pytorch_parallel():
        return scanner(x_seq, A_kernel, B_kernel, return_all=False)

    # Warmup
    for _ in range(warmup):
        run_pytorch_parallel()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(rep):
        h = run_pytorch_parallel()
    torch.cuda.synchronize()
    results['pytorch_parallel'] = (time.perf_counter() - start) / rep * 1000

    return results


def benchmark_jax(T, B, C, D, H, W, K, warmup=5, rep=20):
    """Benchmark JAX implementations."""
    results = {}

    # Create data
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

    # --- JAX Sequential ---
    # Warmup (also compiles)
    for _ in range(warmup):
        h = convssm_sequential_3d_jit(x_seq, A_kernel, B_kernel, spatial_size)
        h.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(rep):
        h = convssm_sequential_3d_jit(x_seq, A_kernel, B_kernel, spatial_size)
        h.block_until_ready()
    results['jax_sequential'] = (time.perf_counter() - start) / rep * 1000

    # --- JAX Parallel Scan ---
    # Warmup (also compiles)
    for _ in range(warmup):
        h = convssm_parallel_3d_jit(x_seq, A_kernel, B_kernel, spatial_size, False)
        h.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(rep):
        h = convssm_parallel_3d_jit(x_seq, A_kernel, B_kernel, spatial_size, False)
        h.block_until_ready()
    results['jax_parallel'] = (time.perf_counter() - start) / rep * 1000

    return results


def main():
    print("=" * 70)
    print("JAX vs PyTorch Benchmark: 3D ConvSSM")
    print("=" * 70)

    # Check devices
    print(f"\nPyTorch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"JAX devices: {jax.devices()}")

    # Test configurations
    configs = [
        # (T, B, C, D, H, W, K)
        (10, 2, 32, 8, 16, 16, 3),    # Small
        (50, 2, 32, 8, 16, 16, 3),    # Medium T
        (100, 2, 32, 8, 16, 16, 3),   # Large T
        (100, 4, 64, 8, 16, 16, 3),   # Large T + larger batch/channels
        (100, 2, 32, 16, 32, 32, 3),  # Large spatial
    ]

    for T, B, C, D, H, W, K in configs:
        print(f"\n{'='*70}")
        print(f"Config: T={T}, B={B}, C={C}, D={D}, H={H}, W={W}, K={K}")
        print(f"{'='*70}")

        try:
            # PyTorch benchmarks
            pytorch_results = benchmark_pytorch(T, B, C, D, H, W, K)
            print(f"\nPyTorch:")
            print(f"  Sequential: {pytorch_results['pytorch_sequential']:.2f} ms")
            print(f"  Parallel:   {pytorch_results['pytorch_parallel']:.2f} ms")

            # JAX benchmarks
            jax_results = benchmark_jax(T, B, C, D, H, W, K)
            print(f"\nJAX:")
            print(f"  Sequential: {jax_results['jax_sequential']:.2f} ms")
            print(f"  Parallel:   {jax_results['jax_parallel']:.2f} ms")

            # Summary
            print(f"\nSpeedup Analysis:")
            fastest = min(
                pytorch_results['pytorch_sequential'],
                pytorch_results['pytorch_parallel'],
                jax_results['jax_sequential'],
                jax_results['jax_parallel'],
            )

            all_results = [
                ('PyTorch Sequential', pytorch_results['pytorch_sequential']),
                ('PyTorch Parallel', pytorch_results['pytorch_parallel']),
                ('JAX Sequential', jax_results['jax_sequential']),
                ('JAX Parallel', jax_results['jax_parallel']),
            ]

            for name, t in sorted(all_results, key=lambda x: x[1]):
                if t == fastest:
                    print(f"  {name}: {t:.2f} ms  ← FASTEST")
                else:
                    print(f"  {name}: {t:.2f} ms  ({t/fastest:.2f}x slower)")

            # JAX parallel vs PyTorch parallel
            speedup = pytorch_results['pytorch_parallel'] / jax_results['jax_parallel']
            print(f"\nJAX parallel vs PyTorch parallel: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

        except Exception as e:
            print(f"  Error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key observations:
- JAX's lax.associative_scan provides native O(log T) parallel scan
- PyTorch parallel scan has Python loop overhead
- JAX XLA compiler can fuse operations better
- For long sequences (T >> 100), JAX parallel should show more advantage
""")


if __name__ == '__main__':
    main()
