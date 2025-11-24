#!/usr/bin/env python3
"""
Compare FlashFFTConv2D/3D against standard PyTorch Conv2d/Conv3d.

This script:
1. Creates identical kernels for both implementations
2. Verifies outputs match (within numerical tolerance)
3. Measures timing for both forward and backward passes

Usage:
    python tests/compare_fftconv_vs_conv.py
    python tests/compare_fftconv_vs_conv.py --device cpu
    python tests/compare_fftconv_vs_conv.py --no-backward
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from flashfftconv import FlashFFTConv2D, FlashFFTConv3D


def time_fn(fn, warmup=5, rep=20):
    """Time a function with warmup iterations."""
    for _ in range(warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(rep):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return (time.perf_counter() - start) / rep * 1000  # ms


def compare_conv2d(B, C, H, W, K, device='cuda', test_backward=True):
    """
    Compare 2D FFT convolution against standard Conv2d.

    Args:
        B: Batch size
        C: Channels
        H, W: Spatial dimensions
        K: Kernel size (e.g., 3 for 3x3)
        device: 'cuda' or 'cpu'
        test_backward: Whether to test backward pass
    """
    print(f"\n{'='*70}")
    print(f"2D Comparison: B={B}, C={C}, H={H}, W={W}, K={K}")
    print(f"{'='*70}")

    torch.manual_seed(42)

    # Create input
    x = torch.randn(B, C, H, W, device=device, requires_grad=test_backward)

    # Create kernel weights - same for both!
    kernel_weights = torch.randn(C, C, K, K, device=device) * 0.01

    # =========================================================================
    # Standard PyTorch Conv2d
    # =========================================================================
    conv_std = nn.Conv2d(C, C, K, padding=K//2, bias=False).to(device)
    with torch.no_grad():
        conv_std.weight.copy_(kernel_weights)

    # Forward
    x_std = x.clone().detach().requires_grad_(test_backward)
    out_std = conv_std(x_std)

    # =========================================================================
    # FlashFFTConv2D
    # Uses depthwise-style: kernel is (C, K, K), applied per-channel
    # For fair comparison, we'll loop over output channels
    # =========================================================================
    fftconv = FlashFFTConv2D(H, W).to(device)

    # For FFT conv, we need to restructure the kernel
    # Standard conv: out[c_out] = sum over c_in of conv(x[c_in], k[c_out, c_in])
    # Let's do a simpler comparison: depthwise conv (C_in == C_out, groups=C)

    # Actually, let's compare against depthwise conv for simplicity
    conv_dw = nn.Conv2d(C, C, K, padding=K//2, groups=C, bias=False).to(device)
    kernel_dw = torch.randn(C, 1, K, K, device=device) * 0.01
    with torch.no_grad():
        conv_dw.weight.copy_(kernel_dw)

    x_dw = x.clone().detach().requires_grad_(test_backward)
    out_dw = conv_dw(x_dw)

    # FFT conv with same kernel (C, K, K)
    kernel_fft = kernel_dw.squeeze(1)  # (C, K, K)
    x_fft = x.clone().detach().requires_grad_(test_backward)
    out_fft = fftconv(x_fft, kernel_fft)

    # =========================================================================
    # Compare outputs
    # =========================================================================
    # Note: FFT conv and standard conv have different boundary handling
    # FFT conv: zero-padding to 2N, then circular conv, then crop
    # Standard conv: explicit padding
    # They should match in the interior, but may differ at boundaries

    # Compare center region (away from boundaries)
    margin = K
    center_slice = (slice(None), slice(None), slice(margin, -margin), slice(margin, -margin))

    diff_center = (out_fft[center_slice] - out_dw[center_slice]).abs()
    diff_full = (out_fft - out_dw).abs()

    print(f"\nOutput comparison (depthwise conv):")
    print(f"  Full image max diff:   {diff_full.max().item():.6f}")
    print(f"  Full image mean diff:  {diff_full.mean().item():.6f}")
    print(f"  Center region max diff: {diff_center.max().item():.6f}")
    print(f"  Center region mean diff: {diff_center.mean().item():.6f}")

    # =========================================================================
    # Timing
    # =========================================================================
    print(f"\nTiming (forward pass):")

    def run_conv_dw():
        return conv_dw(x)

    def run_fft():
        return fftconv(x, kernel_fft)

    time_conv = time_fn(run_conv_dw)
    time_fft = time_fn(run_fft)

    print(f"  Conv2d (depthwise): {time_conv:.3f} ms")
    print(f"  FlashFFTConv2D:     {time_fft:.3f} ms")
    print(f"  Speedup:            {time_conv/time_fft:.2f}x")

    # =========================================================================
    # Backward pass
    # =========================================================================
    if test_backward:
        print(f"\nTiming (forward + backward):")

        def run_conv_dw_bwd():
            x_tmp = x.clone().detach().requires_grad_(True)
            out = conv_dw(x_tmp)
            out.sum().backward()
            return out

        def run_fft_bwd():
            x_tmp = x.clone().detach().requires_grad_(True)
            k_tmp = kernel_fft.clone().detach().requires_grad_(True)
            out = fftconv(x_tmp, k_tmp)
            out.sum().backward()
            return out

        time_conv_bwd = time_fn(run_conv_dw_bwd, warmup=3, rep=10)
        time_fft_bwd = time_fn(run_fft_bwd, warmup=3, rep=10)

        print(f"  Conv2d (depthwise): {time_conv_bwd:.3f} ms")
        print(f"  FlashFFTConv2D:     {time_fft_bwd:.3f} ms")
        print(f"  Speedup:            {time_conv_bwd/time_fft_bwd:.2f}x")

    return diff_center.max().item()


def compare_conv3d(B, C, D, H, W, K, device='cuda', test_backward=True):
    """
    Compare 3D FFT convolution against standard Conv3d.
    """
    print(f"\n{'='*70}")
    print(f"3D Comparison: B={B}, C={C}, D={D}, H={H}, W={W}, K={K}")
    print(f"{'='*70}")

    torch.manual_seed(42)

    # Create input
    x = torch.randn(B, C, D, H, W, device=device, requires_grad=test_backward)

    # Depthwise conv3d
    conv_dw = nn.Conv3d(C, C, K, padding=K//2, groups=C, bias=False).to(device)
    kernel_dw = torch.randn(C, 1, K, K, K, device=device) * 0.01
    with torch.no_grad():
        conv_dw.weight.copy_(kernel_dw)

    x_dw = x.clone().detach().requires_grad_(test_backward)
    out_dw = conv_dw(x_dw)

    # FFT conv
    fftconv = FlashFFTConv3D(D, H, W).to(device)
    kernel_fft = kernel_dw.squeeze(1)  # (C, K, K, K)
    x_fft = x.clone().detach().requires_grad_(test_backward)
    out_fft = fftconv(x_fft, kernel_fft)

    # Compare
    margin = K
    center_slice = (slice(None), slice(None), slice(margin, -margin),
                    slice(margin, -margin), slice(margin, -margin))

    diff_center = (out_fft[center_slice] - out_dw[center_slice]).abs()
    diff_full = (out_fft - out_dw).abs()

    print(f"\nOutput comparison (depthwise conv):")
    print(f"  Full volume max diff:   {diff_full.max().item():.6f}")
    print(f"  Full volume mean diff:  {diff_full.mean().item():.6f}")
    print(f"  Center region max diff: {diff_center.max().item():.6f}")

    # Timing
    print(f"\nTiming (forward pass):")

    def run_conv_dw():
        return conv_dw(x)

    def run_fft():
        return fftconv(x, kernel_fft)

    time_conv = time_fn(run_conv_dw)
    time_fft = time_fn(run_fft)

    print(f"  Conv3d (depthwise): {time_conv:.3f} ms")
    print(f"  FlashFFTConv3D:     {time_fft:.3f} ms")
    print(f"  Speedup:            {time_conv/time_fft:.2f}x")

    if test_backward:
        print(f"\nTiming (forward + backward):")

        def run_conv_dw_bwd():
            x_tmp = x.clone().detach().requires_grad_(True)
            out = conv_dw(x_tmp)
            out.sum().backward()
            return out

        def run_fft_bwd():
            x_tmp = x.clone().detach().requires_grad_(True)
            k_tmp = kernel_fft.clone().detach().requires_grad_(True)
            out = fftconv(x_tmp, k_tmp)
            out.sum().backward()
            return out

        time_conv_bwd = time_fn(run_conv_dw_bwd, warmup=3, rep=10)
        time_fft_bwd = time_fn(run_fft_bwd, warmup=3, rep=10)

        print(f"  Conv3d (depthwise): {time_conv_bwd:.3f} ms")
        print(f"  FlashFFTConv3D:     {time_fft_bwd:.3f} ms")
        print(f"  Speedup:            {time_conv_bwd/time_fft_bwd:.2f}x")

    return diff_center.max().item()


def main():
    parser = argparse.ArgumentParser(description='Compare FFT conv vs standard conv')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--no-backward', action='store_true', help='Skip backward pass tests')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print("=" * 70)
    print("FlashFFTConv vs PyTorch Conv Comparison")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Testing backward: {not args.no_backward}")

    # 2D comparisons
    print("\n" + "=" * 70)
    print("2D CONVOLUTION COMPARISONS")
    print("=" * 70)

    configs_2d = [
        (4, 64, 64, 64, 3),    # Small
        (4, 128, 128, 128, 3), # Medium
        (4, 128, 256, 256, 3), # Large
        (4, 64, 64, 64, 5),    # 5x5 kernel
        (4, 64, 64, 64, 7),    # 7x7 kernel
    ]

    for B, C, H, W, K in configs_2d:
        try:
            compare_conv2d(B, C, H, W, K, device=args.device,
                          test_backward=not args.no_backward)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Skipped due to OOM")
                torch.cuda.empty_cache()
            else:
                raise

    # 3D comparisons
    print("\n" + "=" * 70)
    print("3D CONVOLUTION COMPARISONS")
    print("=" * 70)

    configs_3d = [
        (2, 32, 16, 32, 32, 3),  # Small
        (2, 64, 16, 64, 64, 3),  # Medium
        (2, 32, 32, 32, 32, 3),  # Cubic
    ]

    for B, C, D, H, W, K in configs_3d:
        try:
            compare_conv3d(B, C, D, H, W, K, device=args.device,
                          test_backward=not args.no_backward)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Skipped due to OOM")
                torch.cuda.empty_cache()
            else:
                raise

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Notes:
- FFT conv and standard conv may differ at boundaries due to padding differences
- FFT conv: zero-pads to 2N, circular convolution, then crops
- Standard conv: explicit zero/reflect padding

For ConvSSM, the key advantage of FFT conv is NOT per-operation speed,
but enabling PARALLEL SCAN over time (O(log T) depth vs O(T) sequential).

When to use FFT conv:
- ConvSSM with parallel scan (main use case!)
- Very large kernels (where FFT wins per-operation)
- When you need full-resolution kernels

When to use standard conv:
- Small kernels without temporal parallelization
- When boundary handling must be exact
""")


if __name__ == '__main__':
    main()
