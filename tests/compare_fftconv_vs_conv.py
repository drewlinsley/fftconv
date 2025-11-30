#!/usr/bin/env python3
"""
Compare FlashFFTConv2D/3D against standard PyTorch Conv2d/Conv3d.

FAIR COMPARISON: Both methods use the SAME kernel size (e.g., 3x3).
FFT conv zero-pads the small kernel to FFT size internally.

This script:
1. Creates identical kernels for both implementations
2. Verifies outputs match (within numerical tolerance)
3. Measures timing for both forward and backward passes
4. Visualizes outputs and per-pixel differences

Usage:
    python tests/compare_fftconv_vs_conv.py
    python tests/compare_fftconv_vs_conv.py --device cpu
    python tests/compare_fftconv_vs_conv.py --visualize
"""

import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from flashfftconv import FlashFFTConv2D, FlashFFTConv3D, ConvSSMParallelScan2D


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


def visualize_comparison(out_conv, out_fft, title="2D Comparison", save_path=None):
    """Visualize outputs and per-pixel differences."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not installed, skipping visualization]")
        return

    # Take first batch, first channel
    conv_img = out_conv[0, 0].detach().cpu().float().numpy()
    fft_img = out_fft[0, 0].detach().cpu().float().numpy()
    diff_img = np.abs(conv_img - fft_img)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Conv output
    im0 = axes[0].imshow(conv_img, cmap='viridis')
    axes[0].set_title('Conv2d Output')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # FFT conv output
    im1 = axes[1].imshow(fft_img, cmap='viridis')
    axes[1].set_title('FFTConv Output')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Absolute difference
    im2 = axes[2].imshow(diff_img, cmap='hot')
    axes[2].set_title(f'|Diff| (max={diff_img.max():.2e})')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Histogram of differences
    axes[3].hist(diff_img.flatten(), bins=50, color='steelblue', edgecolor='black')
    axes[3].set_title('Diff Histogram')
    axes[3].set_xlabel('Absolute Difference')
    axes[3].set_ylabel('Count')
    axes[3].set_yscale('log')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_conv2d(B, C, H, W, K, device='cuda', test_backward=True, visualize=False):
    """
    Compare 2D FFT convolution against standard Conv2d.

    FAIR COMPARISON: Both use the same K×K kernel size.

    Args:
        B: Batch size
        C: Channels
        H, W: Spatial dimensions
        K: Kernel size (e.g., 3 for 3x3) - SAME for both methods
        device: 'cuda' or 'cpu'
        test_backward: Whether to test backward pass
        visualize: Whether to show visualization
    """
    print(f"\n{'='*70}")
    print(f"2D Comparison: B={B}, C={C}, H={H}, W={W}, K={K}×{K}")
    print(f"{'='*70}")

    torch.manual_seed(42)

    # Create input
    x = torch.randn(B, C, H, W, device=device)

    # =========================================================================
    # Create IDENTICAL kernel for both methods
    # =========================================================================
    # Depthwise conv kernel: (C, 1, K, K) for groups=C
    kernel_weights = torch.randn(C, 1, K, K, device=device) * 0.1

    # =========================================================================
    # Standard PyTorch Conv2d (depthwise)
    # =========================================================================
    conv_dw = nn.Conv2d(C, C, K, padding=K//2, groups=C, bias=False).to(device)
    with torch.no_grad():
        conv_dw.weight.copy_(kernel_weights)

    x_conv = x.clone().detach()
    out_conv = conv_dw(x_conv)

    # =========================================================================
    # FlashFFTConv2D with SAME small kernel
    # =========================================================================
    fftconv = FlashFFTConv2D(H, W).to(device)

    # FFT conv kernel: (C, K, K) - same weights, just reshaped
    kernel_fft = kernel_weights.squeeze(1)  # (C, K, K)

    x_fft = x.clone().detach()
    out_fft = fftconv(x_fft, kernel_fft)

    # =========================================================================
    # Compare outputs
    # =========================================================================
    # Note: Boundary handling differs:
    # - Conv2d: explicit zero padding
    # - FFT conv: circular (but we pad to 2N, so effectively linear in center)

    margin = K  # Ignore boundary region
    if H > 2*margin and W > 2*margin:
        center_slice = (slice(None), slice(None), slice(margin, -margin), slice(margin, -margin))
        diff_center = (out_fft[center_slice] - out_conv[center_slice]).abs()
        center_max = diff_center.max().item()
        center_mean = diff_center.mean().item()
    else:
        center_max = center_mean = float('nan')

    diff_full = (out_fft - out_conv).abs()

    print(f"\nOutput comparison (SAME {K}×{K} kernel):")
    print(f"  Full image max diff:    {diff_full.max().item():.6f}")
    print(f"  Full image mean diff:   {diff_full.mean().item():.6f}")
    print(f"  Center region max diff: {center_max:.6f}")
    print(f"  Center region mean diff: {center_mean:.6f}")

    # =========================================================================
    # Timing - FAIR comparison with same kernel size
    # =========================================================================
    print(f"\nTiming (forward pass, K={K}×{K} kernel):")

    def run_conv():
        return conv_dw(x)

    def run_fft():
        return fftconv(x, kernel_fft)

    time_conv = time_fn(run_conv)
    time_fft = time_fn(run_fft)

    print(f"  Conv2d (depthwise, K={K}):  {time_conv:.3f} ms")
    print(f"  FlashFFTConv2D (K={K}):     {time_fft:.3f} ms  [includes FFT + multiply + IFFT]")

    if time_fft < time_conv:
        print(f"  FFT conv is {time_conv/time_fft:.2f}x FASTER")
    else:
        print(f"  Conv2d is {time_fft/time_conv:.2f}x FASTER")

    # =========================================================================
    # Backward pass timing
    # =========================================================================
    if test_backward:
        print(f"\nTiming (forward + backward, K={K}×{K} kernel):")

        def run_conv_bwd():
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

        time_conv_bwd = time_fn(run_conv_bwd, warmup=3, rep=10)
        time_fft_bwd = time_fn(run_fft_bwd, warmup=3, rep=10)

        print(f"  Conv2d (depthwise):     {time_conv_bwd:.3f} ms")
        print(f"  FlashFFTConv2D:         {time_fft_bwd:.3f} ms")

        if time_fft_bwd < time_conv_bwd:
            print(f"  FFT conv is {time_conv_bwd/time_fft_bwd:.2f}x FASTER")
        else:
            print(f"  Conv2d is {time_fft_bwd/time_conv_bwd:.2f}x FASTER")

    # =========================================================================
    # Visualization
    # =========================================================================
    if visualize:
        visualize_comparison(
            out_conv, out_fft,
            title=f"Conv2d vs FFTConv2D (K={K}, H={H}, W={W})",
            save_path=f"compare_2d_K{K}_H{H}_W{W}.png"
        )

    return diff_full.max().item()


def compare_conv3d(B, C, D, H, W, K, device='cuda', test_backward=True, visualize=False):
    """
    Compare 3D FFT convolution against standard Conv3d.

    FAIR COMPARISON: Both use the same K×K×K kernel size.
    """
    print(f"\n{'='*70}")
    print(f"3D Comparison: B={B}, C={C}, D={D}, H={H}, W={W}, K={K}×{K}×{K}")
    print(f"{'='*70}")

    torch.manual_seed(42)

    # Create input
    x = torch.randn(B, C, D, H, W, device=device)

    # Depthwise conv3d kernel
    kernel_weights = torch.randn(C, 1, K, K, K, device=device) * 0.1

    # Standard Conv3d
    conv_dw = nn.Conv3d(C, C, K, padding=K//2, groups=C, bias=False).to(device)
    with torch.no_grad():
        conv_dw.weight.copy_(kernel_weights)

    out_conv = conv_dw(x)

    # FFT conv with same kernel
    fftconv = FlashFFTConv3D(D, H, W).to(device)
    kernel_fft = kernel_weights.squeeze(1)  # (C, K, K, K)
    out_fft = fftconv(x, kernel_fft)

    # Compare
    margin = K
    if D > 2*margin and H > 2*margin and W > 2*margin:
        center_slice = (slice(None), slice(None),
                        slice(margin, -margin), slice(margin, -margin), slice(margin, -margin))
        diff_center = (out_fft[center_slice] - out_conv[center_slice]).abs()
        center_max = diff_center.max().item()
    else:
        center_max = float('nan')

    diff_full = (out_fft - out_conv).abs()

    print(f"\nOutput comparison (SAME {K}×{K}×{K} kernel):")
    print(f"  Full volume max diff:   {diff_full.max().item():.6f}")
    print(f"  Full volume mean diff:  {diff_full.mean().item():.6f}")
    print(f"  Center region max diff: {center_max:.6f}")

    # Timing
    print(f"\nTiming (forward pass, K={K}×{K}×{K} kernel):")

    def run_conv():
        return conv_dw(x)

    def run_fft():
        return fftconv(x, kernel_fft)

    time_conv = time_fn(run_conv)
    time_fft = time_fn(run_fft)

    print(f"  Conv3d (depthwise, K={K}):  {time_conv:.3f} ms")
    print(f"  FlashFFTConv3D (K={K}):     {time_fft:.3f} ms  [includes FFT + multiply + IFFT]")

    if time_fft < time_conv:
        print(f"  FFT conv is {time_conv/time_fft:.2f}x FASTER")
    else:
        print(f"  Conv3d is {time_fft/time_conv:.2f}x FASTER")

    if test_backward:
        print(f"\nTiming (forward + backward):")

        def run_conv_bwd():
            x_tmp = x.clone().detach().requires_grad_(True)
            out = conv_dw(x_tmp)
            out.sum().backward()

        def run_fft_bwd():
            x_tmp = x.clone().detach().requires_grad_(True)
            k_tmp = kernel_fft.clone().detach().requires_grad_(True)
            out = fftconv(x_tmp, k_tmp)
            out.sum().backward()

        time_conv_bwd = time_fn(run_conv_bwd, warmup=3, rep=10)
        time_fft_bwd = time_fn(run_fft_bwd, warmup=3, rep=10)

        print(f"  Conv3d (depthwise):     {time_conv_bwd:.3f} ms")
        print(f"  FlashFFTConv3D:         {time_fft_bwd:.3f} ms")

        if time_fft_bwd < time_conv_bwd:
            print(f"  FFT conv is {time_conv_bwd/time_fft_bwd:.2f}x FASTER")
        else:
            print(f"  Conv3d is {time_fft_bwd/time_conv_bwd:.2f}x FASTER")

    # Visualization (middle slice of 3D volume)
    if visualize:
        mid_d = D // 2
        visualize_comparison(
            out_conv[:, :, mid_d], out_fft[:, :, mid_d],
            title=f"Conv3d vs FFTConv3D (K={K}, slice D={mid_d})",
            save_path=f"compare_3d_K{K}_D{D}_H{H}_W{W}.png"
        )

    return diff_full.max().item()


def compare_multi_timestep(B, C, H, W, K, T, device='cuda', visualize=False):
    """
    Compare ConvSSM over T timesteps: h_t = A ★ h_{t-1} + B ★ x_t

    This is the KEY use case for FFT conv. Shows:
    1. Sequential ConvSSM with standard Conv2d
    2. Sequential ConvSSM with FFT conv
    3. Why FFT enables parallel scan (kernel stays constant in FFT space)

    Args:
        T: Number of timesteps
    """
    print(f"\n{'='*70}")
    print(f"MULTI-TIMESTEP ConvSSM: T={T}, B={B}, C={C}, H={H}, W={W}, K={K}×{K}")
    print(f"{'='*70}")
    print(f"Recurrence: h_t = A ★ h_{{t-1}} + B ★ x_t  (★ = convolution)")

    torch.manual_seed(42)

    # Input sequence: T frames
    x_seq = torch.randn(T, B, C, H, W, device=device) * 0.1

    # =========================================================================
    # Setup kernels (identical for both methods)
    # =========================================================================
    A_weights = torch.randn(C, 1, K, K, device=device) * 0.1
    B_weights = torch.randn(C, 1, K, K, device=device) * 0.1

    # Standard conv setup
    conv_A = nn.Conv2d(C, C, K, padding=K//2, groups=C, bias=False).to(device)
    conv_B = nn.Conv2d(C, C, K, padding=K//2, groups=C, bias=False).to(device)
    with torch.no_grad():
        conv_A.weight.copy_(A_weights)
        conv_B.weight.copy_(B_weights)

    # FFT conv setup
    fftconv = FlashFFTConv2D(H, W).to(device)
    A_fft = A_weights.squeeze(1)  # (C, K, K)
    B_fft = B_weights.squeeze(1)  # (C, K, K)

    # =========================================================================
    # Method 1: Sequential ConvSSM with standard Conv2d
    # =========================================================================
    def run_conv_sequential():
        h = torch.zeros(B, C, H, W, device=device)
        for t in range(T):
            h = conv_A(h) + conv_B(x_seq[t])
        return h

    # Warmup and time
    for _ in range(3):
        run_conv_sequential()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        out_conv = run_conv_sequential()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_conv = (time.perf_counter() - start) / 10 * 1000

    # =========================================================================
    # Method 2: Sequential ConvSSM with FFT conv
    # =========================================================================
    def run_fft_sequential():
        h = torch.zeros(B, C, H, W, device=device)
        for t in range(T):
            h = fftconv(h, A_fft) + fftconv(x_seq[t], B_fft)
        return h

    # Warmup and time
    for _ in range(3):
        run_fft_sequential()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        out_fft = run_fft_sequential()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_fft = (time.perf_counter() - start) / 10 * 1000

    # =========================================================================
    # Method 3: TRUE parallel scan using O(log T) algorithm
    # =========================================================================
    scanner = ConvSSMParallelScan2D(H, W).to(device)

    def run_true_parallel_scan():
        """
        True O(log T) parallel scan using associative property.

        The recurrence h_t = A * h_{t-1} + B * x_t can be computed in O(log T)
        parallel depth using the associative combination rule:
        (a1, s1) ⊕ (a2, s2) = (a1 * a2, s1 * a2 + s2)

        Complexity:
        - FFT all inputs: O(T * N log N) - parallelizable across T
        - Parallel scan: O(log T) depth, O(T * N) work
        - IFFT result: O(N log N)
        """
        # Get only final state for fair comparison
        h_final = scanner(x_seq, A_fft, B_fft, return_all=False)
        return h_final

    # Warmup and time
    for _ in range(3):
        run_true_parallel_scan()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        out_parallel = run_true_parallel_scan()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_parallel = (time.perf_counter() - start) / 10 * 1000

    # =========================================================================
    # Method 4: Old sequential FFT (for comparison)
    # =========================================================================
    def run_fft_sequential_in_freq():
        """Sequential scan staying in frequency domain (no per-step IFFT)."""
        fft_size = (2 * H, 2 * W)
        x_seq_f = torch.fft.rfftn(x_seq.float(), s=fft_size, dim=(-2, -1))
        A_f = torch.fft.rfftn(A_fft.float(), s=fft_size, dim=(-2, -1))
        B_f = torch.fft.rfftn(B_fft.float(), s=fft_size, dim=(-2, -1))

        h_f = torch.zeros(B, C, *A_f.shape[-2:], device=device, dtype=torch.complex64)
        for t in range(T):
            h_f = h_f * A_f.unsqueeze(0) + x_seq_f[t] * B_f.unsqueeze(0)

        h = torch.fft.irfftn(h_f, s=fft_size, dim=(-2, -1))[..., :H, :W]
        return h

    # Warmup and time
    for _ in range(3):
        run_fft_sequential_in_freq()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        out_seq_freq = run_fft_sequential_in_freq()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_seq_freq = (time.perf_counter() - start) / 10 * 1000

    # =========================================================================
    # Results
    # =========================================================================
    print(f"\nTiming for {T} timesteps:")
    print(f"  Conv2d sequential:      {time_conv:.3f} ms  ({T}× conv per step)")
    print(f"  FFTConv sequential:     {time_fft:.3f} ms  ({T}× FFT+mul+IFFT per step)")
    print(f"  FFT seq in freq:        {time_seq_freq:.3f} ms  (FFT once, {T}× mul, IFFT once)")
    print(f"  TRUE parallel scan:     {time_parallel:.3f} ms  (O(log {T})={int(math.ceil(math.log2(T)))} steps)")

    print(f"\nPer-timestep cost:")
    print(f"  Conv2d:       {time_conv/T:.3f} ms/step")
    print(f"  FFTConv:      {time_fft/T:.3f} ms/step")
    print(f"  Seq in freq:  {time_seq_freq/T:.3f} ms/step")
    print(f"  Parallel:     {time_parallel/T:.3f} ms/step (amortized)")

    # Compare outputs
    diff_conv_fft = (out_conv - out_fft).abs()
    diff_conv_parallel = (out_conv - out_parallel).abs()
    diff_conv_seq_freq = (out_conv - out_seq_freq).abs()

    print(f"\nOutput comparison (vs Conv2d reference):")
    print(f"  FFTConv sequential max diff:   {diff_conv_fft.max().item():.6f}")
    print(f"  FFT seq in freq max diff:      {diff_conv_seq_freq.max().item():.6f}")
    print(f"  TRUE parallel scan max diff:   {diff_conv_parallel.max().item():.6f}")

    # Speedup analysis
    print(f"\nSpeedup analysis:")
    fastest = min(time_conv, time_fft, time_seq_freq, time_parallel)
    methods = [
        ("Conv2d sequential", time_conv),
        ("FFTConv sequential", time_fft),
        ("FFT seq in freq", time_seq_freq),
        ("TRUE parallel scan", time_parallel),
    ]
    for name, t in methods:
        if t == fastest:
            print(f"  {name}: {t:.3f} ms  ← FASTEST")
        else:
            print(f"  {name}: {t:.3f} ms  ({t/fastest:.2f}x slower)")

    print(f"""
Complexity analysis for T={T} timesteps:
- Conv2d sequential: O(T·N·K²) = O({T}·{H*W}·{K*K}) sequential ops
- FFT seq in freq:   O(T·N) sequential + O(N log N) FFT overhead
- TRUE parallel scan: O(log T · N) = O({int(math.ceil(math.log2(T)))}·{H*W}) parallel depth

Key insight: In FFT space, A^T is still N values (element-wise power)
In pixel space, A^T would be a ({T}·{K})×({T}·{K}) = {T*(K-1)+1}×{T*(K-1)+1} kernel!

NOTE: Current parallel scan uses pure PyTorch. Could be faster with:
- Triton kernel (fused operations)
- Mamba's selective_scan_cuda
- torch.compile optimization
""")

    # Visualization
    if visualize:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Row 1: Final outputs
            conv_img = out_conv[0, 0].detach().cpu().float().numpy()
            fft_img = out_fft[0, 0].detach().cpu().float().numpy()
            parallel_img = out_parallel[0, 0].detach().cpu().float().numpy()

            im0 = axes[0, 0].imshow(conv_img, cmap='viridis')
            axes[0, 0].set_title(f'Conv2d (T={T})')
            axes[0, 0].axis('off')
            plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

            im1 = axes[0, 1].imshow(fft_img, cmap='viridis')
            axes[0, 1].set_title(f'FFTConv (T={T})')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

            im2 = axes[0, 2].imshow(parallel_img, cmap='viridis')
            axes[0, 2].set_title(f'Parallel Scan (T={T})')
            axes[0, 2].axis('off')
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

            # Row 2: Differences
            diff1 = np.abs(conv_img - fft_img)
            diff2 = np.abs(conv_img - parallel_img)

            im3 = axes[1, 0].imshow(diff1, cmap='hot')
            axes[1, 0].set_title(f'|Conv - FFT| (max={diff1.max():.2e})')
            axes[1, 0].axis('off')
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

            im4 = axes[1, 1].imshow(diff2, cmap='hot')
            axes[1, 1].set_title(f'|Conv - Parallel| (max={diff2.max():.2e})')
            axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

            # Timing bar chart
            methods = ['Conv2d', 'FFTConv', 'Parallel']
            times = [time_conv, time_fft, time_parallel]
            colors = ['steelblue', 'darkorange', 'green']
            axes[1, 2].bar(methods, times, color=colors)
            axes[1, 2].set_ylabel('Time (ms)')
            axes[1, 2].set_title(f'Timing Comparison (T={T})')
            for i, t in enumerate(times):
                axes[1, 2].text(i, t + 0.5, f'{t:.1f}ms', ha='center')

            plt.suptitle(f'ConvSSM Multi-Timestep Comparison (T={T}, K={K}, H={H}, W={W})')
            plt.tight_layout()
            plt.savefig(f'compare_timestep_T{T}_K{K}_H{H}_W{W}.png', dpi=150, bbox_inches='tight')
            print(f"  Saved visualization to compare_timestep_T{T}_K{K}_H{H}_W{W}.png")
            plt.close()
        except ImportError:
            print("  [matplotlib not installed, skipping visualization]")

    return time_conv, time_fft, time_parallel


def main():
    parser = argparse.ArgumentParser(description='Compare FFT conv vs standard conv')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--no-backward', action='store_true', help='Skip backward pass tests')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print("=" * 70)
    print("FlashFFTConv vs PyTorch Conv - FAIR Comparison")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Testing backward: {not args.no_backward}")
    print(f"Visualization: {args.visualize}")
    print("\nNOTE: Both methods use the SAME kernel size for fair comparison.")
    print("FFT timing includes: FFT(input) + FFT(kernel) + multiply + IFFT")

    # 2D comparisons
    print("\n" + "=" * 70)
    print("2D CONVOLUTION COMPARISONS")
    print("=" * 70)

    configs_2d = [
        (4, 64, 64, 64, 3),     # Small, 3x3
        (4, 128, 128, 128, 3),  # Medium, 3x3
        (4, 128, 256, 256, 3),  # Large, 3x3
        (4, 64, 64, 64, 5),     # 5x5 kernel
        (4, 64, 64, 64, 7),     # 7x7 kernel
        (4, 64, 128, 128, 7),   # Larger with 7x7
    ]

    for B, C, H, W, K in configs_2d:
        try:
            compare_conv2d(B, C, H, W, K, device=args.device,
                          test_backward=not args.no_backward,
                          visualize=args.visualize)
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
        (2, 32, 16, 32, 32, 5),  # 5x5x5 kernel
    ]

    for B, C, D, H, W, K in configs_3d:
        try:
            compare_conv3d(B, C, D, H, W, K, device=args.device,
                          test_backward=not args.no_backward,
                          visualize=args.visualize)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Skipped due to OOM")
                torch.cuda.empty_cache()
            else:
                raise

    # Multi-timestep comparisons (the KEY use case for FFT conv!)
    print("\n" + "=" * 70)
    print("MULTI-TIMESTEP ConvSSM COMPARISONS (KEY USE CASE)")
    print("=" * 70)
    print("This is where FFT conv shines - parallel scan with constant kernel size")

    timestep_configs = [
        (4, 64, 64, 64, 3, 10),    # T=10 timesteps
        (4, 64, 128, 128, 3, 10),  # Larger spatial
        (4, 64, 64, 64, 3, 50),    # T=50 timesteps
        (4, 64, 64, 64, 3, 100),   # T=100 timesteps
    ]

    for B, C, H, W, K, T in timestep_configs:
        try:
            compare_multi_timestep(B, C, H, W, K, T, device=args.device,
                                   visualize=args.visualize)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Skipped T={T} due to OOM")
                torch.cuda.empty_cache()
            else:
                raise

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings:
- For small kernels (3x3, 5x5), standard Conv2d/3d is typically FASTER
- FFT conv includes overhead: FFT(input) + FFT(kernel) + multiply + IFFT
- FFT conv advantage comes from PARALLEL SCAN, not single-operation speed

When FFT conv wins:
1. ConvSSM parallel scan (main use case!) - kernel stays constant in FFT space
2. Very large kernels (where O(N log N) beats O(N K^2))
3. When you need the kernel in FFT space for other operations

When standard conv wins:
1. Small kernels without temporal parallelization
2. Single convolution operations
3. When boundary handling must be exact (no periodic artifacts)
""")


if __name__ == '__main__':
    main()
