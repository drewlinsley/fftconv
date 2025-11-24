# Benchmarks for N-dimensional FFT convolution
# Compares FlashFFTConv2D/3D against standard PyTorch convolutions

import argparse
import time
import torch
import torch.nn.functional as F

from flashfftconv import FlashFFTConv2D, FlashFFTConv3D


def benchmark_forward(fn, warmup=10, rep=100):
    """Benchmark a function with warmup and multiple repetitions."""
    # Warmup
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / rep * 1000  # ms


def benchmark_2d(B, C, H, W, kernel_size, dtype, device='cuda'):
    """Benchmark 2D convolutions - FAIR comparison with same kernel size."""
    print(f"\n2D Benchmark: B={B}, C={C}, H={H}, W={W}, K={kernel_size}, dtype={dtype}")
    print("-" * 70)

    K = kernel_size
    # Create input
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)

    # =========================================================================
    # Standard PyTorch Conv2d (depthwise for fair comparison)
    # =========================================================================
    conv_std = torch.nn.Conv2d(C, C, K, padding=K // 2, groups=C, bias=False).to(device).to(dtype)

    def run_std():
        return conv_std(x)

    time_std = benchmark_forward(run_std)
    print(f"PyTorch Conv2d (K={K}×{K}): {time_std:.3f} ms")

    # =========================================================================
    # FlashFFTConv2D with SAME small kernel
    # =========================================================================
    fftconv = FlashFFTConv2D(H, W, dtype=dtype).to(device)
    # Use same kernel size as Conv2d for fair comparison
    k_fft = torch.randn(C, K, K, device=device, dtype=torch.float32) * 0.01

    def run_fft():
        return fftconv(x, k_fft)

    time_fft = benchmark_forward(run_fft)
    print(f"FlashFFTConv2D (K={K}×{K}): {time_fft:.3f} ms  [FFT+multiply+IFFT]")

    # =========================================================================
    # PyTorch FFT (baseline) - also with small kernel
    # =========================================================================
    def run_torch_fft():
        fft_size = (2 * H, 2 * W)
        x_f = torch.fft.rfftn(x.float(), s=fft_size, dim=(-2, -1))
        k_f = torch.fft.rfftn(k_fft.float(), s=fft_size, dim=(-2, -1))
        y_f = x_f * k_f.unsqueeze(0)
        y = torch.fft.irfftn(y_f, s=fft_size, dim=(-2, -1))
        return y[..., :H, :W].to(dtype)

    time_torch_fft = benchmark_forward(run_torch_fft)
    print(f"PyTorch torch.fft.rfftn (K={K}×{K}): {time_torch_fft:.3f} ms")

    # Report which is faster
    if time_fft < time_std:
        print(f"\nFFT conv is {time_std / time_fft:.2f}x FASTER than Conv2d")
    else:
        print(f"\nConv2d is {time_fft / time_std:.2f}x FASTER than FFT conv")

    return {
        'conv2d': time_std,
        'fftconv2d': time_fft,
        'torch_fft': time_torch_fft,
    }


def benchmark_3d(B, C, D, H, W, kernel_size, dtype, device='cuda'):
    """Benchmark 3D convolutions - FAIR comparison with same kernel size."""
    print(f"\n3D Benchmark: B={B}, C={C}, D={D}, H={H}, W={W}, K={kernel_size}, dtype={dtype}")
    print("-" * 70)

    K = kernel_size
    # Create input
    x = torch.randn(B, C, D, H, W, device=device, dtype=dtype)

    # =========================================================================
    # Standard PyTorch Conv3d (depthwise for fair comparison)
    # =========================================================================
    conv_std = torch.nn.Conv3d(C, C, K, padding=K // 2, groups=C, bias=False).to(device).to(dtype)

    def run_std():
        return conv_std(x)

    time_std = benchmark_forward(run_std)
    print(f"PyTorch Conv3d (K={K}×{K}×{K}): {time_std:.3f} ms")

    # =========================================================================
    # FlashFFTConv3D with SAME small kernel
    # =========================================================================
    fftconv = FlashFFTConv3D(D, H, W, dtype=dtype).to(device)
    # Use same kernel size as Conv3d for fair comparison
    k_fft = torch.randn(C, K, K, K, device=device, dtype=torch.float32) * 0.01

    def run_fft():
        return fftconv(x, k_fft)

    time_fft = benchmark_forward(run_fft)
    print(f"FlashFFTConv3D (K={K}×{K}×{K}): {time_fft:.3f} ms  [FFT+multiply+IFFT]")

    # =========================================================================
    # PyTorch FFT (baseline) - also with small kernel
    # =========================================================================
    def run_torch_fft():
        fft_size = (2 * D, 2 * H, 2 * W)
        x_f = torch.fft.rfftn(x.float(), s=fft_size, dim=(-3, -2, -1))
        k_f = torch.fft.rfftn(k_fft.float(), s=fft_size, dim=(-3, -2, -1))
        y_f = x_f * k_f.unsqueeze(0)
        y = torch.fft.irfftn(y_f, s=fft_size, dim=(-3, -2, -1))
        return y[..., :D, :H, :W].to(dtype)

    time_torch_fft = benchmark_forward(run_torch_fft)
    print(f"PyTorch torch.fft.rfftn (K={K}×{K}×{K}): {time_torch_fft:.3f} ms")

    # Report which is faster
    if time_fft < time_std:
        print(f"\nFFT conv is {time_std / time_fft:.2f}x FASTER than Conv3d")
    else:
        print(f"\nConv3d is {time_fft / time_std:.2f}x FASTER than FFT conv")

    return {
        'conv3d': time_std,
        'fftconv3d': time_fft,
        'torch_fft': time_torch_fft,
    }


def benchmark_backward_2d(B, C, H, W, dtype, device='cuda'):
    """Benchmark 2D convolution backward pass."""
    print(f"\n2D Backward Benchmark: B={B}, C={C}, H={H}, W={W}, dtype={dtype}")
    print("-" * 70)

    fftconv = FlashFFTConv2D(H, W, dtype=dtype).to(device)

    def run_fwd_bwd():
        # Create fresh tensors each iteration to avoid graph reuse issues
        x = (torch.randn(B, C, H, W, device=device, dtype=dtype) * 0.1).requires_grad_(True)
        k = (torch.randn(C, H, W, device=device, dtype=torch.float32) * 0.01).requires_grad_(True)
        out = fftconv(x, k)
        loss = out.sum()
        loss.backward()

    time_fwd_bwd = benchmark_forward(run_fwd_bwd, warmup=5, rep=50)
    print(f"FlashFFTConv2D forward+backward: {time_fwd_bwd:.3f} ms")

    return {'fwd_bwd': time_fwd_bwd}


def benchmark_convssm_step(B, C, H, W, dtype, device='cuda'):
    """Benchmark a ConvSSM timestep: h_new = A*h_prev + B*x."""
    print(f"\nConvSSM Step Benchmark: B={B}, C={C}, H={H}, W={W}, dtype={dtype}")
    print("-" * 70)

    fftconv = FlashFFTConv2D(H, W, dtype=dtype).to(device)

    # Kernels
    A_kernel = torch.randn(C, H, W, device=device, dtype=torch.float32) * 0.01
    B_kernel = torch.randn(C, H, W, device=device, dtype=torch.float32) * 0.01

    # Apply decay
    decay_h = torch.exp(-0.1 * torch.arange(H, device=device)).view(-1, 1)
    decay_w = torch.exp(-0.1 * torch.arange(W, device=device)).view(1, -1)
    A_kernel = A_kernel * decay_h * decay_w
    B_kernel = B_kernel * decay_h * decay_w

    # Input and hidden state
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)
    h_prev = torch.randn(B, C, H, W, device=device, dtype=dtype)

    def run_step():
        return fftconv(h_prev, A_kernel) + fftconv(x, B_kernel)

    time_step = benchmark_forward(run_step)
    print(f"ConvSSM step (2 FFT convs): {time_step:.3f} ms")

    # Compare with ConvLSTM-like (standard conv)
    conv_A = torch.nn.Conv2d(C, C, 3, padding=1, bias=False).to(device).to(dtype)
    conv_B = torch.nn.Conv2d(C, C, 3, padding=1, bias=False).to(device).to(dtype)

    def run_convlstm_step():
        return conv_A(h_prev) + conv_B(x)

    time_convlstm = benchmark_forward(run_convlstm_step)
    print(f"ConvLSTM step (2 Conv2d K=3): {time_convlstm:.3f} ms")

    return {'convssm': time_step, 'convlstm': time_convlstm}


def main():
    parser = argparse.ArgumentParser(description='Benchmark N-D FFT convolutions')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print("=" * 70)
    print("N-D FFT Convolution Benchmarks")
    print("=" * 70)

    # 2D Benchmarks
    print("\n" + "=" * 70)
    print("2D CONVOLUTION BENCHMARKS")
    print("=" * 70)

    for H, W in [(64, 64), (128, 128), (256, 256)]:
        for K in [3, 5, 7]:
            benchmark_2d(B=4, C=128, H=H, W=W, kernel_size=K, dtype=dtype, device=args.device)

    # 3D Benchmarks
    print("\n" + "=" * 70)
    print("3D CONVOLUTION BENCHMARKS")
    print("=" * 70)

    for D, H, W in [(16, 32, 32), (32, 32, 32), (32, 64, 64)]:
        for K in [3, 5]:
            try:
                benchmark_3d(B=2, C=64, D=D, H=H, W=W, kernel_size=K, dtype=dtype, device=args.device)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Skipped due to OOM: D={D}, H={H}, W={W}, K={K}")
                    torch.cuda.empty_cache()
                else:
                    raise

    # Backward pass benchmarks
    print("\n" + "=" * 70)
    print("BACKWARD PASS BENCHMARKS")
    print("=" * 70)

    benchmark_backward_2d(B=4, C=128, H=64, W=64, dtype=dtype, device=args.device)
    benchmark_backward_2d(B=4, C=128, H=128, W=128, dtype=dtype, device=args.device)

    # ConvSSM step benchmarks
    print("\n" + "=" * 70)
    print("ConvSSM STEP BENCHMARKS")
    print("=" * 70)

    benchmark_convssm_step(B=4, C=128, H=64, W=64, dtype=dtype, device=args.device)
    benchmark_convssm_step(B=4, C=128, H=128, W=128, dtype=dtype, device=args.device)


if __name__ == '__main__':
    main()
