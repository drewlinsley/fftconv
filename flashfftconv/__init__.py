# N-D FFT convolutions (pure PyTorch, no CUDA compilation needed)
from .conv_nd import FlashFFTConvND, FlashFFTConv2D, FlashFFTConv3D

# Parallel scan for ConvSSM (O(log T) depth instead of O(T) sequential)
from .conv_nd import (
    ConvSSMParallelScan,
    ConvSSMParallelScan2D,
    ConvSSMParallelScan3D,
    parallel_scan_fft,
    parallel_scan_ref,
)

# 1D FFT convolutions (requires CUDA compilation of monarch_cuda)
try:
    from .conv import FlashFFTConv
    from .depthwise_1d import FlashDepthWiseConv1d
except ImportError:
    # monarch_cuda not compiled - 1D FlashFFTConv not available
    FlashFFTConv = None
    FlashDepthWiseConv1d = None

# JAX implementations (requires JAX)
try:
    from .conv_nd_jax import (
        # 3D functions
        fft_conv_3d, fft_conv_3d_jit,
        convssm_sequential_3d, convssm_sequential_3d_jit,
        convssm_parallel_3d, convssm_parallel_3d_jit,
        FlashFFTConv3DJAX, ConvSSMParallelScan3DJAX,
        # 2D functions
        fft_conv_2d, fft_conv_2d_jit,
        convssm_sequential_2d, convssm_sequential_2d_jit,
        convssm_parallel_2d, convssm_parallel_2d_jit,
        # Fourier-space (no FFT during forward pass!)
        FourierConvSSM3D, FourierConvSSM2D,
        to_fourier_3d, from_fourier_3d, kernel_to_fourier_3d,
        to_fourier_2d, from_fourier_2d, kernel_to_fourier_2d,
        convssm_fourier_scan, convssm_fourier_scan_2d,
    )
except ImportError:
    # JAX not installed
    fft_conv_3d = fft_conv_2d = None
    FourierConvSSM3D = FourierConvSSM2D = None

# ConvNeXt-SSM model (requires JAX + Flax)
try:
    from .convnext_ssm import (
        ConvNeXtSSM,
        convnext_ssm_tiny, convnext_ssm_small, convnext_ssm_base, convnext_ssm_large,
        ConvSSMBlock, ConvSSMBlockFourier, ConvNeXtSSMBlock,
    )
except ImportError:
    # Flax not installed
    ConvNeXtSSM = None
    convnext_ssm_tiny = convnext_ssm_small = convnext_ssm_base = convnext_ssm_large = None

# ConvNeXt-Fourier: Fully Fourier-domain model (requires JAX + Flax)
try:
    from .convnext_fourier import (
        ConvNeXtFourier,
        convnext_fourier_tiny, convnext_fourier_small, convnext_fourier_base,
        FourierBlock, FourierConvSSM, FourierDownsample,
        # Pre-FFT version: accepts pre-FFT'd images (no FFT in forward pass!)
        ConvNeXtFourierPreFFT, convnext_fourier_prefft_tiny,
    )
except ImportError:
    ConvNeXtFourier = None
    ConvNeXtFourierPreFFT = None
    convnext_fourier_tiny = convnext_fourier_small = convnext_fourier_base = None
    convnext_fourier_prefft_tiny = None