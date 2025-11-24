# N-D FFT convolutions (pure PyTorch, no CUDA compilation needed)
from .conv_nd import FlashFFTConvND, FlashFFTConv2D, FlashFFTConv3D

# 1D FFT convolutions (requires CUDA compilation of monarch_cuda)
try:
    from .conv import FlashFFTConv
    from .depthwise_1d import FlashDepthWiseConv1d
except ImportError:
    # monarch_cuda not compiled - 1D FlashFFTConv not available
    FlashFFTConv = None
    FlashDepthWiseConv1d = None