# Copyright (c) 2024
# N-dimensional FFT convolution for ConvSSMs
# Supports 2D and 3D spatial convolutions using torch.fft.fftn

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashFFTConvNDFunc(torch.autograd.Function):
    """
    Autograd function for N-dimensional FFT convolution.

    Computes: output = IFFT(FFT(input) * FFT(kernel))

    This is the core operation for ConvSSMs where convolution in pixel space
    becomes element-wise multiplication in FFT space.
    """

    @staticmethod
    def forward(ctx, u, k, fftconv_data, pregate, postgate):
        """
        Forward pass of N-D FFT convolution.

        Args:
            u: Input tensor of shape (B, C, *spatial_dims)
            k: Kernel tensor of shape (C, *kernel_dims)
            fftconv_data: FlashFFTConvND module containing spatial_size and dtype
            pregate: Optional gating tensor applied before convolution
            postgate: Optional gating tensor applied after convolution

        Returns:
            Output tensor of shape (B, C, *spatial_dims)
        """
        ndim = fftconv_data.ndim
        spatial_size = fftconv_data.spatial_size
        fft_dims = tuple(range(-ndim, 0))

        # Double FFT size for linear (non-circular) convolution
        fft_size = tuple(2 * s for s in spatial_size)

        # Apply pre-gate if provided
        u_work = u.float()
        if pregate is not None:
            u_work = u_work * pregate.float()

        # N-D FFT of input and kernel
        u_f = torch.fft.rfftn(u_work, s=fft_size, dim=fft_dims)
        k_f = torch.fft.rfftn(k.float(), s=fft_size, dim=fft_dims)

        # Element-wise multiplication in frequency domain
        # Broadcast kernel over batch dimension
        y_f = u_f * k_f.unsqueeze(0)

        # Inverse N-D FFT
        y = torch.fft.irfftn(y_f, s=fft_size, dim=fft_dims)

        # Crop to original size (take the valid convolution region)
        slices = [slice(None), slice(None)] + [slice(0, s) for s in u.shape[2:]]
        y = y[tuple(slices)]

        # Apply post-gate if provided
        if postgate is not None:
            y = y * postgate.float()

        y = y.to(u.dtype).contiguous()

        # Save tensors for backward pass
        ctx.save_for_backward(u, k, pregate, postgate)
        ctx.fftconv_data = fftconv_data
        ctx.ndim = ndim
        ctx.spatial_size = spatial_size
        ctx.fft_size = fft_size

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass computing gradients for input, kernel, and gates.

        For FFT convolution y = IFFT(FFT(u) * FFT(k)):
        - grad_u = IFFT(FFT(grad_y) * conj(FFT(k)))  [correlation with kernel]
        - grad_k = IFFT(conj(FFT(u)) * FFT(grad_y))  [correlation with input]
        """
        u, k, pregate, postgate = ctx.saved_tensors
        ndim = ctx.ndim
        spatial_size = ctx.spatial_size
        fft_size = ctx.fft_size
        fft_dims = tuple(range(-ndim, 0))

        grad_u = grad_k = grad_pregate = grad_postgate = None

        # Apply post-gate to gradient if present
        if postgate is not None:
            grad_work = grad_output.float() * postgate.float()
            if ctx.needs_input_grad[4]:  # postgate gradient
                # Forward output before postgate
                u_work = u.float()
                if pregate is not None:
                    u_work = u_work * pregate.float()
                u_f = torch.fft.rfftn(u_work, s=fft_size, dim=fft_dims)
                k_f = torch.fft.rfftn(k.float(), s=fft_size, dim=fft_dims)
                y_f = u_f * k_f.unsqueeze(0)
                y_pre_gate = torch.fft.irfftn(y_f, s=fft_size, dim=fft_dims)
                slices = [slice(None), slice(None)] + [slice(0, s) for s in u.shape[2:]]
                y_pre_gate = y_pre_gate[tuple(slices)]
                grad_postgate = (grad_output.float() * y_pre_gate).to(postgate.dtype)
        else:
            grad_work = grad_output.float()

        # Compute FFTs needed for gradients
        # Pad gradient to FFT size
        # F.pad expects padding in order: (W_left, W_right, H_left, H_right, D_left, D_right)
        # So we iterate from last dim to first and append
        pad_sizes = []
        for i in range(ndim):
            dim_idx = -(i + 1)
            pad_needed = fft_size[dim_idx] - grad_work.shape[dim_idx]
            pad_sizes = pad_sizes + [0, pad_needed]  # append, not prepend
        grad_padded = F.pad(grad_work, pad_sizes)

        grad_f = torch.fft.rfftn(grad_padded, dim=fft_dims)
        k_f = torch.fft.rfftn(k.float(), s=fft_size, dim=fft_dims)

        # grad_u: correlation of grad_output with kernel
        # In frequency domain: grad_u_f = grad_f * conj(k_f)
        if ctx.needs_input_grad[0]:
            grad_u_f = grad_f * torch.conj(k_f).unsqueeze(0)
            grad_u_full = torch.fft.irfftn(grad_u_f, s=fft_size, dim=fft_dims)

            # Crop to input size
            slices = [slice(None), slice(None)] + [slice(0, s) for s in u.shape[2:]]
            grad_u = grad_u_full[tuple(slices)]

            # Handle pregate
            if pregate is not None:
                if ctx.needs_input_grad[3]:  # pregate gradient
                    grad_pregate = (grad_u * u.float()).to(pregate.dtype)
                grad_u = grad_u * pregate.float()

            grad_u = grad_u.to(u.dtype)

        # grad_k: correlation of input with grad_output
        # In frequency domain: grad_k_f = conj(u_f) * grad_f, summed over batch
        if ctx.needs_input_grad[1]:
            u_work = u.float()
            if pregate is not None:
                u_work = u_work * pregate.float()

            # Pad input to FFT size
            u_padded = F.pad(u_work, pad_sizes)
            u_f = torch.fft.rfftn(u_padded, dim=fft_dims)

            grad_k_f = torch.conj(u_f) * grad_f
            grad_k_f = grad_k_f.sum(dim=0)  # Sum over batch

            grad_k_full = torch.fft.irfftn(grad_k_f, s=fft_size, dim=fft_dims)

            # Crop to kernel size
            k_slices = [slice(None)] + [slice(0, s) for s in k.shape[1:]]
            grad_k = grad_k_full[tuple(k_slices)].to(k.dtype)

        return grad_u, grad_k, None, grad_pregate, grad_postgate


class FlashFFTConvND(nn.Module):
    """
    N-dimensional FFT convolution module.

    Uses torch.fft.fftn/ifftn for efficient convolution in frequency domain.
    This is the core building block for ConvSSMs where spatial convolution
    becomes element-wise multiplication in FFT space.

    The kernel can be ANY size up to the spatial dimensions - small kernels
    (e.g., 3x3) are automatically zero-padded to the FFT size internally.

    Args:
        spatial_size: Tuple of spatial dimensions (H, W) for 2D or (D, H, W) for 3D
        dtype: Data type for computation (torch.float16 or torch.bfloat16)

    Example:
        >>> # 2D convolution with SMALL kernel (typical for ConvSSM)
        >>> fftconv = FlashFFTConvND((64, 64), dtype=torch.bfloat16)
        >>> u = torch.randn(4, 128, 64, 64)  # (B, C, H, W)
        >>> k = torch.randn(128, 3, 3)       # (C, Kh, Kw) - 3x3 kernel!
        >>> y = fftconv(u, k)  # Output: (4, 128, 64, 64)

        >>> # 3D convolution with small kernel
        >>> fftconv = FlashFFTConvND((16, 64, 64), dtype=torch.bfloat16)
        >>> u = torch.randn(4, 128, 16, 64, 64)  # (B, C, D, H, W)
        >>> k = torch.randn(128, 3, 3, 3)        # (C, Kd, Kh, Kw) - 3x3x3 kernel!
        >>> y = fftconv(u, k)  # Output: (4, 128, 16, 64, 64)
    """

    def __init__(
        self,
        spatial_size: Tuple[int, ...],
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.dtype = dtype
        self.ndim = len(spatial_size)

        if self.ndim < 2 or self.ndim > 3:
            raise ValueError(f"spatial_size must have 2 or 3 dimensions, got {self.ndim}")

    def forward(
        self,
        u: torch.Tensor,
        k: torch.Tensor,
        pregate: Optional[torch.Tensor] = None,
        postgate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform N-D FFT convolution.

        Args:
            u: Input tensor of shape (B, C, *spatial_dims)
            k: Kernel tensor of shape (C, *kernel_dims)
               Kernel can be smaller than spatial_dims and will be zero-padded
            pregate: Optional tensor to multiply with input before convolution
            postgate: Optional tensor to multiply with output after convolution

        Returns:
            Output tensor of shape (B, C, *spatial_dims)
        """
        return FlashFFTConvNDFunc.apply(u, k, self, pregate, postgate)

    def extra_repr(self) -> str:
        return f"spatial_size={self.spatial_size}, dtype={self.dtype}, ndim={self.ndim}"


class FlashFFTConv2D(FlashFFTConvND):
    """
    2D FFT convolution - convenience wrapper for FlashFFTConvND.

    Args:
        height: Height of the input spatial dimensions
        width: Width of the input spatial dimensions
        dtype: Data type for computation

    Example:
        >>> # Small 3x3 kernel on 64x64 images
        >>> fftconv = FlashFFTConv2D(64, 64, dtype=torch.bfloat16)
        >>> u = torch.randn(4, 128, 64, 64, device='cuda')  # (B, C, H, W)
        >>> k = torch.randn(128, 3, 3, device='cuda')       # (C, Kh, Kw) - small kernel!
        >>> y = fftconv(u, k)  # Output: (4, 128, 64, 64)

        >>> # ConvSSM usage: h_new = A_conv(h_prev) + B_conv(x)
        >>> A_kernel = torch.randn(128, 3, 3, device='cuda') * 0.01
        >>> B_kernel = torch.randn(128, 3, 3, device='cuda') * 0.01
        >>> h_new = fftconv(h_prev, A_kernel) + fftconv(x, B_kernel)
    """

    def __init__(
        self,
        height: int,
        width: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__((height, width), dtype=dtype)
        self.height = height
        self.width = width

    def extra_repr(self) -> str:
        return f"height={self.height}, width={self.width}, dtype={self.dtype}"


class FlashFFTConv3D(FlashFFTConvND):
    """
    3D FFT convolution - convenience wrapper for FlashFFTConvND.

    For volumetric data (e.g., 3D medical imaging, fluid dynamics).

    Args:
        depth: Depth of the input spatial dimensions
        height: Height of the input spatial dimensions
        width: Width of the input spatial dimensions
        dtype: Data type for computation

    Example:
        >>> # Small 3x3x3 kernel on 16x64x64 volumes
        >>> fftconv = FlashFFTConv3D(16, 64, 64, dtype=torch.bfloat16)
        >>> u = torch.randn(4, 128, 16, 64, 64, device='cuda')  # (B, C, D, H, W)
        >>> k = torch.randn(128, 3, 3, 3, device='cuda')        # (C, Kd, Kh, Kw) - small kernel!
        >>> y = fftconv(u, k)  # Output: (4, 128, 16, 64, 64)

        >>> # 3D ConvSSM for volumetric sequences
        >>> A_kernel = torch.randn(128, 3, 3, 3, device='cuda') * 0.01
        >>> B_kernel = torch.randn(128, 3, 3, 3, device='cuda') * 0.01
        >>> h_new = fftconv(h_prev, A_kernel) + fftconv(x, B_kernel)
    """

    def __init__(
        self,
        depth: int,
        height: int,
        width: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__((depth, height, width), dtype=dtype)
        self.depth = depth
        self.height = height
        self.width = width

    def extra_repr(self) -> str:
        return f"depth={self.depth}, height={self.height}, width={self.width}, dtype={self.dtype}"
