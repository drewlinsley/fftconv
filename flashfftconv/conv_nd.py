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


# =============================================================================
# Parallel Scan for ConvSSM
# =============================================================================

def parallel_scan_ref(A_f: torch.Tensor, B_f: torch.Tensor, x_seq_f: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of parallel scan for ConvSSM (sequential, for validation).

    Computes h_t = A ★ h_{t-1} + B ★ x_t for all t in sequence.

    All inputs should already be in frequency domain.

    Args:
        A_f: FFT of A kernel, shape (C, *fft_spatial_dims)
        B_f: FFT of B kernel, shape (C, *fft_spatial_dims)
        x_seq_f: FFT of input sequence, shape (T, B, C, *fft_spatial_dims)

    Returns:
        h_seq_f: FFT of hidden states, shape (T, B, C, *fft_spatial_dims)
    """
    T = x_seq_f.shape[0]
    device = x_seq_f.device
    dtype = x_seq_f.dtype

    # Initialize output
    h_seq_f = torch.zeros_like(x_seq_f)

    # Sequential recurrence (reference)
    h_f = torch.zeros_like(x_seq_f[0])
    for t in range(T):
        h_f = h_f * A_f.unsqueeze(0) + x_seq_f[t] * B_f.unsqueeze(0)
        h_seq_f[t] = h_f

    return h_seq_f


def _parallel_scan_combine(a1: torch.Tensor, s1: torch.Tensor,
                           a2: torch.Tensor, s2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine operation for parallel scan.

    For recurrence h_t = a * h_{t-1} + s, the combination rule is:
    (a1, s1) ⊕ (a2, s2) = (a1 * a2, s1 * a2 + s2)

    This represents: applying (a1, s1) then (a2, s2) is equivalent to (a1*a2, s1*a2 + s2).
    """
    return a1 * a2, s1 * a2 + s2


def parallel_scan_fft(A_f: torch.Tensor, B_f: torch.Tensor, x_seq_f: torch.Tensor,
                      return_all: bool = True) -> torch.Tensor:
    """
    True O(log T) parallel scan for ConvSSM in frequency domain.

    Computes h_t = A ★ h_{t-1} + B ★ x_t for all t using Blelloch-style parallel scan.

    In frequency domain, convolution becomes element-wise multiplication, so:
    H_t = Â · H_{t-1} + B̂ · X_t

    This is a linear recurrence that can be computed in O(log T) parallel steps
    using the associative combination rule:
    (a1, s1) ⊕ (a2, s2) = (a1 * a2, s1 * a2 + s2)

    Args:
        A_f: FFT of A kernel, shape (C, *fft_spatial_dims) - complex
        B_f: FFT of B kernel, shape (C, *fft_spatial_dims) - complex
        x_seq_f: FFT of input sequence, shape (T, B, C, *fft_spatial_dims) - complex
        return_all: If True, return all hidden states; if False, return only final state

    Returns:
        If return_all: h_seq_f of shape (T, B, C, *fft_spatial_dims)
        If not return_all: h_T_f of shape (B, C, *fft_spatial_dims)

    Complexity:
        - Sequential: O(T) serial operations
        - Parallel scan: O(log T) parallel depth, O(T) total work
    """
    T, B, C = x_seq_f.shape[:3]
    spatial_shape = x_seq_f.shape[3:]
    device = x_seq_f.device
    dtype = x_seq_f.dtype

    # Handle edge cases
    if T == 0:
        if return_all:
            return torch.zeros_like(x_seq_f)
        else:
            return torch.zeros(B, C, *spatial_shape, device=device, dtype=dtype)

    if T == 1:
        h_f = x_seq_f[0] * B_f.unsqueeze(0)
        if return_all:
            return h_f.unsqueeze(0)
        else:
            return h_f

    # Initialize scan arrays
    # a[t] = A_f for all t (will become A^{t+1} after scan)
    # s[t] = B_f * x_seq_f[t] (will become h_{t+1} after scan)
    a = A_f.unsqueeze(0).unsqueeze(0).expand(T, B, -1, *[-1]*len(spatial_shape)).clone()
    s = x_seq_f * B_f.unsqueeze(0).unsqueeze(0)

    # Copy to work arrays
    a_work = a.clone()
    s_work = s.clone()

    # Inclusive parallel scan using doubling technique
    # This computes all prefix sums in O(log T) parallel steps
    # Each iteration doubles the range of elements that contribute to each position
    offset = 1
    while offset < T:
        # For each position i >= offset, combine with position i - offset
        # s_new[i] = s[i - offset] * a[i] + s[i]
        # a_new[i] = a[i - offset] * a[i]

        # Create shifted versions
        a_shifted = torch.zeros_like(a_work)
        s_shifted = torch.zeros_like(s_work)

        a_shifted[offset:] = a_work[:-offset]
        s_shifted[offset:] = s_work[:-offset]

        # For positions >= offset, combine
        # For positions < offset, keep original
        mask = torch.arange(T, device=device) >= offset
        mask = mask.view(T, 1, 1, *([1] * len(spatial_shape)))

        a_new = torch.where(mask, a_shifted * a_work, a_work)
        s_new = torch.where(mask, s_shifted * a_work + s_work, s_work)

        a_work = a_new
        s_work = s_new

        offset *= 2

    # s_work now contains h_1, h_2, ..., h_T
    if return_all:
        return s_work
    else:
        return s_work[-1]


class ConvSSMParallelScan(nn.Module):
    """
    Parallel scan module for ConvSSM: h_t = A ★ h_{t-1} + B ★ x_t

    This module computes the full sequence of hidden states in O(log T) parallel
    depth instead of O(T) sequential steps, by leveraging the associativity of
    the recurrence relation in FFT space.

    Key insight: In frequency domain, convolution becomes element-wise multiplication,
    and the recurrence H_t = Â · H_{t-1} + B̂ · X_t can be computed using parallel
    prefix sum (scan) with the associative operator:
    (a1, s1) ⊕ (a2, s2) = (a1 * a2, s1 * a2 + s2)

    Args:
        spatial_size: Tuple of spatial dimensions (H, W) for 2D or (D, H, W) for 3D
        dtype: Data type for computation

    Example:
        >>> # 2D ConvSSM with parallel scan
        >>> scanner = ConvSSMParallelScan((64, 64))
        >>> A_kernel = torch.randn(128, 3, 3) * 0.1  # (C, Kh, Kw)
        >>> B_kernel = torch.randn(128, 3, 3) * 0.1
        >>> x_seq = torch.randn(100, 4, 128, 64, 64)  # (T, B, C, H, W)
        >>> h_seq = scanner(x_seq, A_kernel, B_kernel)  # (T, B, C, H, W)
        >>> # Computed in O(log 100) ≈ 7 parallel steps instead of 100 sequential!
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

        # FFT size (doubled for linear convolution)
        self.fft_size = tuple(2 * s for s in spatial_size)
        self.fft_dims = tuple(range(-self.ndim, 0))

    def forward(
        self,
        x_seq: torch.Tensor,
        A_kernel: torch.Tensor,
        B_kernel: torch.Tensor,
        return_all: bool = True,
    ) -> torch.Tensor:
        """
        Compute ConvSSM recurrence using parallel scan.

        Args:
            x_seq: Input sequence of shape (T, B, C, *spatial_dims)
            A_kernel: Transition kernel of shape (C, *kernel_dims)
            B_kernel: Input kernel of shape (C, *kernel_dims)
            return_all: If True, return all hidden states (T, B, C, *spatial_dims)
                       If False, return only final state (B, C, *spatial_dims)

        Returns:
            Hidden states computed via h_t = A ★ h_{t-1} + B ★ x_t
        """
        T = x_seq.shape[0]
        original_dtype = x_seq.dtype

        # FFT all inputs
        x_seq_f = torch.fft.rfftn(x_seq.float(), s=self.fft_size, dim=self.fft_dims)
        A_f = torch.fft.rfftn(A_kernel.float(), s=self.fft_size, dim=self.fft_dims)
        B_f = torch.fft.rfftn(B_kernel.float(), s=self.fft_size, dim=self.fft_dims)

        # Parallel scan in frequency domain
        h_seq_f = parallel_scan_fft(A_f, B_f, x_seq_f, return_all=return_all)

        # IFFT to get spatial domain result
        if return_all:
            h_seq = torch.fft.irfftn(h_seq_f, s=self.fft_size, dim=self.fft_dims)
            # Crop to original spatial size
            slices = [slice(None), slice(None), slice(None)] + [slice(0, s) for s in self.spatial_size]
            h_seq = h_seq[tuple(slices)]
        else:
            h = torch.fft.irfftn(h_seq_f, s=self.fft_size, dim=self.fft_dims)
            slices = [slice(None), slice(None)] + [slice(0, s) for s in self.spatial_size]
            h_seq = h[tuple(slices)]

        return h_seq.to(original_dtype)

    def extra_repr(self) -> str:
        return f"spatial_size={self.spatial_size}, dtype={self.dtype}, ndim={self.ndim}"


class ConvSSMParallelScan2D(ConvSSMParallelScan):
    """
    2D parallel scan for ConvSSM - convenience wrapper.

    Args:
        height: Height of spatial dimensions
        width: Width of spatial dimensions
        dtype: Data type for computation

    Example:
        >>> scanner = ConvSSMParallelScan2D(64, 64)
        >>> A_kernel = torch.randn(128, 3, 3) * 0.1
        >>> B_kernel = torch.randn(128, 3, 3) * 0.1
        >>> x_seq = torch.randn(100, 4, 128, 64, 64)  # T=100 timesteps
        >>> h_seq = scanner(x_seq, A_kernel, B_kernel)
        >>> h_final = scanner(x_seq, A_kernel, B_kernel, return_all=False)
    """

    def __init__(self, height: int, width: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__((height, width), dtype=dtype)
        self.height = height
        self.width = width

    def extra_repr(self) -> str:
        return f"height={self.height}, width={self.width}, dtype={self.dtype}"


class ConvSSMParallelScan3D(ConvSSMParallelScan):
    """
    3D parallel scan for ConvSSM - convenience wrapper.

    Args:
        depth: Depth of spatial dimensions
        height: Height of spatial dimensions
        width: Width of spatial dimensions
        dtype: Data type for computation

    Example:
        >>> scanner = ConvSSMParallelScan3D(16, 64, 64)
        >>> A_kernel = torch.randn(64, 3, 3, 3) * 0.1
        >>> B_kernel = torch.randn(64, 3, 3, 3) * 0.1
        >>> x_seq = torch.randn(50, 2, 64, 16, 64, 64)  # T=50 timesteps
        >>> h_seq = scanner(x_seq, A_kernel, B_kernel)
    """

    def __init__(self, depth: int, height: int, width: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__((depth, height, width), dtype=dtype)
        self.depth = depth
        self.height = height
        self.width = width

    def extra_repr(self) -> str:
        return f"depth={self.depth}, height={self.height}, width={self.width}, dtype={self.dtype}"
