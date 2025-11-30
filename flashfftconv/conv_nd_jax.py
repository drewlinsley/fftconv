# JAX implementation of N-dimensional FFT convolution for ConvSSMs
# Focused on 3D case with true O(log T) parallel scan via lax.associative_scan

from typing import Tuple, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


# =============================================================================
# Core FFT Convolution Functions
# =============================================================================

def fft_conv_3d(
    u: jnp.ndarray,
    k: jnp.ndarray,
    spatial_size: Tuple[int, int, int],
) -> jnp.ndarray:
    """
    3D FFT convolution.

    Computes: output = IFFT(FFT(input) * FFT(kernel))

    Args:
        u: Input tensor of shape (B, C, D, H, W)
        k: Kernel tensor of shape (C, Kd, Kh, Kw) - can be smaller than spatial_size
        spatial_size: (D, H, W) of the input

    Returns:
        Output tensor of shape (B, C, D, H, W)
    """
    D, H, W = spatial_size

    # Double FFT size for linear (non-circular) convolution
    fft_size = (2 * D, 2 * H, 2 * W)

    # N-D FFT of input and kernel
    u_f = jnp.fft.rfftn(u, s=fft_size, axes=(-3, -2, -1))
    k_f = jnp.fft.rfftn(k, s=fft_size, axes=(-3, -2, -1))

    # Element-wise multiplication in frequency domain
    # Broadcast kernel over batch dimension
    y_f = u_f * k_f[None, ...]

    # Inverse N-D FFT
    y = jnp.fft.irfftn(y_f, s=fft_size, axes=(-3, -2, -1))

    # Crop to original size
    y = y[..., :D, :H, :W]

    return y


@partial(jax.jit, static_argnums=(2,))
def fft_conv_3d_jit(
    u: jnp.ndarray,
    k: jnp.ndarray,
    spatial_size: Tuple[int, int, int],
) -> jnp.ndarray:
    """JIT-compiled version of fft_conv_3d."""
    return fft_conv_3d(u, k, spatial_size)


# =============================================================================
# 2D FFT Convolution Functions (for images)
# =============================================================================

def fft_conv_2d(
    u: jnp.ndarray,
    k: jnp.ndarray,
    spatial_size: Tuple[int, int],
) -> jnp.ndarray:
    """
    2D FFT convolution for images.

    Computes: output = IFFT(FFT(input) * FFT(kernel))

    Args:
        u: Input tensor of shape (B, C, H, W)
        k: Kernel tensor of shape (C, Kh, Kw) - can be smaller than spatial_size
        spatial_size: (H, W) of the input

    Returns:
        Output tensor of shape (B, C, H, W)
    """
    H, W = spatial_size

    # Double FFT size for linear (non-circular) convolution
    fft_size = (2 * H, 2 * W)

    # 2D FFT of input and kernel
    u_f = jnp.fft.rfftn(u, s=fft_size, axes=(-2, -1))
    k_f = jnp.fft.rfftn(k, s=fft_size, axes=(-2, -1))

    # Element-wise multiplication in frequency domain
    # Broadcast kernel over batch dimension
    y_f = u_f * k_f[None, ...]

    # Inverse 2D FFT
    y = jnp.fft.irfftn(y_f, s=fft_size, axes=(-2, -1))

    # Crop to original size
    y = y[..., :H, :W]

    return y


@partial(jax.jit, static_argnums=(2,))
def fft_conv_2d_jit(
    u: jnp.ndarray,
    k: jnp.ndarray,
    spatial_size: Tuple[int, int],
) -> jnp.ndarray:
    """JIT-compiled version of fft_conv_2d."""
    return fft_conv_2d(u, k, spatial_size)


# =============================================================================
# 2D Sequential ConvSSM
# =============================================================================

def convssm_sequential_2d(
    x_seq: jnp.ndarray,
    A_kernel: jnp.ndarray,
    B_kernel: jnp.ndarray,
    spatial_size: Tuple[int, int],
) -> jnp.ndarray:
    """
    Sequential 2D ConvSSM: h_t = A ★ h_{t-1} + B ★ x_t

    Args:
        x_seq: Input sequence (T, B, C, H, W)
        A_kernel: Transition kernel (C, Kh, Kw)
        B_kernel: Input kernel (C, Kh, Kw)
        spatial_size: (H, W)

    Returns:
        h_seq: Hidden states (T, B, C, H, W)
    """
    B, C = x_seq.shape[1], x_seq.shape[2]
    H, W = spatial_size

    def step_fn(h, x_t):
        h_new = fft_conv_2d(h, A_kernel, spatial_size) + fft_conv_2d(x_t, B_kernel, spatial_size)
        return h_new, h_new

    h_init = jnp.zeros((B, C, H, W), dtype=x_seq.dtype)
    _, h_seq = lax.scan(step_fn, h_init, x_seq)

    return h_seq


@partial(jax.jit, static_argnums=(3,))
def convssm_sequential_2d_jit(
    x_seq: jnp.ndarray,
    A_kernel: jnp.ndarray,
    B_kernel: jnp.ndarray,
    spatial_size: Tuple[int, int],
) -> jnp.ndarray:
    """JIT-compiled sequential 2D ConvSSM."""
    return convssm_sequential_2d(x_seq, A_kernel, B_kernel, spatial_size)


# =============================================================================
# Associative Combination Function (shared by 2D and 3D parallel scans)
# =============================================================================

def _combine_fn(carry1, carry2):
    """
    Associative combination for ConvSSM parallel scan.

    For recurrence h_t = a * h_{t-1} + s, the combination rule is:
    (a1, s1) ⊕ (a2, s2) = (a1 * a2, s1 * a2 + s2)

    This is associative: ((a1,s1) ⊕ (a2,s2)) ⊕ (a3,s3) = (a1,s1) ⊕ ((a2,s2) ⊕ (a3,s3))
    """
    a1, s1 = carry1
    a2, s2 = carry2
    return (a1 * a2, s1 * a2 + s2)


# =============================================================================
# 2D Parallel Scan ConvSSM
# =============================================================================

def convssm_parallel_2d(
    x_seq: jnp.ndarray,
    A_kernel: jnp.ndarray,
    B_kernel: jnp.ndarray,
    spatial_size: Tuple[int, int],
    return_all: bool = True,
) -> jnp.ndarray:
    """
    Parallel 2D ConvSSM using O(log T) associative scan.

    Computes h_t = A ★ h_{t-1} + B ★ x_t for all t in parallel.

    Args:
        x_seq: Input sequence (T, B, C, H, W)
        A_kernel: Transition kernel (C, Kh, Kw)
        B_kernel: Input kernel (C, Kh, Kw)
        spatial_size: (H, W)
        return_all: If True, return all hidden states; if False, return only final

    Returns:
        If return_all: h_seq of shape (T, B, C, H, W)
        If not return_all: h_final of shape (B, C, H, W)
    """
    H, W = spatial_size
    fft_size = (2 * H, 2 * W)

    # FFT all inputs (batched, efficient)
    x_seq_f = jnp.fft.rfftn(x_seq, s=fft_size, axes=(-2, -1))
    A_f = jnp.fft.rfftn(A_kernel, s=fft_size, axes=(-2, -1))
    B_f = jnp.fft.rfftn(B_kernel, s=fft_size, axes=(-2, -1))

    # Parallel scan in frequency domain
    a = jnp.broadcast_to(A_f[None, None, ...], x_seq_f.shape)
    s = x_seq_f * B_f[None, None, ...]
    _, h_seq_f = lax.associative_scan(_combine_fn, (a, s), axis=0)

    # IFFT to get spatial domain result
    h_seq = jnp.fft.irfftn(h_seq_f, s=fft_size, axes=(-2, -1))

    # Crop to original spatial size
    h_seq = h_seq[..., :H, :W]

    if return_all:
        return h_seq
    else:
        return h_seq[-1]


@partial(jax.jit, static_argnums=(3, 4))
def convssm_parallel_2d_jit(
    x_seq: jnp.ndarray,
    A_kernel: jnp.ndarray,
    B_kernel: jnp.ndarray,
    spatial_size: Tuple[int, int],
    return_all: bool = True,
) -> jnp.ndarray:
    """JIT-compiled parallel 2D ConvSSM."""
    return convssm_parallel_2d(x_seq, A_kernel, B_kernel, spatial_size, return_all)


# =============================================================================
# Sequential ConvSSM (for reference/comparison)
# =============================================================================

def convssm_sequential_3d(
    x_seq: jnp.ndarray,
    A_kernel: jnp.ndarray,
    B_kernel: jnp.ndarray,
    spatial_size: Tuple[int, int, int],
) -> jnp.ndarray:
    """
    Sequential ConvSSM: h_t = A ★ h_{t-1} + B ★ x_t

    Args:
        x_seq: Input sequence (T, B, C, D, H, W)
        A_kernel: Transition kernel (C, Kd, Kh, Kw)
        B_kernel: Input kernel (C, Kd, Kh, Kw)
        spatial_size: (D, H, W)

    Returns:
        h_seq: Hidden states (T, B, C, D, H, W)
    """
    T = x_seq.shape[0]
    B, C = x_seq.shape[1], x_seq.shape[2]
    D, H, W = spatial_size

    def step_fn(h, x_t):
        h_new = fft_conv_3d(h, A_kernel, spatial_size) + fft_conv_3d(x_t, B_kernel, spatial_size)
        return h_new, h_new

    h_init = jnp.zeros((B, C, D, H, W), dtype=x_seq.dtype)
    _, h_seq = lax.scan(step_fn, h_init, x_seq)

    return h_seq


@partial(jax.jit, static_argnums=(3,))
def convssm_sequential_3d_jit(
    x_seq: jnp.ndarray,
    A_kernel: jnp.ndarray,
    B_kernel: jnp.ndarray,
    spatial_size: Tuple[int, int, int],
) -> jnp.ndarray:
    """JIT-compiled sequential ConvSSM."""
    return convssm_sequential_3d(x_seq, A_kernel, B_kernel, spatial_size)


# =============================================================================
# Parallel Scan ConvSSM using lax.associative_scan
# =============================================================================

def parallel_scan_fft_3d(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
) -> jnp.ndarray:
    """
    True O(log T) parallel scan for ConvSSM in frequency domain.

    Uses JAX's lax.associative_scan for efficient parallel prefix computation.

    Args:
        A_f: FFT of A kernel, shape (C, D', H', W') - complex
        B_f: FFT of B kernel, shape (C, D', H', W') - complex
        x_seq_f: FFT of input sequence, shape (T, B, C, D', H', W') - complex

    Returns:
        h_seq_f: FFT of hidden states, shape (T, B, C, D', H', W') - complex
    """
    T = x_seq_f.shape[0]

    # Initialize: a[t] = A_f, s[t] = B_f * x_seq_f[t]
    # Broadcast A_f and B_f to match x_seq_f shape
    a = jnp.broadcast_to(A_f[None, None, ...], x_seq_f.shape)  # (T, B, C, D', H', W')
    s = x_seq_f * B_f[None, None, ...]  # (T, B, C, D', H', W')

    # Parallel scan using associative_scan
    # This computes all prefix sums in O(log T) parallel depth
    _, h_seq_f = lax.associative_scan(_combine_fn, (a, s), axis=0)

    return h_seq_f


def convssm_parallel_3d(
    x_seq: jnp.ndarray,
    A_kernel: jnp.ndarray,
    B_kernel: jnp.ndarray,
    spatial_size: Tuple[int, int, int],
    return_all: bool = True,
) -> jnp.ndarray:
    """
    Parallel ConvSSM using O(log T) associative scan.

    Computes h_t = A ★ h_{t-1} + B ★ x_t for all t in parallel.

    Args:
        x_seq: Input sequence (T, B, C, D, H, W)
        A_kernel: Transition kernel (C, Kd, Kh, Kw)
        B_kernel: Input kernel (C, Kd, Kh, Kw)
        spatial_size: (D, H, W)
        return_all: If True, return all hidden states; if False, return only final

    Returns:
        If return_all: h_seq of shape (T, B, C, D, H, W)
        If not return_all: h_final of shape (B, C, D, H, W)
    """
    D, H, W = spatial_size
    fft_size = (2 * D, 2 * H, 2 * W)

    # FFT all inputs (batched, efficient)
    x_seq_f = jnp.fft.rfftn(x_seq, s=fft_size, axes=(-3, -2, -1))
    A_f = jnp.fft.rfftn(A_kernel, s=fft_size, axes=(-3, -2, -1))
    B_f = jnp.fft.rfftn(B_kernel, s=fft_size, axes=(-3, -2, -1))

    # Parallel scan in frequency domain
    h_seq_f = parallel_scan_fft_3d(A_f, B_f, x_seq_f)

    # IFFT to get spatial domain result
    h_seq = jnp.fft.irfftn(h_seq_f, s=fft_size, axes=(-3, -2, -1))

    # Crop to original spatial size
    h_seq = h_seq[..., :D, :H, :W]

    if return_all:
        return h_seq
    else:
        return h_seq[-1]


@partial(jax.jit, static_argnums=(3, 4))
def convssm_parallel_3d_jit(
    x_seq: jnp.ndarray,
    A_kernel: jnp.ndarray,
    B_kernel: jnp.ndarray,
    spatial_size: Tuple[int, int, int],
    return_all: bool = True,
) -> jnp.ndarray:
    """JIT-compiled parallel ConvSSM."""
    return convssm_parallel_3d(x_seq, A_kernel, B_kernel, spatial_size, return_all)


# =============================================================================
# Convenience class-like interface (functional style)
# =============================================================================

class FlashFFTConv3DJAX:
    """
    JAX implementation of 3D FFT convolution.

    Functional/stateless - just holds configuration.

    Example:
        >>> conv = FlashFFTConv3DJAX(16, 64, 64)
        >>> u = jnp.ones((4, 128, 16, 64, 64))
        >>> k = jnp.ones((128, 3, 3, 3)) * 0.01
        >>> y = conv(u, k)
    """

    def __init__(self, depth: int, height: int, width: int):
        self.depth = depth
        self.height = height
        self.width = width
        self.spatial_size = (depth, height, width)

    def __call__(self, u: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
        return fft_conv_3d_jit(u, k, self.spatial_size)

    def __repr__(self):
        return f"FlashFFTConv3DJAX(depth={self.depth}, height={self.height}, width={self.width})"


class ConvSSMParallelScan3DJAX:
    """
    JAX implementation of parallel scan ConvSSM for 3D.

    Uses lax.associative_scan for true O(log T) parallel depth.

    Example:
        >>> scanner = ConvSSMParallelScan3DJAX(16, 64, 64)
        >>> x_seq = jnp.ones((100, 4, 64, 16, 64, 64))  # T=100
        >>> A_kernel = jnp.ones((64, 3, 3, 3)) * 0.1
        >>> B_kernel = jnp.ones((64, 3, 3, 3)) * 0.1
        >>> h_seq = scanner(x_seq, A_kernel, B_kernel)
    """

    def __init__(self, depth: int, height: int, width: int):
        self.depth = depth
        self.height = height
        self.width = width
        self.spatial_size = (depth, height, width)

    def __call__(
        self,
        x_seq: jnp.ndarray,
        A_kernel: jnp.ndarray,
        B_kernel: jnp.ndarray,
        return_all: bool = True,
    ) -> jnp.ndarray:
        return convssm_parallel_3d_jit(x_seq, A_kernel, B_kernel, self.spatial_size, return_all)

    def sequential(
        self,
        x_seq: jnp.ndarray,
        A_kernel: jnp.ndarray,
        B_kernel: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sequential version for comparison."""
        return convssm_sequential_3d_jit(x_seq, A_kernel, B_kernel, self.spatial_size)

    def __repr__(self):
        return f"ConvSSMParallelScan3DJAX(depth={self.depth}, height={self.height}, width={self.width})"


# =============================================================================
# Gradient support (JAX handles this automatically via autodiff)
# =============================================================================

# JAX automatically supports gradients through jax.grad, jax.value_and_grad
# No need for manual backward implementation!

def convssm_loss_and_grad(
    x_seq: jnp.ndarray,
    A_kernel: jnp.ndarray,
    B_kernel: jnp.ndarray,
    spatial_size: Tuple[int, int, int],
    target: jnp.ndarray,
):
    """
    Example of computing loss and gradients for ConvSSM.

    Args:
        x_seq: Input sequence
        A_kernel, B_kernel: Learnable kernels
        spatial_size: (D, H, W)
        target: Target sequence

    Returns:
        loss, (grad_A, grad_B)
    """
    def loss_fn(A, B):
        h_seq = convssm_parallel_3d(x_seq, A, B, spatial_size, return_all=True)
        return jnp.mean((h_seq - target) ** 2)

    loss, (grad_A, grad_B) = jax.value_and_grad(loss_fn, argnums=(0, 1))(A_kernel, B_kernel)
    return loss, grad_A, grad_B


# =============================================================================
# FOURIER-SPACE ConvSSM (No FFT during forward pass!)
# =============================================================================
#
# Key insight: If inputs are pre-FFT'd and we stay in Fourier space,
# the entire ConvSSM is just element-wise multiplications.
#
# Workflow:
#   1. Pre-FFT dataset once (at data loading time)
#   2. Store kernels in Fourier space
#   3. Forward pass: pure element-wise operations (no FFT!)
#   4. IFFT only at the very end for loss/output
#
# This eliminates ~90% of compute for FFT-heavy workloads.
# =============================================================================

def compute_fft_size(spatial_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Compute FFT size (2x each dimension for linear convolution)."""
    return tuple(2 * s for s in spatial_size)


def compute_rfft_shape(spatial_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Compute output shape of rfftn (last dim is N//2 + 1 for real FFT)."""
    fft_size = compute_fft_size(spatial_size)
    return (fft_size[0], fft_size[1], fft_size[2] // 2 + 1)


def to_fourier_3d(
    x: jnp.ndarray,
    spatial_size: Tuple[int, int, int],
) -> jnp.ndarray:
    """
    Convert spatial-domain tensor to Fourier domain.

    Use this to pre-FFT your dataset once.

    Args:
        x: Spatial tensor of shape (..., D, H, W)
        spatial_size: (D, H, W) - the spatial dimensions

    Returns:
        x_f: Fourier tensor of shape (..., 2D, 2H, W+1) - complex
    """
    fft_size = compute_fft_size(spatial_size)
    return jnp.fft.rfftn(x, s=fft_size, axes=(-3, -2, -1))


def from_fourier_3d(
    x_f: jnp.ndarray,
    spatial_size: Tuple[int, int, int],
) -> jnp.ndarray:
    """
    Convert Fourier-domain tensor back to spatial domain.

    Use this only at the end for loss computation or visualization.

    Args:
        x_f: Fourier tensor of shape (..., 2D, 2H, W+1) - complex
        spatial_size: (D, H, W) - the target spatial dimensions

    Returns:
        x: Spatial tensor of shape (..., D, H, W) - real
    """
    D, H, W = spatial_size
    fft_size = compute_fft_size(spatial_size)
    x = jnp.fft.irfftn(x_f, s=fft_size, axes=(-3, -2, -1))
    return x[..., :D, :H, :W]


def kernel_to_fourier_3d(
    kernel: jnp.ndarray,
    spatial_size: Tuple[int, int, int],
) -> jnp.ndarray:
    """
    Convert spatial-domain kernel to Fourier domain.

    Args:
        kernel: Spatial kernel of shape (C, Kd, Kh, Kw)
        spatial_size: (D, H, W) - the input spatial dimensions

    Returns:
        kernel_f: Fourier kernel of shape (C, 2D, 2H, W+1) - complex
    """
    fft_size = compute_fft_size(spatial_size)
    return jnp.fft.rfftn(kernel, s=fft_size, axes=(-3, -2, -1))


@partial(jax.jit, static_argnums=(1,))
def to_fourier_3d_jit(x: jnp.ndarray, spatial_size: Tuple[int, int, int]) -> jnp.ndarray:
    """JIT-compiled to_fourier_3d."""
    return to_fourier_3d(x, spatial_size)


@partial(jax.jit, static_argnums=(1,))
def from_fourier_3d_jit(x_f: jnp.ndarray, spatial_size: Tuple[int, int, int]) -> jnp.ndarray:
    """JIT-compiled from_fourier_3d."""
    return from_fourier_3d(x_f, spatial_size)


@partial(jax.jit, static_argnums=(1,))
def kernel_to_fourier_3d_jit(kernel: jnp.ndarray, spatial_size: Tuple[int, int, int]) -> jnp.ndarray:
    """JIT-compiled kernel_to_fourier_3d."""
    return kernel_to_fourier_3d(kernel, spatial_size)


# =============================================================================
# 2D FOURIER-SPACE Utilities (for images)
# =============================================================================

def compute_fft_size_2d(spatial_size: Tuple[int, int]) -> Tuple[int, int]:
    """Compute FFT size (2x each dimension for linear convolution)."""
    return (2 * spatial_size[0], 2 * spatial_size[1])


def compute_rfft_shape_2d(spatial_size: Tuple[int, int]) -> Tuple[int, int]:
    """Compute output shape of 2D rfftn (last dim is N//2 + 1 for real FFT)."""
    fft_size = compute_fft_size_2d(spatial_size)
    return (fft_size[0], fft_size[1] // 2 + 1)


def to_fourier_2d(
    x: jnp.ndarray,
    spatial_size: Tuple[int, int],
) -> jnp.ndarray:
    """
    Convert spatial-domain 2D tensor to Fourier domain.

    Use this to pre-FFT your dataset once.

    Args:
        x: Spatial tensor of shape (..., H, W)
        spatial_size: (H, W) - the spatial dimensions

    Returns:
        x_f: Fourier tensor of shape (..., 2H, W+1) - complex
    """
    fft_size = compute_fft_size_2d(spatial_size)
    return jnp.fft.rfftn(x, s=fft_size, axes=(-2, -1))


def from_fourier_2d(
    x_f: jnp.ndarray,
    spatial_size: Tuple[int, int],
) -> jnp.ndarray:
    """
    Convert Fourier-domain 2D tensor back to spatial domain.

    Use this only at the end for loss computation or visualization.

    Args:
        x_f: Fourier tensor of shape (..., 2H, W+1) - complex
        spatial_size: (H, W) - the target spatial dimensions

    Returns:
        x: Spatial tensor of shape (..., H, W) - real
    """
    H, W = spatial_size
    fft_size = compute_fft_size_2d(spatial_size)
    x = jnp.fft.irfftn(x_f, s=fft_size, axes=(-2, -1))
    return x[..., :H, :W]


def kernel_to_fourier_2d(
    kernel: jnp.ndarray,
    spatial_size: Tuple[int, int],
) -> jnp.ndarray:
    """
    Convert spatial-domain 2D kernel to Fourier domain.

    Args:
        kernel: Spatial kernel of shape (C, Kh, Kw)
        spatial_size: (H, W) - the input spatial dimensions

    Returns:
        kernel_f: Fourier kernel of shape (C, 2H, W+1) - complex
    """
    fft_size = compute_fft_size_2d(spatial_size)
    return jnp.fft.rfftn(kernel, s=fft_size, axes=(-2, -1))


@partial(jax.jit, static_argnums=(1,))
def to_fourier_2d_jit(x: jnp.ndarray, spatial_size: Tuple[int, int]) -> jnp.ndarray:
    """JIT-compiled to_fourier_2d."""
    return to_fourier_2d(x, spatial_size)


@partial(jax.jit, static_argnums=(1,))
def from_fourier_2d_jit(x_f: jnp.ndarray, spatial_size: Tuple[int, int]) -> jnp.ndarray:
    """JIT-compiled from_fourier_2d."""
    return from_fourier_2d(x_f, spatial_size)


@partial(jax.jit, static_argnums=(1,))
def kernel_to_fourier_2d_jit(kernel: jnp.ndarray, spatial_size: Tuple[int, int]) -> jnp.ndarray:
    """JIT-compiled kernel_to_fourier_2d."""
    return kernel_to_fourier_2d(kernel, spatial_size)


def convssm_fourier_scan_parallel_2d(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
    return_all: bool = True,
) -> jnp.ndarray:
    """
    2D ConvSSM PARALLEL scan in Fourier space using associative_scan.

    O(log T) depth, O(T log T) total work.
    Best for: small/medium image sizes (≤256x256).

    Args:
        A_f: Fourier-domain A kernel, shape (C, H', W') - complex
        B_f: Fourier-domain B kernel, shape (C, H', W') - complex
        x_seq_f: Fourier-domain input sequence, shape (T, B, C, H', W') - complex
        return_all: If True, return all hidden states; if False, return only final

    Returns:
        h_seq_f: Fourier-domain hidden states
    """
    a = jnp.broadcast_to(A_f[None, None, ...], x_seq_f.shape)
    s = x_seq_f * B_f[None, None, ...]
    _, h_seq_f = lax.associative_scan(_combine_fn, (a, s), axis=0)

    if return_all:
        return h_seq_f
    else:
        return h_seq_f[-1]


def convssm_fourier_scan_sequential_2d(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
    return_all: bool = True,
) -> jnp.ndarray:
    """
    2D ConvSSM SEQUENTIAL scan in Fourier space using lax.scan.

    O(T) depth, O(T) total work.
    Best for: very large image sizes (>256x256).

    Args:
        A_f: Fourier-domain A kernel, shape (C, H', W') - complex
        B_f: Fourier-domain B kernel, shape (C, H', W') - complex
        x_seq_f: Fourier-domain input sequence, shape (T, B, C, H', W') - complex
        return_all: If True, return all hidden states; if False, return only final

    Returns:
        h_seq_f: Fourier-domain hidden states
    """
    def step_fn(h_f, x_t_f):
        h_new_f = A_f * h_f + B_f * x_t_f
        return h_new_f, h_new_f

    h_init_f = jnp.zeros_like(x_seq_f[0])
    h_final_f, h_seq_f = lax.scan(step_fn, h_init_f, x_seq_f)

    if return_all:
        return h_seq_f
    else:
        return h_final_f


def convssm_fourier_scan_2d(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
    return_all: bool = True,
    mode: str = 'auto',
) -> jnp.ndarray:
    """
    2D ConvSSM scan operating ENTIRELY in Fourier space.

    NO FFT/IFFT operations - just element-wise multiplications!

    Args:
        A_f: Fourier-domain A kernel, shape (C, H', W') - complex
        B_f: Fourier-domain B kernel, shape (C, H', W') - complex
        x_seq_f: Fourier-domain input sequence, shape (T, B, C, H', W') - complex
        return_all: If True, return all hidden states; if False, return only final
        mode: 'auto', 'parallel', or 'sequential'
            - 'parallel': O(log T) depth, best for images ≤256x256
            - 'sequential': O(T) depth, best for very large images
            - 'auto': automatically select (uses parallel for most 2D cases)

    Returns:
        h_seq_f: Fourier-domain hidden states
            If return_all: shape (T, B, C, H', W')
            If not return_all: shape (B, C, H', W')
    """
    if mode == 'parallel':
        return convssm_fourier_scan_parallel_2d(A_f, B_f, x_seq_f, return_all)
    elif mode == 'sequential':
        return convssm_fourier_scan_sequential_2d(A_f, B_f, x_seq_f, return_all)
    else:  # auto
        # For 2D images, parallel wins in almost all cases (up to 256x256)
        # Only use sequential for very large images (>512x512)
        spatial_elements = A_f.shape[-2] * A_f.shape[-1]
        if spatial_elements > 500000:  # ~512x512 in FFT space
            return convssm_fourier_scan_sequential_2d(A_f, B_f, x_seq_f, return_all)
        else:
            return convssm_fourier_scan_parallel_2d(A_f, B_f, x_seq_f, return_all)


@jax.jit
def convssm_fourier_scan_2d_jit(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled 2D Fourier-space ConvSSM scan (returns all states, auto mode)."""
    return convssm_fourier_scan_2d(A_f, B_f, x_seq_f, return_all=True, mode='auto')


@jax.jit
def convssm_fourier_scan_2d_final_jit(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled 2D Fourier-space ConvSSM scan (returns only final state)."""
    return convssm_fourier_scan_2d(A_f, B_f, x_seq_f, return_all=False, mode='auto')


class FourierConvSSM2D:
    """
    2D Fourier-space ConvSSM for images.

    Operates entirely without FFT during forward pass - dramatically faster!

    This is designed for the "virtual timesteps" approach where:
    - Input is a static image (B, C, H, W)
    - We run T iterations of SSM to expand receptive field
    - Same input is broadcast to all T timesteps

    Workflow:
        # Setup (once)
        model = FourierConvSSM2D(H, W, C)

        # Pre-FFT image (can replicate across T timesteps)
        x_f = model.precompute_input_fft(x)  # (B, C, H', W')
        x_seq_f = model.broadcast_to_timesteps(x_f, T)  # (T, B, C, H', W')

        # Fast forward pass (no FFT!)
        h_seq_f = model.forward_fourier(A_f, B_f, x_seq_f)

        # Get final hidden state in spatial domain
        h_final = model.to_spatial(h_seq_f[-1])

    Example:
        >>> model = FourierConvSSM2D(224, 224, channels=96)
        >>>
        >>> # Pre-FFT the image once
        >>> x = jnp.ones((4, 96, 224, 224))  # (B, C, H, W)
        >>> x_f = model.precompute_input_fft(x)
        >>> x_seq_f = model.broadcast_to_timesteps(x_f, T=8)
        >>>
        >>> # Initialize kernels in Fourier space
        >>> A_f, B_f = model.init_kernels_fourier(key, kernel_size=7)
        >>>
        >>> # Fast forward (no FFT!)
        >>> h_final_f = model.forward_fourier(A_f, B_f, x_seq_f, return_all=False)
        >>>
        >>> # Convert to spatial
        >>> h_final = model.to_spatial(h_final_f)
    """

    def __init__(self, height: int, width: int, channels: Optional[int] = None):
        self.height = height
        self.width = width
        self.channels = channels
        self.spatial_size = (height, width)
        self.fft_size = compute_fft_size_2d(self.spatial_size)
        self.rfft_shape = compute_rfft_shape_2d(self.spatial_size)

    def precompute_input_fft(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Pre-FFT input image. Do this once per image.

        Args:
            x: Spatial input (B, C, H, W)

        Returns:
            x_f: Fourier input (B, C, H', W') - complex
        """
        return to_fourier_2d_jit(x, self.spatial_size)

    def broadcast_to_timesteps(self, x_f: jnp.ndarray, T: int) -> jnp.ndarray:
        """
        Broadcast single-frame Fourier input to T timesteps.

        For "virtual timesteps" where same image is processed T times.

        Args:
            x_f: Fourier input (B, C, H', W')
            T: Number of timesteps

        Returns:
            x_seq_f: Broadcasted input (T, B, C, H', W')
        """
        return jnp.broadcast_to(x_f[None, ...], (T,) + x_f.shape)

    def init_kernels_fourier(
        self,
        key: jnp.ndarray,
        kernel_size: int = 7,
        channels: Optional[int] = None,
        scale: float = 0.1,
        decay_rate: float = 0.3,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize A and B kernels directly in Fourier space.

        Args:
            key: JAX random key
            kernel_size: Size of spatial kernel (e.g., 7 for 7x7)
            channels: Number of channels (uses self.channels if not provided)
            scale: Scale of random initialization
            decay_rate: Spatial decay rate for stability

        Returns:
            A_f, B_f: Fourier-domain kernels
        """
        from jax import random

        C = channels or self.channels
        if C is None:
            raise ValueError("channels must be provided either in __init__ or here")

        K = kernel_size
        k1, k2 = random.split(key)

        # Initialize in spatial domain with decay
        A_spatial = random.normal(k1, (C, K, K)) * scale
        B_spatial = random.normal(k2, (C, K, K)) * scale

        # Apply decay for stability
        decay = jnp.exp(-decay_rate * jnp.arange(K))
        decay_2d = decay[:, None] * decay[None, :]
        A_spatial = A_spatial * decay_2d
        B_spatial = B_spatial * decay_2d

        # Convert to Fourier
        A_f = kernel_to_fourier_2d_jit(A_spatial, self.spatial_size)
        B_f = kernel_to_fourier_2d_jit(B_spatial, self.spatial_size)

        return A_f, B_f

    def spatial_kernel_to_fourier(self, kernel: jnp.ndarray) -> jnp.ndarray:
        """Convert a spatial kernel to Fourier domain."""
        return kernel_to_fourier_2d_jit(kernel, self.spatial_size)

    def forward_fourier(
        self,
        A_f: jnp.ndarray,
        B_f: jnp.ndarray,
        x_seq_f: jnp.ndarray,
        return_all: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass entirely in Fourier space - NO FFT!

        Args:
            A_f: Fourier-domain A kernel (C, H', W')
            B_f: Fourier-domain B kernel (C, H', W')
            x_seq_f: Fourier-domain input (T, B, C, H', W')
            return_all: Whether to return all timesteps

        Returns:
            h_seq_f: Fourier-domain hidden states
        """
        if return_all:
            return convssm_fourier_scan_2d_jit(A_f, B_f, x_seq_f)
        else:
            return convssm_fourier_scan_2d_final_jit(A_f, B_f, x_seq_f)

    def to_spatial(self, x_f: jnp.ndarray) -> jnp.ndarray:
        """
        Convert Fourier-domain tensor to spatial domain.

        Only call this when you need spatial output (e.g., for loss).
        """
        return from_fourier_2d_jit(x_f, self.spatial_size)

    def to_fourier(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert spatial-domain tensor to Fourier domain."""
        return to_fourier_2d_jit(x, self.spatial_size)

    def __repr__(self):
        return (f"FourierConvSSM2D(height={self.height}, width={self.width}, "
                f"channels={self.channels}, rfft_shape={self.rfft_shape})")


def convssm_fourier_scan_parallel(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
    return_all: bool = True,
) -> jnp.ndarray:
    """
    ConvSSM PARALLEL scan in Fourier space using associative_scan.

    O(log T) depth, O(T log T) total work.
    Best for: small spatial sizes where GPU isn't saturated.

    Args:
        A_f: Fourier-domain A kernel, shape (C, D', H', W') - complex
        B_f: Fourier-domain B kernel, shape (C, D', H', W') - complex
        x_seq_f: Fourier-domain input sequence, shape (T, B, C, D', H', W') - complex
        return_all: If True, return all hidden states; if False, return only final

    Returns:
        h_seq_f: Fourier-domain hidden states
    """
    a = jnp.broadcast_to(A_f[None, None, ...], x_seq_f.shape)
    s = x_seq_f * B_f[None, None, ...]
    _, h_seq_f = lax.associative_scan(_combine_fn, (a, s), axis=0)

    if return_all:
        return h_seq_f
    else:
        return h_seq_f[-1]


def convssm_fourier_scan_sequential(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
    return_all: bool = True,
) -> jnp.ndarray:
    """
    ConvSSM SEQUENTIAL scan in Fourier space using lax.scan.

    O(T) depth, O(T) total work.
    Best for: large spatial sizes where GPU is already saturated.

    Args:
        A_f: Fourier-domain A kernel, shape (C, D', H', W') - complex
        B_f: Fourier-domain B kernel, shape (C, D', H', W') - complex
        x_seq_f: Fourier-domain input sequence, shape (T, B, C, D', H', W') - complex
        return_all: If True, return all hidden states; if False, return only final

    Returns:
        h_seq_f: Fourier-domain hidden states
    """
    def step_fn(h_f, x_t_f):
        # h_t = A_f * h_{t-1} + B_f * x_t (element-wise in frequency domain)
        h_new_f = A_f * h_f + B_f * x_t_f
        return h_new_f, h_new_f

    h_init_f = jnp.zeros_like(x_seq_f[0])
    h_final_f, h_seq_f = lax.scan(step_fn, h_init_f, x_seq_f)

    if return_all:
        return h_seq_f
    else:
        return h_final_f


def convssm_fourier_scan(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
    return_all: bool = True,
    mode: str = 'auto',
) -> jnp.ndarray:
    """
    ConvSSM scan operating ENTIRELY in Fourier space.

    NO FFT/IFFT operations - just element-wise multiplications!

    Args:
        A_f: Fourier-domain A kernel, shape (C, D', H', W') - complex
        B_f: Fourier-domain B kernel, shape (C, D', H', W') - complex
        x_seq_f: Fourier-domain input sequence, shape (T, B, C, D', H', W') - complex
        return_all: If True, return all hidden states; if False, return only final
        mode: 'auto', 'parallel', or 'sequential'
            - 'parallel': O(log T) depth, best for small spatial + large T
            - 'sequential': O(T) depth, best for large spatial (GPU saturated)
            - 'auto': automatically select based on tensor size

    Returns:
        h_seq_f: Fourier-domain hidden states
            If return_all: shape (T, B, C, D', H', W')
            If not return_all: shape (B, C, D', H', W')
    """
    if mode == 'parallel':
        return convssm_fourier_scan_parallel(A_f, B_f, x_seq_f, return_all)
    elif mode == 'sequential':
        return convssm_fourier_scan_sequential(A_f, B_f, x_seq_f, return_all)
    else:  # auto
        # Heuristic: use parallel for small tensors, sequential for large
        # Based on benchmarks: crossover at ~1000 elements per spatial slice
        spatial_elements = A_f.shape[-3] * A_f.shape[-2] * A_f.shape[-1]
        batch_channels = x_seq_f.shape[1] * x_seq_f.shape[2]
        total_elements = spatial_elements * batch_channels

        # If GPU will be saturated with element-wise ops, use sequential
        # (sequential has O(T) work vs O(T log T) for parallel)
        if total_elements > 5000:
            return convssm_fourier_scan_sequential(A_f, B_f, x_seq_f, return_all)
        else:
            return convssm_fourier_scan_parallel(A_f, B_f, x_seq_f, return_all)


@jax.jit
def convssm_fourier_scan_jit(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled Fourier-space ConvSSM scan (returns all states, auto mode)."""
    return convssm_fourier_scan(A_f, B_f, x_seq_f, return_all=True, mode='auto')


@jax.jit
def convssm_fourier_scan_final_jit(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled Fourier-space ConvSSM scan (returns only final state, auto mode)."""
    return convssm_fourier_scan(A_f, B_f, x_seq_f, return_all=False, mode='auto')


@jax.jit
def convssm_fourier_scan_parallel_jit(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled parallel Fourier scan (O(log T) depth)."""
    return convssm_fourier_scan_parallel(A_f, B_f, x_seq_f, return_all=True)


@jax.jit
def convssm_fourier_scan_sequential_jit(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled sequential Fourier scan (O(T) depth, less total work)."""
    return convssm_fourier_scan_sequential(A_f, B_f, x_seq_f, return_all=True)


class FourierConvSSM3D:
    """
    Fourier-space ConvSSM that operates entirely without FFT during forward pass.

    This is dramatically faster than spatial-domain ConvSSM because:
    - FFT of inputs is done ONCE at data loading time
    - Kernels are stored in Fourier domain
    - Forward pass is pure element-wise multiplications
    - IFFT only needed at the very end for loss/output

    Workflow:
        # Setup (once)
        model = FourierConvSSM3D(D, H, W, C)

        # Pre-FFT dataset (once, can be cached to disk)
        x_seq_f = model.precompute_input_fft(x_seq)  # or use to_fourier_3d directly

        # Fast forward pass (no FFT!)
        h_seq_f = model.forward_fourier(x_seq_f)

        # Convert to spatial only when needed
        h_seq = model.to_spatial(h_seq_f)

    Example:
        >>> model = FourierConvSSM3D(16, 64, 64, channels=128)
        >>>
        >>> # Pre-FFT the dataset once
        >>> x_seq = jnp.ones((100, 4, 128, 16, 64, 64))  # (T, B, C, D, H, W)
        >>> x_seq_f = model.precompute_input_fft(x_seq)
        >>>
        >>> # Initialize kernels in Fourier space
        >>> A_f, B_f = model.init_kernels_fourier(key, kernel_size=3)
        >>>
        >>> # Fast forward (no FFT!)
        >>> h_seq_f = model.forward_fourier(A_f, B_f, x_seq_f)
        >>>
        >>> # Convert to spatial only for loss
        >>> h_seq = model.to_spatial(h_seq_f)
    """

    def __init__(self, depth: int, height: int, width: int, channels: Optional[int] = None):
        self.depth = depth
        self.height = height
        self.width = width
        self.channels = channels
        self.spatial_size = (depth, height, width)
        self.fft_size = compute_fft_size(self.spatial_size)
        self.rfft_shape = compute_rfft_shape(self.spatial_size)

    def precompute_input_fft(self, x_seq: jnp.ndarray) -> jnp.ndarray:
        """
        Pre-FFT input sequence. Do this once per dataset.

        Args:
            x_seq: Spatial input (T, B, C, D, H, W)

        Returns:
            x_seq_f: Fourier input (T, B, C, D', H', W') - complex
        """
        return to_fourier_3d_jit(x_seq, self.spatial_size)

    def init_kernels_fourier(
        self,
        key: jnp.ndarray,
        kernel_size: int = 3,
        channels: Optional[int] = None,
        scale: float = 0.1,
        decay_rate: float = 0.3,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize A and B kernels directly in Fourier space.

        Args:
            key: JAX random key
            kernel_size: Size of spatial kernel (e.g., 3 for 3x3x3)
            channels: Number of channels (uses self.channels if not provided)
            scale: Scale of random initialization
            decay_rate: Spatial decay rate for stability

        Returns:
            A_f, B_f: Fourier-domain kernels
        """
        from jax import random

        C = channels or self.channels
        if C is None:
            raise ValueError("channels must be provided either in __init__ or here")

        K = kernel_size
        k1, k2 = random.split(key)

        # Initialize in spatial domain with decay
        A_spatial = random.normal(k1, (C, K, K, K)) * scale
        B_spatial = random.normal(k2, (C, K, K, K)) * scale

        # Apply decay for stability
        decay = jnp.exp(-decay_rate * jnp.arange(K))
        decay_3d = decay[:, None, None] * decay[None, :, None] * decay[None, None, :]
        A_spatial = A_spatial * decay_3d
        B_spatial = B_spatial * decay_3d

        # Convert to Fourier
        A_f = kernel_to_fourier_3d_jit(A_spatial, self.spatial_size)
        B_f = kernel_to_fourier_3d_jit(B_spatial, self.spatial_size)

        return A_f, B_f

    def spatial_kernel_to_fourier(self, kernel: jnp.ndarray) -> jnp.ndarray:
        """Convert a spatial kernel to Fourier domain."""
        return kernel_to_fourier_3d_jit(kernel, self.spatial_size)

    def forward_fourier(
        self,
        A_f: jnp.ndarray,
        B_f: jnp.ndarray,
        x_seq_f: jnp.ndarray,
        return_all: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass entirely in Fourier space - NO FFT!

        Args:
            A_f: Fourier-domain A kernel (C, D', H', W')
            B_f: Fourier-domain B kernel (C, D', H', W')
            x_seq_f: Fourier-domain input (T, B, C, D', H', W')
            return_all: Whether to return all timesteps

        Returns:
            h_seq_f: Fourier-domain hidden states
        """
        if return_all:
            return convssm_fourier_scan_jit(A_f, B_f, x_seq_f)
        else:
            return convssm_fourier_scan_final_jit(A_f, B_f, x_seq_f)

    def to_spatial(self, x_f: jnp.ndarray) -> jnp.ndarray:
        """
        Convert Fourier-domain tensor to spatial domain.

        Only call this when you need spatial output (e.g., for loss).
        """
        return from_fourier_3d_jit(x_f, self.spatial_size)

    def to_fourier(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert spatial-domain tensor to Fourier domain."""
        return to_fourier_3d_jit(x, self.spatial_size)

    def __repr__(self):
        return (f"FourierConvSSM3D(depth={self.depth}, height={self.height}, "
                f"width={self.width}, channels={self.channels}, "
                f"rfft_shape={self.rfft_shape})")


# =============================================================================
# Loss computation in Fourier space
# =============================================================================

def fourier_mse_loss(
    h_seq_f: jnp.ndarray,
    target_f: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute MSE loss directly in Fourier space.

    By Parseval's theorem: ||x||² = (1/N) * ||X||²
    So MSE in spatial domain ≈ MSE in Fourier domain (up to normalization)

    This avoids the IFFT entirely for training!

    Args:
        h_seq_f: Predicted hidden states in Fourier domain
        target_f: Target in Fourier domain

    Returns:
        loss: Scalar MSE loss
    """
    # For complex arrays, we need |a - b|² = (a-b) * conj(a-b)
    diff = h_seq_f - target_f
    # MSE of complex = mean of |diff|²
    return jnp.mean(jnp.abs(diff) ** 2)


def fourier_convssm_loss_and_grad(
    A_f: jnp.ndarray,
    B_f: jnp.ndarray,
    x_seq_f: jnp.ndarray,
    target_f: jnp.ndarray,
):
    """
    Compute loss and gradients entirely in Fourier space.

    No FFT or IFFT anywhere - maximum speed!

    Args:
        A_f, B_f: Fourier-domain kernels
        x_seq_f: Fourier-domain input sequence
        target_f: Fourier-domain target

    Returns:
        loss, (grad_A_f, grad_B_f)
    """
    def loss_fn(A_f, B_f):
        h_seq_f = convssm_fourier_scan(A_f, B_f, x_seq_f, return_all=True)
        return fourier_mse_loss(h_seq_f, target_f)

    loss, (grad_A_f, grad_B_f) = jax.value_and_grad(loss_fn, argnums=(0, 1))(A_f, B_f)
    return loss, grad_A_f, grad_B_f
