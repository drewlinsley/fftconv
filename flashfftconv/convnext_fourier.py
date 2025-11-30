# ConvNeXt-Fourier: Fully Fourier-domain ConvNeXt
# Pre-FFT input once, process entirely in Fourier space, iFFT at end
#
# Key idea: Avoid repeated FFT/iFFT per layer by staying in Fourier domain.
# Convolutions become element-wise multiplications.
# Downsampling = crop high frequencies (efficient!)
#
# Architecture:
#   Input: (B, H, W, 3) real image
#   Stem: Conv4x4_s4 -> LayerNorm -> FFT2D [to Fourier]
#   Stages: All processing in Fourier domain
#   Head: iFFT2D -> GlobalPool -> Linear [back to spatial for classification]

from typing import Sequence, Optional, Tuple
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn


# =============================================================================
# Fourier-domain utilities
# =============================================================================

def to_fourier_2d(x: jnp.ndarray) -> jnp.ndarray:
    """Convert spatial tensor to Fourier domain.

    Args:
        x: Real tensor (B, C, H, W) channels-first

    Returns:
        Complex tensor (B, C, H, W) in Fourier domain
    """
    return jnp.fft.fft2(x, axes=(-2, -1))


def from_fourier_2d(x_f: jnp.ndarray) -> jnp.ndarray:
    """Convert Fourier tensor back to spatial domain.

    Args:
        x_f: Complex tensor (B, C, H, W) in Fourier domain

    Returns:
        Real tensor (B, C, H, W) in spatial domain
    """
    return jnp.fft.ifft2(x_f, axes=(-2, -1)).real


def fourier_downsample(x_f: jnp.ndarray, factor: int = 2) -> jnp.ndarray:
    """Downsample in Fourier domain by cropping high frequencies.

    This is equivalent to low-pass filtering + downsampling.
    Much more efficient than spatial conv + stride!

    Args:
        x_f: Complex tensor (B, C, H, W) in Fourier domain
        factor: Downsampling factor (default 2)

    Returns:
        Complex tensor (B, C, H//factor, W//factor)
    """
    B, C, H, W = x_f.shape
    new_H, new_W = H // factor, W // factor

    # FFT2 puts DC at [0,0] and frequencies wrap around
    # To downsample, we need to keep the low frequencies
    # For fftshift'd data, this would be the center
    # For non-shifted, we need corners

    # Keep low frequencies (DC + nearby)
    # Top-left and corners contain low frequencies in non-shifted FFT
    h_keep = new_H // 2
    w_keep = new_W // 2

    # Crop: keep corners (low frequencies)
    top_left = x_f[:, :, :h_keep, :w_keep]
    top_right = x_f[:, :, :h_keep, -w_keep:]
    bottom_left = x_f[:, :, -h_keep:, :w_keep]
    bottom_right = x_f[:, :, -h_keep:, -w_keep:]

    # Reconstruct smaller Fourier tensor
    top = jnp.concatenate([top_left, top_right], axis=-1)
    bottom = jnp.concatenate([bottom_left, bottom_right], axis=-1)
    x_f_down = jnp.concatenate([top, bottom], axis=-2)

    # Scale by 1/factor^2 to preserve energy
    x_f_down = x_f_down / (factor * factor)

    return x_f_down


# =============================================================================
# Fourier-domain layers
# =============================================================================

class FourierLayerNorm(nn.Module):
    """Layer normalization in Fourier domain.

    Operates on the magnitude/phase representation or real/imag separately.
    """
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_f: jnp.ndarray) -> jnp.ndarray:
        # x_f is complex (B, C, H, W) in channels-first format
        # Normalize real and imaginary parts separately per-channel

        C = x_f.shape[1]

        # Split into real and imaginary
        x_real = x_f.real
        x_imag = x_f.imag

        # Normalize each over spatial dims (keep B, C)
        mean_real = jnp.mean(x_real, axis=(-2, -1), keepdims=True)
        var_real = jnp.var(x_real, axis=(-2, -1), keepdims=True)
        x_real = (x_real - mean_real) / jnp.sqrt(var_real + self.epsilon)

        mean_imag = jnp.mean(x_imag, axis=(-2, -1), keepdims=True)
        var_imag = jnp.var(x_imag, axis=(-2, -1), keepdims=True)
        x_imag = (x_imag - mean_imag) / jnp.sqrt(var_imag + self.epsilon)

        # Learnable scale (per channel)
        scale = self.param('scale', nn.initializers.ones, (C, 1, 1), self.dtype)

        x_real = x_real * scale
        x_imag = x_imag * scale

        return x_real + 1j * x_imag


class FourierConvSSM(nn.Module):
    """ConvSSM that operates in Fourier domain with small spatial kernels.

    EFFICIENT VERSION: Learns small spatial kernels (k x k) instead of full
    resolution Fourier filters (H x W). The kernels are FFT'd at runtime.

    This reduces parameters from O(C * H * W) to O(C * k * k):
    - Old: 96 channels * 56 * 56 = 301,056 params
    - New: 96 channels * 7 * 7 = 4,704 params (64x reduction!)

    Attributes:
        dim: Number of channels
        T: Number of SSM iterations
        kernel_size: Size of learned spatial kernel (default 7, like ConvNeXt)
        dtype: Computation dtype
    """
    dim: int
    T: int = 8
    kernel_size: int = 7
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_f: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x_f: Complex tensor (B, C, H, W) in Fourier domain

        Returns:
            Complex tensor (B, C, H, W) in Fourier domain
        """
        B, C, H, W = x_f.shape
        k = self.kernel_size

        # Learn small spatial kernels (depthwise: one kernel per channel)
        # Shape: (C, k, k) instead of (C, H, W) - MUCH smaller!
        A_kernel = self.param(
            'A_kernel',
            nn.initializers.normal(stddev=0.02),
            (C, k, k),
            self.dtype
        )
        B_kernel = self.param(
            'B_kernel',
            nn.initializers.normal(stddev=0.02),
            (C, k, k),
            self.dtype
        )

        # Convert kernels to Fourier domain
        # Shapes are concrete during tracing, so we can use Python if
        if H >= k and W >= k:
            # Pad kernel to target size (normal case - most stages)
            pad_h = (H - k) // 2
            pad_w = (W - k) // 2
            A_padded = jnp.pad(
                A_kernel,
                ((0, 0), (pad_h, H - k - pad_h), (pad_w, W - k - pad_w)),
                mode='constant',
                constant_values=0
            )
            B_padded = jnp.pad(
                B_kernel,
                ((0, 0), (pad_h, H - k - pad_h), (pad_w, W - k - pad_w)),
                mode='constant',
                constant_values=0
            )
        else:
            # Crop kernel center to target size (last stage with 7x7 features)
            start_h = (k - H) // 2
            start_w = (k - W) // 2
            A_padded = A_kernel[:, start_h:start_h+H, start_w:start_w+W]
            B_padded = B_kernel[:, start_h:start_h+H, start_w:start_w+W]

        A_f = jnp.fft.fft2(A_padded, axes=(-2, -1))
        B_f = jnp.fft.fft2(B_padded, axes=(-2, -1))

        # Apply stability constraint: |A_f| < 1 for stability
        # Use sigmoid to bound magnitude
        A_f_mag = jnp.abs(A_f)
        A_f_phase = jnp.angle(A_f)
        A_f_mag_bounded = 0.95 * jax.nn.sigmoid(A_f_mag)  # max ~0.95
        A_f = A_f_mag_bounded * jnp.exp(1j * A_f_phase)

        # Run SSM for T steps using lax.scan (faster compilation than for-loop)
        # h_t = A_f * h_{t-1} + B_f * x_f
        def step_fn(h, _):
            h_new = A_f * h + B_f * x_f
            return h_new, None

        h_init = jnp.zeros_like(x_f)
        h_final, _ = lax.scan(step_fn, h_init, None, length=self.T)

        return h_final


class FourierBlock(nn.Module):
    """ConvNeXt block operating in Fourier domain.

    Structure:
        x_f -> FourierConvSSM -> FourierNorm -> PointwiseMLP -> + residual

    Note: We do PointwiseMLP in Fourier domain by applying Dense layers
    to each spatial location independently (this works because Dense
    operates on the channel dimension, and Fourier transform is linear).

    Attributes:
        dim: Number of channels
        T: SSM iterations
        kernel_size: Size of SSM spatial kernel (default 7, like ConvNeXt)
        expansion_ratio: MLP expansion (default 4)
        layer_scale_init: Layer scale initial value
        drop_path_rate: Stochastic depth rate
        use_checkpoint: If True, use gradient checkpointing to save memory (default True)
    """
    dim: int
    T: int = 8
    kernel_size: int = 7
    expansion_ratio: int = 4
    layer_scale_init: float = 1e-6
    drop_path_rate: float = 0.0
    use_checkpoint: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_f: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x_f: Complex tensor (B, C, H, W) in Fourier domain, channels-first

        Returns:
            Complex tensor (B, C, H, W)
        """
        residual = x_f
        B, C, H, W = x_f.shape

        # ConvSSM in Fourier domain (with optional gradient checkpointing)
        SSMClass = FourierConvSSM
        if self.use_checkpoint:
            SSMClass = nn.remat(FourierConvSSM)
        x_f = SSMClass(dim=C, T=self.T, kernel_size=self.kernel_size, dtype=self.dtype)(x_f)

        # Fourier LayerNorm
        x_f = FourierLayerNorm(dtype=self.dtype)(x_f)

        # Pointwise MLP (operates on each frequency independently)
        # Transpose to (B, H, W, C) for Dense layers, then back
        x_f = jnp.transpose(x_f, (0, 2, 3, 1))  # (B, H, W, C)

        # Split to real/imag, apply MLP to each
        x_real = x_f.real
        x_imag = x_f.imag

        hidden_dim = int(C * self.expansion_ratio)

        # Expand
        x_real = nn.Dense(hidden_dim, dtype=self.dtype, name='mlp_up_real')(x_real)
        x_imag = nn.Dense(hidden_dim, dtype=self.dtype, name='mlp_up_imag')(x_imag)

        # Activation (GELU on magnitude, preserve phase approximately)
        # Simpler: just apply to real/imag separately
        x_real = nn.gelu(x_real)
        x_imag = nn.gelu(x_imag)

        # Project back
        x_real = nn.Dense(C, dtype=self.dtype, name='mlp_down_real')(x_real)
        x_imag = nn.Dense(C, dtype=self.dtype, name='mlp_down_imag')(x_imag)

        x_f = x_real + 1j * x_imag

        # Back to channels-first
        x_f = jnp.transpose(x_f, (0, 3, 1, 2))  # (B, C, H, W)

        # Layer scale
        gamma_real = self.param(
            'layer_scale_real',
            nn.initializers.constant(self.layer_scale_init),
            (C, 1, 1),
            self.dtype
        )
        gamma_imag = self.param(
            'layer_scale_imag',
            nn.initializers.constant(0.0),  # Start with real scale
            (C, 1, 1),
            self.dtype
        )
        gamma = gamma_real + 1j * gamma_imag
        x_f = x_f * gamma

        # Stochastic depth
        if train and self.drop_path_rate > 0:
            keep_prob = 1.0 - self.drop_path_rate
            rng = self.make_rng('dropout')
            mask = jax.random.bernoulli(rng, keep_prob, (B, 1, 1, 1))
            x_f = x_f / keep_prob * mask

        # Residual
        x_f = x_f + residual

        return x_f


class FourierDownsample(nn.Module):
    """Downsample in Fourier domain by cropping frequencies.

    Also does channel projection via complex linear.
    """
    out_dim: int
    factor: int = 2
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_f: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x_f: Complex tensor (B, C_in, H, W)

        Returns:
            Complex tensor (B, C_out, H//factor, W//factor)
        """
        B, C_in, H, W = x_f.shape

        # Frequency cropping for spatial downsampling
        x_f = fourier_downsample(x_f, self.factor)

        # Channel projection (complex linear)
        # Transpose to (B, H, W, C) for Dense
        x_f = jnp.transpose(x_f, (0, 2, 3, 1))

        x_real = nn.Dense(self.out_dim, dtype=self.dtype, name='proj_real')(x_f.real)
        x_imag = nn.Dense(self.out_dim, dtype=self.dtype, name='proj_imag')(x_f.imag)
        x_f = x_real + 1j * x_imag

        # Back to channels-first
        x_f = jnp.transpose(x_f, (0, 3, 1, 2))

        return x_f


# =============================================================================
# ConvNeXt-Fourier Model
# =============================================================================

class ConvNeXtFourier(nn.Module):
    """
    Fully Fourier-domain ConvNeXt.

    FFT once at input, process entirely in Fourier domain, iFFT at end.
    This avoids repeated FFT/iFFT operations per layer.

    Architecture:
        Input: (B, H, W, 3) - NHWC real image
        Stem: Conv4x4_s4 -> LayerNorm -> FFT2D (to Fourier domain)
        Stage 1-4: FourierBlocks + FourierDownsample
        Head: iFFT2D -> GlobalAvgPool -> Linear (classification)

    Attributes:
        num_classes: Number of output classes
        depths: Number of blocks per stage
        dims: Channel dimensions per stage
        T: SSM iterations per block
        drop_path_rate: Maximum stochastic depth rate
        use_checkpoint: If True, use gradient checkpointing (default True)
    """
    num_classes: int = 1000
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    T: int = 8
    drop_path_rate: float = 0.1
    use_checkpoint: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input image (B, H, W, 3) - NHWC format
            train: Training mode

        Returns:
            Logits (B, num_classes)
        """
        # Stem: Spatial conv to get initial features, then FFT
        # Input: (B, H, W, 3) -> (B, H/4, W/4, dims[0])
        x = nn.Conv(
            self.dims[0],
            kernel_size=(4, 4),
            strides=(4, 4),
            padding='VALID',
            dtype=self.dtype,
            name='stem_conv'
        )(x)

        # LayerNorm before going to Fourier
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-6)

        scale = self.param('stem_ln_scale', nn.initializers.ones, (self.dims[0],), self.dtype)
        bias = self.param('stem_ln_bias', nn.initializers.zeros, (self.dims[0],), self.dtype)
        x = x * scale + bias

        # Convert to channels-first and FFT
        x = jnp.transpose(x, (0, 3, 1, 2))  # (B, C, H, W)
        x_f = to_fourier_2d(x)  # Complex tensor

        # Calculate drop path rates for each block (use numpy for concrete values)
        total_blocks = sum(self.depths)
        dp_rates = np.linspace(0, self.drop_path_rate, total_blocks).tolist()

        # Stages
        block_idx = 0
        for stage_idx, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            # Downsample between stages (except first)
            if stage_idx > 0:
                x_f = FourierDownsample(
                    out_dim=dim,
                    factor=2,
                    dtype=self.dtype,
                    name=f'downsample_{stage_idx}'
                )(x_f)

            # Blocks in this stage
            for block_i in range(depth):
                x_f = FourierBlock(
                    dim=dim,
                    T=self.T,
                    drop_path_rate=dp_rates[block_idx],
                    use_checkpoint=self.use_checkpoint,
                    dtype=self.dtype,
                    name=f'stage{stage_idx}_block{block_i}'
                )(x_f, train=train)
                block_idx += 1

        # Head: Convert back to spatial domain
        x = from_fourier_2d(x_f)  # (B, C, H, W) real

        # Global average pooling
        x = jnp.mean(x, axis=(-2, -1))  # (B, C)

        # Final LayerNorm
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-6)

        head_scale = self.param('head_ln_scale', nn.initializers.ones, (self.dims[-1],), self.dtype)
        head_bias = self.param('head_ln_bias', nn.initializers.zeros, (self.dims[-1],), self.dtype)
        x = x * head_scale + head_bias

        # Classification head
        x = nn.Dense(self.num_classes, dtype=self.dtype, name='head')(x)

        return x


# =============================================================================
# Model variants
# =============================================================================

def convnext_fourier_tiny(
    num_classes: int = 1000,
    T: int = 8,
    drop_path_rate: float = 0.1,
    dtype: jnp.dtype = jnp.float32,
    use_checkpoint: bool = True,
) -> ConvNeXtFourier:
    """ConvNeXt-Fourier-Tiny: ~28M parameters."""
    return ConvNeXtFourier(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        T=T,
        drop_path_rate=drop_path_rate,
        dtype=dtype,
        use_checkpoint=use_checkpoint,
    )


def convnext_fourier_small(
    num_classes: int = 1000,
    T: int = 8,
    drop_path_rate: float = 0.3,
    dtype: jnp.dtype = jnp.float32,
    use_checkpoint: bool = True,
) -> ConvNeXtFourier:
    """ConvNeXt-Fourier-Small: ~50M parameters."""
    return ConvNeXtFourier(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        T=T,
        drop_path_rate=drop_path_rate,
        dtype=dtype,
        use_checkpoint=use_checkpoint,
    )


def convnext_fourier_base(
    num_classes: int = 1000,
    T: int = 8,
    drop_path_rate: float = 0.4,
    dtype: jnp.dtype = jnp.float32,
    use_checkpoint: bool = True,
) -> ConvNeXtFourier:
    """ConvNeXt-Fourier-Base: ~89M parameters."""
    return ConvNeXtFourier(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        T=T,
        drop_path_rate=drop_path_rate,
        dtype=dtype,
        use_checkpoint=use_checkpoint,
    )


# =============================================================================
# PRE-FFT Model: Accepts pre-FFT'd input from data loader
# =============================================================================

class ConvNeXtFourierPreFFT(nn.Module):
    """
    ConvNeXt that accepts PRE-FFT'd images from data loader.

    NO FFT operations in forward pass at all!
    - Data loader FFTs images using numpy (outside JAX)
    - Model receives complex Fourier tensors
    - All processing in Fourier domain
    - Only iFFT at the very end for classification

    Input: Complex tensor (B, C, H, W) already in Fourier domain
           where C=3 (RGB), H=W=224 (or any size)

    Stem: Frequency cropping (downsample 4x) + channel projection
    """
    num_classes: int = 1000
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    T: int = 8
    drop_path_rate: float = 0.1
    use_checkpoint: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_f: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x_f: PRE-FFT'd complex tensor (B, C, H, W) in Fourier domain
                 Already FFT'd by data loader using numpy!
            train: Training mode

        Returns:
            Logits (B, num_classes)
        """
        # Input is already (B, C, H, W) complex, C=3 channels
        B, C_in, H, W = x_f.shape

        # Stem: Frequency cropping (4x downsample) + channel projection
        # This replaces the spatial Conv4x4_s4
        x_f = fourier_downsample(x_f, factor=4)  # (B, 3, H/4, W/4)

        # Channel projection 3 -> dims[0] (complex linear)
        x_f = jnp.transpose(x_f, (0, 2, 3, 1))  # (B, H/4, W/4, 3)

        x_real = nn.Dense(self.dims[0], dtype=self.dtype, name='stem_real')(x_f.real)
        x_imag = nn.Dense(self.dims[0], dtype=self.dtype, name='stem_imag')(x_f.imag)
        x_f = x_real + 1j * x_imag  # (B, H/4, W/4, dims[0])

        # Back to channels-first
        x_f = jnp.transpose(x_f, (0, 3, 1, 2))  # (B, dims[0], H/4, W/4)

        # Fourier LayerNorm on stem output
        x_f = FourierLayerNorm(dtype=self.dtype, name='stem_ln')(x_f)

        # Calculate drop path rates (use numpy for concrete values)
        total_blocks = sum(self.depths)
        dp_rates = np.linspace(0, self.drop_path_rate, total_blocks).tolist()

        # Stages
        block_idx = 0
        for stage_idx, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            # Downsample between stages (except first)
            if stage_idx > 0:
                x_f = FourierDownsample(
                    out_dim=dim,
                    factor=2,
                    dtype=self.dtype,
                    name=f'downsample_{stage_idx}'
                )(x_f)

            # Blocks
            for block_i in range(depth):
                x_f = FourierBlock(
                    dim=dim,
                    T=self.T,
                    drop_path_rate=dp_rates[block_idx],
                    use_checkpoint=self.use_checkpoint,
                    dtype=self.dtype,
                    name=f'stage{stage_idx}_block{block_i}'
                )(x_f, train=train)
                block_idx += 1

        # Head: Convert back to spatial ONLY for final classification
        x = from_fourier_2d(x_f)  # (B, C, H, W) real

        # Global average pooling
        x = jnp.mean(x, axis=(-2, -1))  # (B, C)

        # Final LayerNorm
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-6)

        head_scale = self.param('head_ln_scale', nn.initializers.ones, (self.dims[-1],), self.dtype)
        head_bias = self.param('head_ln_bias', nn.initializers.zeros, (self.dims[-1],), self.dtype)
        x = x * head_scale + head_bias

        # Classification
        x = nn.Dense(self.num_classes, dtype=self.dtype, name='head')(x)

        return x


def convnext_fourier_prefft_tiny(
    num_classes: int = 1000,
    T: int = 8,
    drop_path_rate: float = 0.1,
    dtype: jnp.dtype = jnp.float32,
    use_checkpoint: bool = True,
) -> ConvNeXtFourierPreFFT:
    """ConvNeXt-Fourier-PreFFT-Tiny: Accepts pre-FFT'd input."""
    return ConvNeXtFourierPreFFT(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        T=T,
        drop_path_rate=drop_path_rate,
        dtype=dtype,
        use_checkpoint=use_checkpoint,
    )
