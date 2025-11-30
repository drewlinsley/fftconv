# ConvNeXt-SSM: ConvNeXt with Mamba-style FFT ConvSSM
# Replaces ConvNeXt's 7x7 depthwise conv with a linear recurrence using FFT convolution
#
# Architecture (ConvNeXt-Tiny):
#   - Depths: [3, 3, 9, 3]
#   - Channels: [96, 192, 384, 768]
#   - SSM: T=8 virtual timesteps, 7x7 kernel, unidirectional
#
# Key insight: For static images, we run T iterations of the SSM to expand
# the receptive field. This is equivalent to running a recurrence:
#   h_t = A ★ h_{t-1} + B ★ x
# where A and B are 7x7 kernels (like depthwise conv, one per channel).

from typing import Sequence, Optional, Tuple, Any
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn

from flashfftconv.conv_nd_jax import (
    convssm_parallel_2d,
    convssm_sequential_2d,
    FourierConvSSM2D,
    to_fourier_2d,
    from_fourier_2d,
    kernel_to_fourier_2d,
    _combine_fn,
)


# =============================================================================
# ConvSSM Layer (replaces depthwise 7x7 conv)
# =============================================================================

class ConvSSMBlock(nn.Module):
    """
    Mamba-style SSM using FFT convolution.

    Replaces ConvNeXt's 7x7 depthwise conv with a recurrence:
        h_t = A ★ h_{t-1} + B ★ x

    For T iterations, this expands the receptive field to ~T*kernel_size.

    Attributes:
        dim: Number of channels (depthwise, one SSM per channel)
        kernel_size: Size of convolution kernel (default: 7)
        T: Number of SSM iterations (virtual timesteps)
        use_parallel: If True, use O(log T) parallel scan; if False, use sequential
        dtype: Computation dtype
    """
    dim: int
    kernel_size: int = 7
    T: int = 8
    use_parallel: bool = True  # Parallel is faster for most 2D image sizes
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tensor (B, H, W, C) - NHWC format for compatibility
            train: Training mode (unused, kept for API compatibility)

        Returns:
            Output tensor (B, H, W, C)
        """
        B, H, W, C = x.shape

        # Initialize A and B kernels (per-channel, like depthwise conv)
        # A: transition kernel (controls decay/propagation)
        # B: input kernel (controls input mixing)
        A_kernel = self.param(
            'A_kernel',
            nn.initializers.normal(stddev=0.02),
            (C, self.kernel_size, self.kernel_size),
            self.dtype
        )
        B_kernel = self.param(
            'B_kernel',
            nn.initializers.normal(stddev=0.02),
            (C, self.kernel_size, self.kernel_size),
            self.dtype
        )

        # Apply exponential decay for stability (prevents explosion over T steps)
        decay_rate = 0.3
        decay = jnp.exp(-decay_rate * jnp.arange(self.kernel_size, dtype=self.dtype))
        decay_2d = decay[:, None] * decay[None, :]
        A_kernel = A_kernel * decay_2d

        # Convert to channels-first for FFT conv: (B, H, W, C) -> (B, C, H, W)
        x = jnp.transpose(x, (0, 3, 1, 2))

        # Create sequence: replicate same input for T timesteps
        # Shape: (T, B, C, H, W)
        x_seq = jnp.broadcast_to(x[None, ...], (self.T, B, C, H, W))

        # Run ConvSSM
        if self.use_parallel:
            h_seq = convssm_parallel_2d(x_seq, A_kernel, B_kernel, (H, W), return_all=True)
        else:
            h_seq = convssm_sequential_2d(x_seq, A_kernel, B_kernel, (H, W))

        # Take final hidden state: (T, B, C, H, W) -> (B, C, H, W)
        h_final = h_seq[-1]

        # Convert back to channels-last: (B, C, H, W) -> (B, H, W, C)
        h_final = jnp.transpose(h_final, (0, 2, 3, 1))

        return h_final


class ConvSSMBlockFourier(nn.Module):
    """
    Fourier-space ConvSSM - operates entirely in frequency domain.

    Faster variant that pre-FFTs the input once and operates in Fourier space.
    Best for when you want maximum speed and can afford the memory for complex tensors.

    Attributes:
        dim: Number of channels
        kernel_size: Size of convolution kernel (default: 7)
        T: Number of SSM iterations
        dtype: Computation dtype
    """
    dim: int
    kernel_size: int = 7
    T: int = 8
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tensor (B, H, W, C) - NHWC format
            train: Training mode

        Returns:
            Output tensor (B, H, W, C)
        """
        B, H, W, C = x.shape
        spatial_size = (H, W)
        fft_size = (2 * H, 2 * W)

        # Initialize kernels in spatial domain
        A_kernel = self.param(
            'A_kernel',
            nn.initializers.normal(stddev=0.02),
            (C, self.kernel_size, self.kernel_size),
            self.dtype
        )
        B_kernel = self.param(
            'B_kernel',
            nn.initializers.normal(stddev=0.02),
            (C, self.kernel_size, self.kernel_size),
            self.dtype
        )

        # Apply decay
        decay_rate = 0.3
        decay = jnp.exp(-decay_rate * jnp.arange(self.kernel_size, dtype=self.dtype))
        decay_2d = decay[:, None] * decay[None, :]
        A_kernel = A_kernel * decay_2d

        # Convert to channels-first
        x = jnp.transpose(x, (0, 3, 1, 2))  # (B, C, H, W)

        # Pre-FFT input (done once, not per timestep!)
        x_f = jnp.fft.rfftn(x, s=fft_size, axes=(-2, -1))  # (B, C, H', W')

        # FFT kernels
        A_f = jnp.fft.rfftn(A_kernel, s=fft_size, axes=(-2, -1))  # (C, H', W')
        B_f = jnp.fft.rfftn(B_kernel, s=fft_size, axes=(-2, -1))  # (C, H', W')

        # Create sequence in Fourier domain
        x_seq_f = jnp.broadcast_to(x_f[None, ...], (self.T, B, C) + x_f.shape[-2:])

        # Parallel scan in Fourier domain (O(log T) depth, no FFT during scan!)
        a = jnp.broadcast_to(A_f[None, None, ...], x_seq_f.shape)
        s = x_seq_f * B_f[None, None, ...]
        _, h_seq_f = lax.associative_scan(_combine_fn, (a, s), axis=0)

        # Take final state and IFFT back to spatial
        h_final_f = h_seq_f[-1]
        h_final = jnp.fft.irfftn(h_final_f, s=fft_size, axes=(-2, -1))
        h_final = h_final[..., :H, :W]

        # Convert back to channels-last
        h_final = jnp.transpose(h_final, (0, 2, 3, 1))

        return h_final


# =============================================================================
# ConvNeXt Building Blocks
# =============================================================================

class LayerNorm(nn.Module):
    """Layer Normalization with optional bias."""
    epsilon: float = 1e-6
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Normalize over channel dimension (last axis for NHWC)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.epsilon)

        C = x.shape[-1]
        scale = self.param('scale', nn.initializers.ones, (C,), self.dtype)
        x = x * scale

        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (C,), self.dtype)
            x = x + bias

        return x


class ConvNeXtSSMBlock(nn.Module):
    """
    ConvNeXt block with SSM replacing depthwise conv.

    Structure:
        x -> ConvSSM -> LayerNorm -> 1x1 (expand 4x) -> GELU -> 1x1 (project) -> + residual

    This follows ConvNeXt's inverted bottleneck design with layer scale.

    Attributes:
        dim: Number of channels
        kernel_size: SSM kernel size (default: 7)
        T: SSM iterations (default: 8)
        expansion_ratio: MLP expansion ratio (default: 4)
        layer_scale_init: Initial value for layer scale (default: 1e-6)
        use_fourier: Use Fourier-space SSM (faster but more memory)
        drop_path_rate: Stochastic depth rate
        dtype: Computation dtype
    """
    dim: int
    kernel_size: int = 7
    T: int = 8
    expansion_ratio: int = 4
    layer_scale_init: float = 1e-6
    use_fourier: bool = True
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        B, H, W, C = x.shape

        # ConvSSM (replaces depthwise 7x7)
        if self.use_fourier:
            x = ConvSSMBlockFourier(
                dim=C,
                kernel_size=self.kernel_size,
                T=self.T,
                dtype=self.dtype
            )(x, train=train)
        else:
            x = ConvSSMBlock(
                dim=C,
                kernel_size=self.kernel_size,
                T=self.T,
                use_parallel=True,
                dtype=self.dtype
            )(x, train=train)

        # LayerNorm
        x = LayerNorm(dtype=self.dtype)(x)

        # Pointwise expansion (1x1 conv, 4x channels)
        hidden_dim = int(C * self.expansion_ratio)
        x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)

        # Pointwise projection (1x1 conv, back to C)
        x = nn.Dense(C, dtype=self.dtype)(x)

        # Layer scale
        gamma = self.param(
            'layer_scale',
            nn.initializers.constant(self.layer_scale_init),
            (C,),
            self.dtype
        )
        x = x * gamma

        # Stochastic depth (drop path)
        if train and self.drop_path_rate > 0:
            keep_prob = 1.0 - self.drop_path_rate
            rng = self.make_rng('dropout')
            mask = jax.random.bernoulli(rng, keep_prob, (B, 1, 1, 1))
            x = x / keep_prob * mask

        # Residual
        x = x + residual

        return x


class Downsample(nn.Module):
    """Spatial downsampling: LayerNorm + 2x2 strided conv."""
    out_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = LayerNorm(dtype=self.dtype)(x)
        # 2x2 conv with stride 2 for downsampling
        x = nn.Conv(
            self.out_dim,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='VALID',
            dtype=self.dtype
        )(x)
        return x


# =============================================================================
# ConvNeXt-SSM Model
# =============================================================================

class ConvNeXtSSM(nn.Module):
    """
    ConvNeXt-SSM: ConvNeXt backbone with Mamba-style FFT ConvSSM.

    Replaces all 7x7 depthwise convolutions with a linear recurrence
    computed via FFT convolution. This gives O(log T) parallel depth
    while expanding the receptive field by T times.

    Attributes:
        num_classes: Number of output classes (1000 for ImageNet)
        depths: Number of blocks per stage (default: [3, 3, 9, 3] for Tiny)
        dims: Channel dimensions per stage (default: [96, 192, 384, 768])
        kernel_size: SSM kernel size (default: 7)
        T: SSM iterations per block (default: 8)
        use_fourier: Use Fourier-space SSM (faster)
        drop_path_rate: Maximum stochastic depth rate
        dtype: Computation dtype
    """
    num_classes: int = 1000
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    kernel_size: int = 7
    T: int = 8
    use_fourier: bool = True
    drop_path_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input images (B, H, W, 3) in NHWC format
            train: Training mode

        Returns:
            Logits (B, num_classes)
        """
        # Stem: patchify with 4x4 conv, stride 4
        x = nn.Conv(
            self.dims[0],
            kernel_size=(4, 4),
            strides=(4, 4),
            padding='VALID',
            dtype=self.dtype,
            name='stem'
        )(x)
        x = LayerNorm(dtype=self.dtype, name='stem_norm')(x)

        # Stochastic depth schedule (linear increase)
        total_blocks = sum(self.depths)
        dpr = [self.drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0

        # Four stages
        for stage_idx, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            # Downsample between stages (except first)
            if stage_idx > 0:
                x = Downsample(dim, dtype=self.dtype, name=f'downsample_{stage_idx}')(x)

            # Stack of ConvNeXt-SSM blocks
            for block_i in range(depth):
                x = ConvNeXtSSMBlock(
                    dim=dim,
                    kernel_size=self.kernel_size,
                    T=self.T,
                    use_fourier=self.use_fourier,
                    drop_path_rate=dpr[block_idx],
                    dtype=self.dtype,
                    name=f'stage_{stage_idx}_block_{block_i}'
                )(x, train=train)
                block_idx += 1

        # Head: global average pool + LayerNorm + classifier
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = LayerNorm(dtype=self.dtype, name='head_norm')(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype, name='head')(x)

        return x


# =============================================================================
# Model Variants
# =============================================================================

def convnext_ssm_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXtSSM:
    """ConvNeXt-SSM-Tiny: ~28M params"""
    return ConvNeXtSSM(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        **kwargs
    )


def convnext_ssm_small(num_classes: int = 1000, **kwargs) -> ConvNeXtSSM:
    """ConvNeXt-SSM-Small: ~50M params"""
    return ConvNeXtSSM(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        **kwargs
    )


def convnext_ssm_base(num_classes: int = 1000, **kwargs) -> ConvNeXtSSM:
    """ConvNeXt-SSM-Base: ~89M params"""
    return ConvNeXtSSM(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        **kwargs
    )


def convnext_ssm_large(num_classes: int = 1000, **kwargs) -> ConvNeXtSSM:
    """ConvNeXt-SSM-Large: ~198M params"""
    return ConvNeXtSSM(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        **kwargs
    )


# =============================================================================
# Utility Functions
# =============================================================================

def count_params(params) -> int:
    """Count total number of parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def create_model_and_init(
    model_fn,
    rng: jax.Array,
    input_shape: Tuple[int, ...] = (1, 224, 224, 3),
    **kwargs
) -> Tuple[nn.Module, Any]:
    """
    Create model and initialize parameters.

    Args:
        model_fn: Model constructor (e.g., convnext_ssm_tiny)
        rng: JAX random key
        input_shape: Input shape (B, H, W, C)
        **kwargs: Additional model arguments

    Returns:
        (model, params)
    """
    model = model_fn(**kwargs)

    # Initialize
    dummy_input = jnp.ones(input_shape)
    variables = model.init({'params': rng, 'dropout': rng}, dummy_input, train=False)

    return model, variables['params']


if __name__ == '__main__':
    # Quick test
    import jax.random as random

    print("Testing ConvNeXt-SSM-Tiny...")

    key = random.PRNGKey(0)
    model = convnext_ssm_tiny(num_classes=1000, T=8)

    # Initialize
    dummy = jnp.ones((2, 224, 224, 3))
    variables = model.init({'params': key, 'dropout': key}, dummy, train=False)
    params = variables['params']

    n_params = count_params(params)
    print(f"  Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Forward pass
    logits = model.apply({'params': params}, dummy, train=False, rngs={'dropout': key})
    print(f"  Input: {dummy.shape}")
    print(f"  Output: {logits.shape}")

    # Check gradients
    def loss_fn(params, x, y):
        logits = model.apply({'params': params}, x, train=True, rngs={'dropout': key})
        return jnp.mean((logits - y) ** 2)

    y = jnp.zeros((2, 1000))
    loss, grads = jax.value_and_grad(loss_fn)(params, dummy, y)
    print(f"  Loss: {loss:.4f}")
    print(f"  Gradient norm: {jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))):.4f}")

    print("\nAll tests passed!")
