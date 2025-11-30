# Standard ConvNeXt baseline (no SSM, no FFT)
# Uses standard 7x7 depthwise convolution as in the original ConvNeXt paper
#
# This serves as the baseline to compare against our ConvNeXt-SSM and ConvNeXt-Fourier models.

from typing import Sequence, Tuple, Any

import jax
import jax.numpy as jnp
import flax.linen as nn


class LayerNorm(nn.Module):
    """Layer Normalization with optional bias."""
    epsilon: float = 1e-6
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
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


class ConvNeXtBlock(nn.Module):
    """
    Standard ConvNeXt block with 7x7 depthwise convolution.

    Structure:
        x -> DWConv 7x7 -> LayerNorm -> 1x1 (expand 4x) -> GELU -> 1x1 (project) -> + residual

    This is the original ConvNeXt design.

    Attributes:
        dim: Number of channels
        kernel_size: Depthwise conv kernel size (default: 7)
        expansion_ratio: MLP expansion ratio (default: 4)
        layer_scale_init: Initial value for layer scale (default: 1e-6)
        drop_path_rate: Stochastic depth rate
        dtype: Computation dtype
    """
    dim: int
    kernel_size: int = 7
    expansion_ratio: int = 4
    layer_scale_init: float = 1e-6
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        B, H, W, C = x.shape

        # Depthwise 7x7 conv (standard ConvNeXt)
        x = nn.Conv(
            features=C,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding='SAME',
            feature_group_count=C,  # Depthwise
            dtype=self.dtype,
            name='dwconv'
        )(x)

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
        x = nn.Conv(
            self.out_dim,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='VALID',
            dtype=self.dtype
        )(x)
        return x


class ConvNeXt(nn.Module):
    """
    Standard ConvNeXt model.

    This is the original ConvNeXt architecture with 7x7 depthwise convolutions.
    Serves as a baseline for comparison with ConvNeXt-SSM and ConvNeXt-Fourier.

    Attributes:
        num_classes: Number of output classes
        depths: Number of blocks per stage (default: [3, 3, 9, 3] for Tiny)
        dims: Channel dimensions per stage (default: [96, 192, 384, 768])
        kernel_size: Depthwise conv kernel size (default: 7)
        drop_path_rate: Maximum stochastic depth rate
        dtype: Computation dtype
    """
    num_classes: int = 1000
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    kernel_size: int = 7
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

            # Stack of ConvNeXt blocks
            for block_i in range(depth):
                x = ConvNeXtBlock(
                    dim=dim,
                    kernel_size=self.kernel_size,
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

def convnext_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    """ConvNeXt-Tiny: ~28M params"""
    return ConvNeXt(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        **kwargs
    )


def convnext_small(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    """ConvNeXt-Small: ~50M params"""
    return ConvNeXt(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        **kwargs
    )


def convnext_base(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    """ConvNeXt-Base: ~89M params"""
    return ConvNeXt(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        **kwargs
    )


def convnext_large(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    """ConvNeXt-Large: ~198M params"""
    return ConvNeXt(
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


if __name__ == '__main__':
    import jax.random as random

    print("Testing ConvNeXt-Tiny (standard baseline)...")

    key = random.PRNGKey(0)
    model = convnext_tiny(num_classes=1000)

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
