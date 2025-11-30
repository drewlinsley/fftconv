"""ConvNeXt with spatial (standard) convolution + ConvSSM.

This is the baseline ConvNeXt architecture with ConvSSM blocks added.
Uses standard Flax nn.Conv for depthwise conv (not FFT).

Architecture per block:
1. Standard depthwise conv (nn.Conv with groups=channels)
2. ConvSSM (T iterations of h = A*h + B*x using spatial convolution)
3. LayerNorm -> MLP -> LayerScale -> Residual
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from typing import Sequence
import numpy as np


# =============================================================================
# Spatial ConvSSM - Using standard convolution (not FFT)
# =============================================================================

class SpatialConvSSM(nn.Module):
    """ConvSSM using standard spatial convolution.

    SSM recurrence: h_t = A * h_{t-1} + B * x
    where * denotes depthwise convolution (using standard nn.Conv).

    This is the reference implementation using spatial domain convolution.

    Attributes:
        dim: Number of channels
        T: Number of SSM timesteps
        kernel_size: Size of A and B convolution kernels
        dtype: Compute dtype
    """
    dim: int
    T: int = 8
    kernel_size: int = 7
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run ConvSSM for T timesteps.

        Args:
            x: (B, H, W, C) spatial input

        Returns:
            (B, H, W, C) output after T SSM iterations
        """
        B, H, W, C = x.shape
        k = self.kernel_size

        # Learn spatial kernels for A and B
        # A: state transition (how much of previous state to keep)
        # B: input mixing (how much of input to add)
        # Kernel shape: (k, k, 1, C) for depthwise with feature_group_count=C
        A_kernel = self.param(
            'A_kernel',
            nn.initializers.normal(0.02),
            (k, k, 1, C),  # Depthwise kernel format: (H, W, in=1, out=C)
            self.dtype
        )
        B_kernel = self.param(
            'B_kernel',
            nn.initializers.normal(0.02),
            (k, k, 1, C),  # Depthwise kernel format: (H, W, in=1, out=C)
            self.dtype
        )

        # Apply tanh to A for stability (keeps eigenvalues bounded)
        # Scale by 0.9 to ensure contraction
        A_kernel_stable = 0.9 * jnp.tanh(A_kernel)

        # Define convolution functions using lax.conv_general_dilated for depthwise
        def apply_A_conv(h):
            """Apply A convolution (depthwise) to hidden state."""
            return lax.conv_general_dilated(
                h,
                A_kernel_stable,
                window_strides=(1, 1),
                padding='SAME',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                feature_group_count=C
            )

        def apply_B_conv(x_input):
            """Apply B convolution (depthwise) to input."""
            return lax.conv_general_dilated(
                x_input,
                B_kernel,
                window_strides=(1, 1),
                padding='SAME',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                feature_group_count=C
            )

        # Pre-compute B * x (doesn't change during scan)
        Bx = apply_B_conv(x)

        # SSM recurrence
        def step_fn(h, _):
            # h_new = A * h + B * x
            h_new = apply_A_conv(h) + Bx
            return h_new, None

        # Initialize hidden state
        h_init = jnp.zeros_like(x)

        # Run T timesteps
        h_final, _ = lax.scan(step_fn, h_init, None, length=self.T)

        return h_final


# =============================================================================
# LayerNorm
# =============================================================================

class LayerNorm(nn.Module):
    """Layer Normalization."""
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.epsilon)

        C = x.shape[-1]
        scale = self.param('scale', nn.initializers.ones, (C,), self.dtype)
        bias = self.param('bias', nn.initializers.zeros, (C,), self.dtype)
        return x * scale + bias


# =============================================================================
# ConvNeXt Block with Spatial Conv + ConvSSM
# =============================================================================

class ConvNeXtSpatialSSMBlock(nn.Module):
    """ConvNeXt block with standard spatial convolution and ConvSSM.

    Architecture:
    1. Standard depthwise conv (7x7)
    2. ConvSSM (T iterations for temporal/spatial mixing)
    3. LayerNorm -> MLP -> LayerScale -> Residual

    The ConvSSM adds recurrent dynamics on top of the static convolution.
    """
    dim: int
    T: int = 8
    kernel_size: int = 7
    expansion_ratio: int = 4
    layer_scale_init: float = 1e-6
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        B, H, W, C = x.shape

        # 1. Standard depthwise conv (same as baseline ConvNeXt)
        x = nn.Conv(
            features=C,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding='SAME',
            feature_group_count=C,  # Depthwise
            dtype=self.dtype,
            name='dwconv'
        )(x)

        # 2. ConvSSM (T iterations of h = A*h + B*x)
        x = SpatialConvSSM(
            dim=C,
            T=self.T,
            kernel_size=self.kernel_size,
            dtype=self.dtype,
            name='convssm'
        )(x)

        # 3. LayerNorm
        x = LayerNorm(dtype=self.dtype)(x)

        # 4. Pointwise MLP
        hidden_dim = int(C * self.expansion_ratio)
        x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(C, dtype=self.dtype)(x)

        # 5. Layer scale
        gamma = self.param(
            'layer_scale',
            nn.initializers.constant(self.layer_scale_init),
            (C,),
            self.dtype
        )
        x = x * gamma

        # 6. Stochastic depth
        if train and self.drop_path_rate > 0:
            keep_prob = 1.0 - self.drop_path_rate
            rng = self.make_rng('dropout')
            mask = jax.random.bernoulli(rng, keep_prob, (B, 1, 1, 1))
            x = x / keep_prob * mask

        return x + residual


class Downsample(nn.Module):
    """Spatial downsampling."""
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


# =============================================================================
# Full Model
# =============================================================================

class ConvNeXtSpatialSSM(nn.Module):
    """ConvNeXt with standard spatial convolution + ConvSSM.

    Same architecture as ConvNeXt-Tiny but with:
    - Standard Flax nn.Conv for depthwise conv (baseline behavior)
    - ConvSSM added after depthwise conv for recurrent dynamics
    """
    num_classes: int = 1000
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    T: int = 8
    kernel_size: int = 7
    drop_path_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Stem
        x = nn.Conv(
            self.dims[0],
            kernel_size=(4, 4),
            strides=(4, 4),
            padding='VALID',
            dtype=self.dtype,
            name='stem'
        )(x)
        x = LayerNorm(dtype=self.dtype, name='stem_norm')(x)

        # Stochastic depth schedule
        total_blocks = sum(self.depths)
        dpr = [self.drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0

        # Four stages
        for stage_idx, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            if stage_idx > 0:
                x = Downsample(dim, dtype=self.dtype, name=f'downsample_{stage_idx}')(x)

            for block_i in range(depth):
                x = ConvNeXtSpatialSSMBlock(
                    dim=dim,
                    T=self.T,
                    kernel_size=self.kernel_size,
                    drop_path_rate=dpr[block_idx],
                    dtype=self.dtype,
                    name=f'stage_{stage_idx}_block_{block_i}'
                )(x, train=train)
                block_idx += 1

        # Head
        x = jnp.mean(x, axis=(1, 2))
        x = LayerNorm(dtype=self.dtype, name='head_norm')(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype, name='head')(x)

        return x


def convnext_spatial_ssm_tiny(num_classes: int = 1000, T: int = 8, **kwargs) -> ConvNeXtSpatialSSM:
    """ConvNeXt-Spatial-SSM-Tiny"""
    return ConvNeXtSpatialSSM(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        T=T,
        **kwargs
    )


# =============================================================================
# Tests
# =============================================================================

if __name__ == '__main__':
    import jax.random as random

    print("=" * 70)
    print("TEST: ConvNeXt-Spatial-SSM Model")
    print("=" * 70)

    key = random.PRNGKey(0)

    # Test with T=8
    print("\nT=8 (default):")
    model = convnext_spatial_ssm_tiny(num_classes=10, T=8)
    dummy = jnp.ones((2, 224, 224, 3))
    variables = model.init({'params': key, 'dropout': key}, dummy, train=False)
    params = variables['params']

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Forward pass
    logits = model.apply({'params': params}, dummy, train=False, rngs={'dropout': key})
    print(f"Output shape: {logits.shape}")
    print(f"Output mean: {logits.mean():.6f}, std: {logits.std():.6f}")

    # Gradient check
    def model_loss(params, x):
        logits = model.apply({'params': params}, x, train=True, rngs={'dropout': key})
        return jnp.mean(logits ** 2)

    loss, grads = jax.value_and_grad(model_loss)(params, dummy)
    total_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))

    print(f"Model loss: {loss:.6f}")
    print(f"Total gradient norm: {total_grad_norm:.6f}")

    # Test with T=1 (minimal SSM)
    print("\nT=1 (minimal SSM):")
    model_t1 = convnext_spatial_ssm_tiny(num_classes=10, T=1)
    variables_t1 = model_t1.init({'params': key, 'dropout': key}, dummy, train=False)
    n_params_t1 = sum(x.size for x in jax.tree_util.tree_leaves(variables_t1['params']))
    print(f"Parameters: {n_params_t1:,} ({n_params_t1 / 1e6:.1f}M)")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
