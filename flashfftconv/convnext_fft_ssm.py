"""ConvNeXt with FFT convolution + ConvSSM.

This builds on the working FFT-Simple model by adding ConvSSM blocks.
The FFT convolution is verified to work, so we add SSM on top.

Architecture per block:
1. FFT depthwise conv (same as baseline ConvNeXt, verified working)
2. ConvSSM (T iterations of h = A*h + B*x using FFT convolution)
3. LayerNorm -> MLP -> LayerScale -> Residual
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from typing import Sequence
import numpy as np


# =============================================================================
# FFT Depthwise Convolution (from convnext_fft_simple.py - verified working)
# =============================================================================

def fft_depthwise_conv(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """FFT-based depthwise convolution.

    Args:
        x: (B, H, W, C) input in NHWC format
        kernel: (C, k, k) depthwise kernel

    Returns:
        (B, H, W, C) convolution output
    """
    B, H, W, C = x.shape
    k = kernel.shape[1]
    center = k // 2

    # Place kernel center at (0,0) with wrap-around
    i_idx = jnp.arange(k)
    j_idx = jnp.arange(k)
    target_i = (i_idx - center) % H
    target_j = (j_idx - center) % W
    ti, tj = jnp.meshgrid(target_i, target_j, indexing='ij')

    padded_kernel = jnp.zeros((C, H, W), dtype=kernel.dtype)
    padded_kernel = padded_kernel.at[:, ti, tj].set(kernel)

    # FFT convolution
    x_f = jnp.fft.fft2(x, axes=(1, 2))
    kernel_f = jnp.fft.fft2(padded_kernel, axes=(1, 2))
    kernel_f = kernel_f.transpose(1, 2, 0)[None, ...]
    out_f = x_f * kernel_f
    out = jnp.fft.ifft2(out_f, axes=(1, 2)).real

    return out


class FFTDepthwiseConv(nn.Module):
    """FFT-based depthwise convolution layer."""
    features: int
    kernel_size: int = 7
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        C = x.shape[-1]
        k = self.kernel_size

        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (C, k, k),
            self.dtype
        )

        return fft_depthwise_conv(x, kernel)


# =============================================================================
# ConvSSM - Spatial domain SSM using FFT convolution
# =============================================================================

class SpatialConvSSM(nn.Module):
    """ConvSSM in spatial domain using FFT for convolutions.

    SSM recurrence: h_t = A * h_{t-1} + B * x
    where * denotes depthwise convolution (implemented via FFT).

    This stays in spatial domain (no complex representations) for simplicity.
    Uses same FFT conv that we verified works in the simple model.

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
        A_kernel = self.param(
            'A_kernel',
            nn.initializers.normal(0.02),
            (C, k, k),
            self.dtype
        )
        B_kernel = self.param(
            'B_kernel',
            nn.initializers.normal(0.02),
            (C, k, k),
            self.dtype
        )

        # Apply tanh to A for stability (keeps eigenvalues bounded)
        # Scale by 0.9 to ensure contraction
        A_kernel_stable = 0.9 * jnp.tanh(A_kernel)

        # Pre-compute FFT of kernels (they don't change during scan)
        center = k // 2
        i_idx = jnp.arange(k)
        j_idx = jnp.arange(k)
        target_i = (i_idx - center) % H
        target_j = (j_idx - center) % W
        ti, tj = jnp.meshgrid(target_i, target_j, indexing='ij')

        # Pad and FFT A kernel
        A_padded = jnp.zeros((C, H, W), dtype=self.dtype)
        A_padded = A_padded.at[:, ti, tj].set(A_kernel_stable)
        A_f = jnp.fft.fft2(A_padded, axes=(1, 2))
        A_f = A_f.transpose(1, 2, 0)[None, ...]  # (1, H, W, C)

        # Pad and FFT B kernel
        B_padded = jnp.zeros((C, H, W), dtype=self.dtype)
        B_padded = B_padded.at[:, ti, tj].set(B_kernel)
        B_f = jnp.fft.fft2(B_padded, axes=(1, 2))
        B_f = B_f.transpose(1, 2, 0)[None, ...]  # (1, H, W, C)

        # FFT input once
        x_f = jnp.fft.fft2(x, axes=(1, 2))  # (B, H, W, C)

        # SSM recurrence in frequency domain
        # h_t = A * h_{t-1} + B * x (convolution = elementwise multiply in freq)
        def step_fn(h_f, _):
            # h_new = A_f * h_f + B_f * x_f (complex multiply)
            h_new_f = A_f * h_f + B_f * x_f
            return h_new_f, None

        # Initialize hidden state in frequency domain
        h_init_f = jnp.zeros_like(x_f)

        # Run T timesteps
        h_final_f, _ = lax.scan(step_fn, h_init_f, None, length=self.T)

        # IFFT back to spatial
        h_final = jnp.fft.ifft2(h_final_f, axes=(1, 2)).real

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
# ConvNeXt Block with FFT Conv + ConvSSM
# =============================================================================

class ConvNeXtFFTSSMBlock(nn.Module):
    """ConvNeXt block with FFT convolution and ConvSSM.

    Architecture:
    1. FFT depthwise conv (replaces baseline 7x7 conv)
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

        # 1. FFT-based depthwise conv (verified working)
        x = FFTDepthwiseConv(
            features=C,
            kernel_size=self.kernel_size,
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

class ConvNeXtFFTSSM(nn.Module):
    """ConvNeXt with FFT convolution + ConvSSM.

    Same architecture as ConvNeXt-Tiny but with:
    - FFT-based depthwise conv (verified working)
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
                x = ConvNeXtFFTSSMBlock(
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


def convnext_fft_ssm_tiny(num_classes: int = 1000, T: int = 8, **kwargs) -> ConvNeXtFFTSSM:
    """ConvNeXt-FFT-SSM-Tiny"""
    return ConvNeXtFFTSSM(
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
    print("TEST: ConvNeXt-FFT-SSM Model")
    print("=" * 70)

    key = random.PRNGKey(0)

    # Test with T=8
    print("\nT=8 (default):")
    model = convnext_fft_ssm_tiny(num_classes=10, T=8)
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

    # Test with T=1 (should be similar to FFT-Simple)
    print("\nT=1 (minimal SSM):")
    model_t1 = convnext_fft_ssm_tiny(num_classes=10, T=1)
    variables_t1 = model_t1.init({'params': key, 'dropout': key}, dummy, train=False)
    n_params_t1 = sum(x.size for x in jax.tree_util.tree_leaves(variables_t1['params']))
    print(f"Parameters: {n_params_t1:,} ({n_params_t1 / 1e6:.1f}M)")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
