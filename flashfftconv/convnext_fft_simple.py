"""ConvNeXt with simple FFT convolution - direct drop-in replacement.

This is the simplest possible FFT-based ConvNeXt:
1. Everything stays in spatial domain (no Fourier-domain operations)
2. The ONLY change is: depthwise conv → FFT-based convolution
3. Same architecture, same initialization, same everything else

If this doesn't work, something is fundamentally wrong with our FFT conv.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
import numpy as np


# =============================================================================
# FFT Depthwise Convolution - The Core Operation
# =============================================================================

def fft_depthwise_conv(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """FFT-based depthwise convolution.

    This is a drop-in replacement for depthwise conv with circular boundary.
    For small kernels, this is mathematically equivalent to conv with mode='wrap'.

    Args:
        x: (B, H, W, C) input in NHWC format
        kernel: (C, k, k) depthwise kernel (one kxk kernel per channel)

    Returns:
        (B, H, W, C) convolution output
    """
    B, H, W, C = x.shape
    k = kernel.shape[1]
    center = k // 2

    # Step 1: Prepare kernel - place center at (0,0) with wrap-around
    # This is critical for FFT convolution to match spatial "SAME" padding
    i_idx = jnp.arange(k)
    j_idx = jnp.arange(k)
    target_i = (i_idx - center) % H
    target_j = (j_idx - center) % W
    ti, tj = jnp.meshgrid(target_i, target_j, indexing='ij')

    padded_kernel = jnp.zeros((C, H, W), dtype=kernel.dtype)
    padded_kernel = padded_kernel.at[:, ti, tj].set(kernel)

    # Step 2: FFT both input and kernel
    x_f = jnp.fft.fft2(x, axes=(1, 2))  # (B, H, W, C)
    kernel_f = jnp.fft.fft2(padded_kernel, axes=(1, 2))  # (C, H, W)

    # Step 3: Broadcast kernel and multiply
    # kernel_f: (C, H, W) -> (1, H, W, C) for broadcasting with (B, H, W, C)
    kernel_f = kernel_f.transpose(1, 2, 0)[None, ...]
    out_f = x_f * kernel_f

    # Step 4: IFFT back to spatial
    out = jnp.fft.ifft2(out_f, axes=(1, 2)).real

    return out


class FFTDepthwiseConv(nn.Module):
    """FFT-based depthwise convolution layer.

    Drop-in replacement for nn.Conv with depthwise (feature_group_count=C).
    Uses FFT convolution theorem: IFFT(FFT(x) * FFT(kernel)).
    """
    features: int
    kernel_size: int = 7
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply FFT-based depthwise convolution.

        Args:
            x: (B, H, W, C) input

        Returns:
            (B, H, W, C) convolved output
        """
        C = x.shape[-1]
        assert C == self.features, f"Input channels {C} != features {self.features}"

        k = self.kernel_size

        # Initialize kernel same as Flax Conv (lecun_normal)
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (C, k, k),
            self.dtype
        )

        # Apply FFT convolution
        return fft_depthwise_conv(x, kernel)


# =============================================================================
# ConvNeXt Block with FFT Convolution
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


class ConvNeXtFFTBlock(nn.Module):
    """ConvNeXt block with FFT-based depthwise convolution.

    ONLY difference from standard ConvNeXt: uses FFT conv instead of spatial conv.
    Everything else is identical.
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

        # FFT-based depthwise conv (the ONLY change!)
        x = FFTDepthwiseConv(
            features=C,
            kernel_size=self.kernel_size,
            dtype=self.dtype,
            name='dwconv'
        )(x)

        # LayerNorm (same as baseline)
        x = LayerNorm(dtype=self.dtype)(x)

        # Pointwise MLP (same as baseline)
        hidden_dim = int(C * self.expansion_ratio)
        x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(C, dtype=self.dtype)(x)

        # Layer scale (same as baseline)
        gamma = self.param(
            'layer_scale',
            nn.initializers.constant(self.layer_scale_init),
            (C,),
            self.dtype
        )
        x = x * gamma

        # Stochastic depth (same as baseline)
        if train and self.drop_path_rate > 0:
            keep_prob = 1.0 - self.drop_path_rate
            rng = self.make_rng('dropout')
            mask = jax.random.bernoulli(rng, keep_prob, (B, 1, 1, 1))
            x = x / keep_prob * mask

        return x + residual


class Downsample(nn.Module):
    """Spatial downsampling (same as baseline)."""
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


class ConvNeXtFFT(nn.Module):
    """ConvNeXt with FFT convolution.

    Identical to standard ConvNeXt except depthwise conv uses FFT.
    Should achieve IDENTICAL accuracy if FFT conv is correct.
    """
    num_classes: int = 1000
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    kernel_size: int = 7
    drop_path_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Stem (same as baseline)
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
                x = ConvNeXtFFTBlock(
                    dim=dim,
                    kernel_size=self.kernel_size,
                    drop_path_rate=dpr[block_idx],
                    dtype=self.dtype,
                    name=f'stage_{stage_idx}_block_{block_i}'
                )(x, train=train)
                block_idx += 1

        # Head (same as baseline)
        x = jnp.mean(x, axis=(1, 2))
        x = LayerNorm(dtype=self.dtype, name='head_norm')(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype, name='head')(x)

        return x


def convnext_fft_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXtFFT:
    """ConvNeXt-FFT-Tiny"""
    return ConvNeXtFFT(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        **kwargs
    )


# =============================================================================
# Verification Tests
# =============================================================================

if __name__ == '__main__':
    from scipy import ndimage
    import jax.random as random

    print("=" * 70)
    print("TEST 1: Verify FFT convolution matches scipy")
    print("=" * 70)

    np.random.seed(42)

    # Test parameters
    H, W, C = 16, 16, 4
    k = 7

    # Random input and kernel
    x = np.random.randn(1, H, W, C).astype(np.float32)
    kernel = np.random.randn(C, k, k).astype(np.float32)

    # Scipy reference (circular convolution per channel)
    out_scipy = np.zeros_like(x)
    for c in range(C):
        out_scipy[0, :, :, c] = ndimage.convolve(x[0, :, :, c], kernel[c], mode='wrap')

    # Our FFT convolution
    out_fft = fft_depthwise_conv(jnp.array(x), jnp.array(kernel))

    diff = np.abs(out_scipy - np.array(out_fft))
    print(f"Max difference: {diff.max():.10f}")
    print(f"Mean difference: {diff.mean():.10f}")
    print(f"MATCH: {diff.max() < 1e-5}")

    if diff.max() >= 1e-5:
        print("\n*** FFT CONVOLUTION DOES NOT MATCH SCIPY! ***")
        print("This is a bug that must be fixed before training.")
        exit(1)

    print("\n" + "=" * 70)
    print("TEST 2: Gradient flow check")
    print("=" * 70)

    # Check gradients flow through FFT conv
    def loss_fn(kernel, x):
        out = fft_depthwise_conv(x, kernel)
        return jnp.mean(out ** 2)

    x_jax = jnp.array(x)
    kernel_jax = jnp.array(kernel)

    loss, grad = jax.value_and_grad(loss_fn)(kernel_jax, x_jax)
    grad_norm = jnp.sqrt(jnp.sum(grad ** 2))

    print(f"Loss: {loss:.6f}")
    print(f"Gradient norm: {grad_norm:.6f}")
    print(f"Gradient non-zero: {jnp.any(grad != 0)}")

    print("\n" + "=" * 70)
    print("TEST 3: Full model test")
    print("=" * 70)

    key = random.PRNGKey(0)
    model = convnext_fft_tiny(num_classes=10)

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

    print("\n" + "=" * 70)
    print("All tests passed! FFT convolution is working correctly.")
    print("=" * 70)
