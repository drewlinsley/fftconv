"""ConvNeXt with 3D ConvSSM - treating images as pseudo-videos.

Key idea: Repeat input image T times to create (B, T, H, W, C), then use
3D ConvSSM layers throughout the network. The SSM recurrence naturally
operates over the temporal dimension T.

This is different from the 2D ConvSSM approach where T was an internal
iteration count. Here T is an actual spatial dimension that gets processed
by 3D convolutions and 3D FFTs.

Architecture:
1. Input (B, H, W, 3) -> repeat T times -> (B, T, H, W, 3)
2. 3D stem conv: (B, T, H/4, W/4, C)
3. 3D ConvSSM blocks with 3D depthwise conv + 3D FFT SSM
4. Pool over (T, H, W) -> (B, C) -> classifier

Starting with basic Parallel SSM (no gates) per user request.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from typing import Sequence
import numpy as np


# =============================================================================
# Helper Functions for 3D FFT Operations
# =============================================================================

def kernel_to_freq_3d(kernel: jnp.ndarray, T: int, H: int, W: int) -> jnp.ndarray:
    """Convert small 3D spatial kernel to frequency domain.

    Places kernel center at (0,0,0) with wrap-around for proper FFT convolution.

    Args:
        kernel: (C, k_t, k_h, k_w) 3D spatial kernel
        T, H, W: target spatial dimensions

    Returns:
        (C, T, H, W) complex frequency representation
    """
    C, k_t, k_h, k_w = kernel.shape
    center_t = k_t // 2
    center_h = k_h // 2
    center_w = k_w // 2

    # Place kernel center at (0,0,0) with wrap-around
    t_idx = jnp.arange(k_t)
    h_idx = jnp.arange(k_h)
    w_idx = jnp.arange(k_w)

    target_t = (t_idx - center_t) % T
    target_h = (h_idx - center_h) % H
    target_w = (w_idx - center_w) % W

    # Create meshgrid for indexing
    tt, th, tw = jnp.meshgrid(target_t, target_h, target_w, indexing='ij')

    # Zero-pad and place kernel
    padded = jnp.zeros((C, T, H, W), dtype=kernel.dtype)
    padded = padded.at[:, tt, th, tw].set(kernel)

    # 3D FFT
    return jnp.fft.fftn(padded, axes=(1, 2, 3))


def ssm_associative_op(left, right):
    """Associative operation for linear recurrence.

    For h_t = a_t * h_{t-1} + b_t, the associative op is:
    (a1, b1) ⊕ (a2, b2) = (a1 * a2, a2 * b1 + b2)

    This allows parallel prefix computation in O(log T) steps.
    """
    a_left, b_left = left
    a_right, b_right = right
    return (a_left * a_right, a_right * b_left + b_right)


# =============================================================================
# 3D Parallel ConvSSM (Basic - No Gates)
# =============================================================================

class ParallelConvSSM3D(nn.Module):
    """3D Parallel ConvSSM operating on (T, H, W) dimensions.

    Uses 3D FFT for efficient convolution and associative scan for
    O(log T) parallel recurrence computation.

    The recurrence h_t = a * h_{t-1} + b operates in 3D frequency domain:
    - a = A_kernel_f (learned 3D kernel in freq domain)
    - b = B_kernel_f * x_f (input modulated by B kernel)

    Attributes:
        dim: Number of channels
        kernel_size: Spatial kernel size (H, W dimensions)
        kernel_size_t: Temporal kernel size (T dimension)
        dtype: Compute dtype
    """
    dim: int
    kernel_size: int = 7  # Spatial (H, W)
    kernel_size_t: int = 5  # Temporal (T)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run 3D parallel ConvSSM.

        Args:
            x: (B, T, H, W, C) 3D spatial input

        Returns:
            (B, T, H, W, C) output after SSM iterations
        """
        B, T, H, W, C = x.shape
        k = self.kernel_size  # Spatial (H, W)
        k_t = self.kernel_size_t  # Temporal (T)

        # Learnable 3D convolution kernels with anisotropic shape (k_t, k, k)
        # A_kernel: state transition (how previous state influences current)
        # B_kernel: input mixing (how input enters the state)
        # Shape: (C, k_t, k_h, k_w) = (C, 5, 7, 7) for 7x7x5
        A_kernel = self.param(
            'A_kernel',
            nn.initializers.lecun_normal(),
            (C, k_t, k, k),
            self.dtype
        )
        B_kernel = self.param(
            'B_kernel',
            nn.initializers.lecun_normal(),
            (C, k_t, k, k),
            self.dtype
        )

        # Convert kernels to frequency domain
        A_f = kernel_to_freq_3d(A_kernel, T, H, W)  # (C, T, H, W) complex
        B_f = kernel_to_freq_3d(B_kernel, T, H, W)  # (C, T, H, W) complex

        # Reshape for broadcasting: (C, T, H, W) -> (T, H, W, C)
        A_f = A_f.transpose(1, 2, 3, 0)  # (T, H, W, C)
        B_f = B_f.transpose(1, 2, 3, 0)  # (T, H, W, C)

        # 3D FFT of input
        x_f = jnp.fft.fftn(x, axes=(1, 2, 3))  # (B, T, H, W, C) complex

        # SSM coefficients in frequency domain
        # a = A_f (state transition)
        # b = B_f * x_f (input contribution)
        a = A_f[None, ...]  # (1, T, H, W, C) -> broadcast to (B, T, H, W, C)
        b = B_f[None, ...] * x_f  # (B, T, H, W, C)

        # For parallel scan, we need to scan over one dimension
        # We'll scan over T (the temporal/depth dimension)
        # Reshape: (B, T, H, W, C) -> (T, B, H, W, C) for scan over T

        a_seq = jnp.broadcast_to(a, (B, T, H, W, C)).transpose(1, 0, 2, 3, 4)  # (T, B, H, W, C)
        b_seq = b.transpose(1, 0, 2, 3, 4)  # (T, B, H, W, C)

        # Parallel associative scan over T dimension
        _, h_all_f = lax.associative_scan(
            ssm_associative_op,
            (a_seq, b_seq),
            axis=0
        )

        # Reshape back: (T, B, H, W, C) -> (B, T, H, W, C)
        h_f = h_all_f.transpose(1, 0, 2, 3, 4)

        # 3D IFFT back to spatial domain
        h = jnp.fft.ifftn(h_f, axes=(1, 2, 3)).real

        return h.astype(self.dtype)


# =============================================================================
# 3D FFT Depthwise Convolution
# =============================================================================

class FFTDepthwiseConv3D(nn.Module):
    """3D FFT-based depthwise convolution layer with anisotropic kernels."""
    features: int
    kernel_size: int = 7  # Spatial (H, W)
    kernel_size_t: int = 5  # Temporal (T)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, T, H, W, C) input

        Returns:
            (B, T, H, W, C) convolved output
        """
        B, T, H, W, C = x.shape
        k = self.kernel_size  # Spatial
        k_t = self.kernel_size_t  # Temporal

        # Anisotropic kernel: (C, k_t, k, k) = (C, 5, 7, 7) for 7x7x5
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (C, k_t, k, k),
            self.dtype
        )

        # Convert kernel to frequency domain
        kernel_f = kernel_to_freq_3d(kernel, T, H, W)  # (C, T, H, W)
        kernel_f = kernel_f.transpose(1, 2, 3, 0)[None, ...]  # (1, T, H, W, C)

        # 3D FFT convolution
        x_f = jnp.fft.fftn(x, axes=(1, 2, 3))
        out_f = x_f * kernel_f
        out = jnp.fft.ifftn(out_f, axes=(1, 2, 3)).real

        return out.astype(self.dtype)


# =============================================================================
# LayerNorm for 3D
# =============================================================================

class LayerNorm3D(nn.Module):
    """Layer Normalization for 3D inputs (normalizes over channel dim)."""
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, T, H, W, C) - normalize over C
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.epsilon)

        C = x.shape[-1]
        scale = self.param('scale', nn.initializers.ones, (C,), self.dtype)
        bias = self.param('bias', nn.initializers.zeros, (C,), self.dtype)
        return x * scale + bias


# =============================================================================
# ConvNeXt 3D Block with SSM
# =============================================================================

class ConvNeXt3DSSMBlock(nn.Module):
    """ConvNeXt block adapted for 3D with ConvSSM.

    Architecture:
    1. 3D FFT depthwise conv (7×7×1 - spatial only)
    2. 3D Parallel ConvSSM (handles temporal processing)
    3. LayerNorm -> MLP -> LayerScale -> Residual
    """
    dim: int
    kernel_size: int = 7  # Spatial (H, W) for both dwconv and SSM
    ssm_kernel_size: int = 3  # SSM spatial kernel size
    ssm_kernel_size_t: int = 3  # SSM temporal kernel size
    expansion_ratio: int = 4
    layer_scale_init: float = 1e-6
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        B, T, H, W, C = x.shape

        # 1. 3D FFT-based depthwise conv (7×7×1 - spatial only, no temporal mixing)
        x = FFTDepthwiseConv3D(
            features=C,
            kernel_size=self.kernel_size,  # 7 spatial
            kernel_size_t=1,  # No temporal mixing in depthwise conv
            dtype=self.dtype,
            name='dwconv3d'
        )(x)

        # 2. 3D Parallel ConvSSM (handles temporal processing with ssm_kernel_size)
        x = ParallelConvSSM3D(
            dim=C,
            kernel_size=self.ssm_kernel_size,
            kernel_size_t=self.ssm_kernel_size_t,
            dtype=self.dtype,
            name='convssm3d'
        )(x)

        # 3. LayerNorm
        x = LayerNorm3D(dtype=self.dtype)(x)

        # 4. Pointwise MLP (applied per-position)
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
            mask = jax.random.bernoulli(rng, keep_prob, (B, 1, 1, 1, 1))
            x = x / keep_prob * mask

        return x + residual


# =============================================================================
# Downsampling for 3D (spatial only, preserve T)
# =============================================================================

class Downsample3D(nn.Module):
    """Spatial downsampling (H, W only, preserves T)."""
    out_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape

        # LayerNorm
        x = LayerNorm3D(dtype=self.dtype)(x)

        # Reshape to apply 2D conv per time step
        # (B, T, H, W, C) -> (B*T, H, W, C)
        x = x.reshape(B * T, H, W, C)

        # 2D strided conv for spatial downsampling
        x = nn.Conv(
            self.out_dim,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='VALID',
            dtype=self.dtype
        )(x)

        # Reshape back: (B*T, H/2, W/2, C') -> (B, T, H/2, W/2, C')
        _, H_new, W_new, C_new = x.shape
        x = x.reshape(B, T, H_new, W_new, C_new)

        return x


# =============================================================================
# Full 3D ConvNeXt-SSM Model
# =============================================================================

class ConvNeXt3DSSM(nn.Module):
    """ConvNeXt with 3D ConvSSM for image classification.

    Input image is repeated T times to create pseudo-video (B, T, H, W, 3),
    then processed through 3D ConvSSM blocks.

    Architecture per block:
    - FFTDepthwiseConv3D: 7×7×1 (spatial only, no temporal mixing)
    - ParallelConvSSM3D: ssm_kernel_size × ssm_kernel_size × ssm_kernel_size_t

    Attributes:
        num_classes: Number of output classes
        T: Number of times to repeat input (temporal depth)
        depths: Number of blocks per stage
        dims: Channel dimensions per stage
        kernel_size: Spatial kernel size for depthwise conv (always 7)
        ssm_kernel_size: Spatial kernel size for SSM (H, W)
        ssm_kernel_size_t: Temporal kernel size for SSM (T)
        drop_path_rate: Stochastic depth rate
        dtype: Compute dtype
    """
    num_classes: int = 1000
    T: int = 8  # Temporal depth (input repetition)
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    kernel_size: int = 7  # Spatial (H, W) for depthwise conv
    ssm_kernel_size: int = 3  # Spatial kernel for SSM
    ssm_kernel_size_t: int = 3  # Temporal kernel for SSM
    drop_path_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, H, W, 3) input image

        Returns:
            (B, num_classes) logits
        """
        B, H, W, C_in = x.shape

        # 1. Repeat input T times: (B, H, W, 3) -> (B, T, H, W, 3)
        x = jnp.tile(x[:, None, ...], (1, self.T, 1, 1, 1))
        # Now x is (B, T, H, W, 3)

        # 2. 3D Stem: (B, T, H, W, 3) -> (B, T, H/4, W/4, dims[0])
        # Apply 2D conv per time step for stem
        x = x.reshape(B * self.T, H, W, C_in)
        x = nn.Conv(
            self.dims[0],
            kernel_size=(4, 4),
            strides=(4, 4),
            padding='VALID',
            dtype=self.dtype,
            name='stem'
        )(x)
        _, H_stem, W_stem, _ = x.shape
        x = x.reshape(B, self.T, H_stem, W_stem, self.dims[0])

        # Stem norm
        x = LayerNorm3D(dtype=self.dtype, name='stem_norm')(x)

        # 3. Stochastic depth schedule
        total_blocks = sum(self.depths)
        dpr = [self.drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0

        # 4. Four stages
        for stage_idx, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            # Downsample between stages (except first)
            if stage_idx > 0:
                x = Downsample3D(dim, dtype=self.dtype, name=f'downsample_{stage_idx}')(x)

            # Blocks in this stage
            for block_i in range(depth):
                x = ConvNeXt3DSSMBlock(
                    dim=dim,
                    kernel_size=self.kernel_size,  # 7 for depthwise conv
                    ssm_kernel_size=self.ssm_kernel_size,  # SSM spatial
                    ssm_kernel_size_t=self.ssm_kernel_size_t,  # SSM temporal
                    drop_path_rate=dpr[block_idx],
                    dtype=self.dtype,
                    name=f'stage_{stage_idx}_block_{block_i}'
                )(x, train=train)
                block_idx += 1

        # 5. Global pooling over (T, H, W)
        x = jnp.mean(x, axis=(1, 2, 3))  # (B, C)

        # 6. Head
        x = LayerNorm3D(dtype=self.dtype, name='head_norm')(x[..., None, None, None, :])
        x = x.squeeze(axis=(1, 2, 3))  # Back to (B, C)
        x = nn.Dense(self.num_classes, dtype=self.dtype, name='head')(x)

        return x


# =============================================================================
# Model Constructors
# =============================================================================

def convnext_3d_ssm_tiny(
    num_classes: int = 1000,
    T: int = 8,
    kernel_size: int = 7,  # Depthwise conv spatial (7×7×1)
    ssm_kernel_size: int = 3,  # SSM spatial kernel
    ssm_kernel_size_t: int = 3,  # SSM temporal kernel
    **kwargs
) -> ConvNeXt3DSSM:
    """ConvNeXt-3D-SSM-Tiny.

    Architecture per block:
    - FFTDepthwiseConv3D: kernel_size × kernel_size × 1 (spatial only)
    - ParallelConvSSM3D: ssm_kernel_size × ssm_kernel_size × ssm_kernel_size_t
    """
    return ConvNeXt3DSSM(
        num_classes=num_classes,
        T=T,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        kernel_size=kernel_size,
        ssm_kernel_size=ssm_kernel_size,
        ssm_kernel_size_t=ssm_kernel_size_t,
        **kwargs
    )


def convnext_3d_ssm_small(
    num_classes: int = 1000,
    T: int = 8,
    kernel_size: int = 7,  # Depthwise conv spatial (7×7×1)
    ssm_kernel_size: int = 3,  # SSM spatial kernel
    ssm_kernel_size_t: int = 3,  # SSM temporal kernel
    **kwargs
) -> ConvNeXt3DSSM:
    """ConvNeXt-3D-SSM-Small.

    Architecture per block:
    - FFTDepthwiseConv3D: kernel_size × kernel_size × 1 (spatial only)
    - ParallelConvSSM3D: ssm_kernel_size × ssm_kernel_size × ssm_kernel_size_t
    """
    return ConvNeXt3DSSM(
        num_classes=num_classes,
        T=T,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        kernel_size=kernel_size,
        ssm_kernel_size=ssm_kernel_size,
        ssm_kernel_size_t=ssm_kernel_size_t,
        **kwargs
    )


# =============================================================================
# Tests
# =============================================================================

if __name__ == '__main__':
    import jax.random as random
    import time

    print("=" * 70)
    print("TEST: ConvNeXt-3D-SSM (3D ConvSSM with anisotropic 7x7x5 kernels)")
    print("=" * 70)

    key = random.PRNGKey(0)

    # Test with T=4 (smaller for testing), anisotropic kernels 7x7x5
    print("\nT=4, kernel_size=7 (spatial), kernel_size_t=5 (temporal):")
    model = convnext_3d_ssm_tiny(num_classes=10, T=4, kernel_size=7, kernel_size_t=5)

    # Smaller input for testing
    dummy = jnp.ones((2, 224, 224, 3))

    print("Initializing model...")
    t0 = time.time()
    variables = model.init({'params': key, 'dropout': key}, dummy, train=False)
    params = variables['params']
    init_time = time.time() - t0
    print(f"Init time: {init_time:.2f}s")

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # JIT compile
    print("\nJIT compiling...")
    t0 = time.time()

    @jax.jit
    def forward(params, x):
        return model.apply({'params': params}, x, train=False, rngs={'dropout': key})

    logits = forward(params, dummy)
    logits.block_until_ready()
    compile_time = time.time() - t0
    print(f"JIT compile time: {compile_time:.2f}s")
    print(f"Output shape: {logits.shape}")

    # Runtime benchmark
    print("\nRuntime benchmark (5 iterations)...")
    times = []
    for i in range(5):
        t0 = time.time()
        logits = forward(params, dummy)
        logits.block_until_ready()
        times.append(time.time() - t0)

    avg_time = np.mean(times[2:])  # Skip first 2 warmup
    print(f"Average forward time: {avg_time*1000:.2f}ms")

    # Gradient check
    print("\nGradient check...")
    def model_loss(params, x):
        logits = model.apply({'params': params}, x, train=True, rngs={'dropout': key})
        return jnp.mean(logits ** 2)

    loss, grads = jax.value_and_grad(model_loss)(params, dummy)
    total_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))

    print(f"Model loss: {loss:.6f}")
    print(f"Total gradient norm: {total_grad_norm:.6f}")

    # Test with T=8
    print("\n" + "-" * 70)
    print("T=8, kernel_size=7 (spatial), kernel_size_t=5 (temporal):")
    model_t8 = convnext_3d_ssm_tiny(num_classes=10, T=8, kernel_size=7, kernel_size_t=5)
    variables_t8 = model_t8.init({'params': key, 'dropout': key}, dummy, train=False)
    params_t8 = variables_t8['params']
    n_params_t8 = sum(x.size for x in jax.tree_util.tree_leaves(params_t8))
    print(f"Parameters: {n_params_t8:,} ({n_params_t8 / 1e6:.1f}M)")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
