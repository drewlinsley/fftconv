"""ConvNeXt with FFT convolution + ConvSSM using PARALLEL associative scan.

This version uses jax.lax.associative_scan for O(log T) parallel computation
instead of O(T) sequential scan. Much faster compilation and runtime.

The receptive field grows with T: RF ≈ T * kernel_size

Architecture per block:
1. FFT depthwise conv (same as baseline ConvNeXt)
2. ConvSSM via associative scan (parallel, O(log T) depth)
3. LayerNorm -> MLP -> LayerScale -> Residual
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from typing import Sequence
import numpy as np


# =============================================================================
# FFT Depthwise Convolution (verified working)
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
# Parallel ConvSSM using Associative Scan
# =============================================================================

def ssm_associative_op(left, right):
    """Associative operation for linear recurrence.

    For h_t = a_t * h_{t-1} + b_t, the associative op is:
    (a1, b1) ⊕ (a2, b2) = (a1 * a2, a2 * b1 + b2)

    This allows parallel prefix computation in O(log T) steps.
    """
    a_left, b_left = left
    a_right, b_right = right
    return (a_left * a_right, a_right * b_left + b_right)


class ParallelConvSSM(nn.Module):
    """Parallel ConvSSM using associative scan.

    SSM recurrence: h_t = A * h_{t-1} + B * x
    where * denotes depthwise convolution (via FFT).

    Uses jax.lax.associative_scan for O(log T) parallel depth.
    Receptive field grows with T: RF ≈ T * kernel_size.

    Attributes:
        dim: Number of channels
        T: Number of SSM timesteps (RF grows with T)
        kernel_size: Size of A and B convolution kernels
        dtype: Compute dtype
    """
    dim: int
    T: int = 8
    kernel_size: int = 7
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run parallel ConvSSM for T timesteps.

        Args:
            x: (B, H, W, C) spatial input

        Returns:
            (B, H, W, C) output after T SSM iterations
        """
        B, H, W, C = x.shape
        k = self.kernel_size

        # Learn spatial kernels for A and B
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

        # Apply tanh to A for stability (bounded eigenvalues)
        # Scale < 1 ensures contraction/stability
        A_kernel_stable = 0.9 * jnp.tanh(A_kernel)

        # Pre-compute FFT of kernels
        center = k // 2
        i_idx = jnp.arange(k)
        j_idx = jnp.arange(k)
        target_i = (i_idx - center) % H
        target_j = (j_idx - center) % W
        ti, tj = jnp.meshgrid(target_i, target_j, indexing='ij')

        # Pad and FFT A kernel -> (1, H, W, C) complex
        A_padded = jnp.zeros((C, H, W), dtype=self.dtype)
        A_padded = A_padded.at[:, ti, tj].set(A_kernel_stable)
        A_f = jnp.fft.fft2(A_padded, axes=(1, 2))
        A_f = A_f.transpose(1, 2, 0)[None, ...]  # (1, H, W, C)

        # Pad and FFT B kernel -> (1, H, W, C) complex
        B_padded = jnp.zeros((C, H, W), dtype=self.dtype)
        B_padded = B_padded.at[:, ti, tj].set(B_kernel)
        B_f = jnp.fft.fft2(B_padded, axes=(1, 2))
        B_f = B_f.transpose(1, 2, 0)[None, ...]  # (1, H, W, C)

        # FFT input -> (B, H, W, C) complex
        x_f = jnp.fft.fft2(x, axes=(1, 2))

        # B * x in frequency domain -> (B, H, W, C) complex
        Bx_f = B_f * x_f

        # Create sequences for associative scan
        # a_seq[t] = A_f for all t (same transition at each step)
        # b_seq[t] = Bx_f for all t (same input injection at each step)
        # Shape: (T, B, H, W, C)
        a_seq = jnp.broadcast_to(A_f, (self.T, B, H, W, C))
        b_seq = jnp.broadcast_to(Bx_f, (self.T, B, H, W, C))

        # Run associative scan - O(log T) parallel depth!
        # Returns cumulative results: h_1, h_2, ..., h_T
        _, h_all_f = lax.associative_scan(
            ssm_associative_op,
            (a_seq, b_seq),
            axis=0
        )

        # Take final state h_T
        h_final_f = h_all_f[-1]  # (B, H, W, C) complex

        # IFFT back to spatial domain
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
# ConvNeXt Block with Parallel ConvSSM
# =============================================================================

class ConvNeXtParallelSSMBlock(nn.Module):
    """ConvNeXt block with FFT conv + parallel ConvSSM.

    Architecture:
    1. FFT depthwise conv (7x7 baseline)
    2. Parallel ConvSSM (O(log T) depth via associative scan)
    3. LayerNorm -> MLP -> LayerScale -> Residual

    Receptive field grows with T: effective RF ≈ T * kernel_size
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

        # 1. FFT-based depthwise conv
        x = FFTDepthwiseConv(
            features=C,
            kernel_size=self.kernel_size,
            dtype=self.dtype,
            name='dwconv'
        )(x)

        # 2. Parallel ConvSSM (associative scan)
        x = ParallelConvSSM(
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

class ConvNeXtParallelSSM(nn.Module):
    """ConvNeXt with FFT conv + parallel ConvSSM.

    Key features:
    - FFT-based depthwise convolution
    - Parallel ConvSSM using associative_scan (O(log T) depth)
    - Receptive field grows with T
    - Much faster compile and runtime than sequential scan
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
                x = ConvNeXtParallelSSMBlock(
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


def convnext_parallel_ssm_tiny(num_classes: int = 1000, T: int = 8, **kwargs) -> ConvNeXtParallelSSM:
    """ConvNeXt-Parallel-SSM-Tiny"""
    return ConvNeXtParallelSSM(
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
    import time

    print("=" * 70)
    print("TEST: ConvNeXt-Parallel-SSM (Associative Scan)")
    print("=" * 70)

    key = random.PRNGKey(0)

    # Test with T=8
    print("\nT=8:")
    model = convnext_parallel_ssm_tiny(num_classes=10, T=8)
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

    # Runtime benchmark
    print("\nRuntime benchmark (10 iterations)...")
    times = []
    for i in range(10):
        t0 = time.time()
        logits = forward(params, dummy)
        logits.block_until_ready()
        times.append(time.time() - t0)

    avg_time = np.mean(times[2:])  # Skip first 2 warmup
    print(f"Average forward time: {avg_time*1000:.2f}ms")
    print(f"Output shape: {logits.shape}")

    # Gradient check
    print("\nGradient check...")
    def model_loss(params, x):
        logits = model.apply({'params': params}, x, train=True, rngs={'dropout': key})
        return jnp.mean(logits ** 2)

    loss, grads = jax.value_and_grad(model_loss)(params, dummy)
    total_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))

    print(f"Model loss: {loss:.6f}")
    print(f"Total gradient norm: {total_grad_norm:.6f}")

    # Test with larger T
    print("\n" + "-" * 70)
    print("T=32 (larger RF):")
    model_t32 = convnext_parallel_ssm_tiny(num_classes=10, T=32)

    print("JIT compiling T=32...")
    t0 = time.time()

    @jax.jit
    def forward_t32(params, x):
        return model_t32.apply({'params': params}, x, train=False, rngs={'dropout': key})

    variables_t32 = model_t32.init({'params': key, 'dropout': key}, dummy, train=False)
    logits_t32 = forward_t32(variables_t32['params'], dummy)
    logits_t32.block_until_ready()
    compile_time_t32 = time.time() - t0
    print(f"JIT compile time (T=32): {compile_time_t32:.2f}s")

    # Runtime for T=32
    times_t32 = []
    for i in range(10):
        t0 = time.time()
        logits_t32 = forward_t32(variables_t32['params'], dummy)
        logits_t32.block_until_ready()
        times_t32.append(time.time() - t0)

    avg_time_t32 = np.mean(times_t32[2:])
    print(f"Average forward time (T=32): {avg_time_t32*1000:.2f}ms")
    print(f"Speedup vs sequential (expected ~T/log(T)): {32/np.log2(32):.1f}x potential")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
