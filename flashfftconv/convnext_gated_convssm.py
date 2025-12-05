"""ConvNeXt with Gated ConvSSM (minGRU-style).

This module implements a gated SSM using the minGRU formulation:
    h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t

Where:
    z_t = sigmoid(gate) is the update gate
    h̃_t = candidate hidden state (input contribution)

We use log-space computation for numerical stability (Heinsen's trick):
    log_coeffs = -softplus(gate) = log(1 - sigmoid(gate))
    log_z = -softplus(-gate) = log(sigmoid(gate))

This allows us to use associative scan in log-space for O(log T) parallel computation.

Reference: https://github.com/lucidrains/minGRU-pytorch
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from typing import Sequence, Tuple
import numpy as np


# =============================================================================
# Log-space Operations for Numerical Stability
# =============================================================================

def log1mexp(x: jnp.ndarray) -> jnp.ndarray:
    """Compute log(1 - exp(x)) in a numerically stable way.

    For x < -0.693 (log(0.5)): log(1 - exp(x))
    For x >= -0.693: log(-expm1(x))

    Args:
        x: Input tensor (should be negative)

    Returns:
        log(1 - exp(x))
    """
    # Stable computation following numpy/scipy conventions
    return jnp.where(
        x < -0.693,
        jnp.log(-jnp.expm1(x)),  # For small exp(x)
        jnp.log1p(-jnp.exp(x))   # For larger exp(x)
    )


def logaddexp_pair(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Stable log(exp(a) + exp(b))."""
    return jnp.logaddexp(a, b)


def heinsen_associative_scan_log(log_coeffs: jnp.ndarray, log_values: jnp.ndarray) -> jnp.ndarray:
    """Heinsen's associative scan in log-space.

    Computes the parallel prefix sum for:
        h_t = c_t * h_{t-1} + v_t

    Where c_t = coefficients (1 - z_t for minGRU), v_t = z_t * h̃_t

    In log-space:
        log(h_t) = logaddexp(log(c_t) + log(h_{t-1}), log(v_t))

    Args:
        log_coeffs: (T, ...) log of coefficients (log(1 - z))
        log_values: (T, ...) log of values (log(z) + log(h̃))

    Returns:
        (T, ...) log of output states
    """
    # Associative operation in log-space
    # (log_a1, log_b1) ⊕ (log_a2, log_b2) = (log_a1 + log_a2, logaddexp(log_a2 + log_b1, log_b2))
    def associative_op(left, right):
        log_a_left, log_b_left = left
        log_a_right, log_b_right = right

        # New coefficient: a1 * a2 -> log(a1 * a2) = log_a1 + log_a2
        new_log_a = log_a_left + log_a_right

        # New value: a2 * b1 + b2 -> logaddexp(log_a2 + log_b1, log_b2)
        new_log_b = jnp.logaddexp(log_a_right + log_b_left, log_b_right)

        return (new_log_a, new_log_b)

    # Run associative scan
    _, log_h = lax.associative_scan(
        associative_op,
        (log_coeffs, log_values),
        axis=0
    )

    return log_h


# =============================================================================
# Helper Functions for 3D FFT Operations (same as pure SSM)
# =============================================================================

def kernel_to_freq_3d(kernel: jnp.ndarray, T: int, H: int, W: int) -> jnp.ndarray:
    """Convert small 3D spatial kernel to frequency domain.

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

    t_idx = jnp.arange(k_t)
    h_idx = jnp.arange(k_h)
    w_idx = jnp.arange(k_w)

    target_t = (t_idx - center_t) % T
    target_h = (h_idx - center_h) % H
    target_w = (w_idx - center_w) % W

    tt, th, tw = jnp.meshgrid(target_t, target_h, target_w, indexing='ij')

    padded = jnp.zeros((C, T, H, W), dtype=kernel.dtype)
    padded = padded.at[:, tt, th, tw].set(kernel)

    return jnp.fft.fftn(padded, axes=(1, 2, 3))


# =============================================================================
# Gated ConvSSM (minGRU-style)
# =============================================================================

class GatedConvSSM3D(nn.Module):
    """Gated 3D ConvSSM using minGRU-style formulation.

    Uses log-space computation for numerical stability:
        h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t

    Where:
        z_t = sigmoid(W_z * x_t) is the update gate
        h̃_t = W_h * x_t is the candidate hidden state

    In log-space with associative scan for O(log T) parallel computation.

    Attributes:
        dim: Number of channels
        kernel_size: Spatial kernel size (H, W dimensions)
        kernel_size_t: Temporal kernel size (T dimension)
        expansion_factor: Expansion factor for hidden state (like minGRU)
        dtype: Compute dtype
    """
    dim: int
    kernel_size: int = 7
    kernel_size_t: int = 5
    expansion_factor: float = 1.5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run gated 3D ConvSSM.

        Args:
            x: (B, T, H, W, C) 3D spatial input

        Returns:
            (B, T, H, W, C) output after gated SSM
        """
        B, T, H, W, C = x.shape
        k = self.kernel_size
        k_t = self.kernel_size_t
        hidden_dim = int(C * self.expansion_factor)

        # === Gate and Hidden projections ===
        # Project input to get gate and hidden candidate
        # We do this per-position first, then apply spatial convolution

        # Gate projection: x -> z (gate values)
        gate_proj = nn.Dense(C, dtype=self.dtype, name='gate_proj')(x)

        # Hidden projection: x -> h̃ (candidate hidden)
        hidden_proj = nn.Dense(C, dtype=self.dtype, name='hidden_proj')(x)

        # === Spatial mixing with 3D convolution (in frequency domain) ===
        # Learnable 3D kernel for spatial mixing of the gate
        gate_kernel = self.param(
            'gate_kernel',
            nn.initializers.lecun_normal(),
            (C, k_t, k, k),
            self.dtype
        )

        # Learnable 3D kernel for spatial mixing of hidden candidate
        hidden_kernel = self.param(
            'hidden_kernel',
            nn.initializers.lecun_normal(),
            (C, k_t, k, k),
            self.dtype
        )

        # Convert kernels to frequency domain
        gate_kernel_f = kernel_to_freq_3d(gate_kernel, T, H, W)  # (C, T, H, W)
        hidden_kernel_f = kernel_to_freq_3d(hidden_kernel, T, H, W)  # (C, T, H, W)

        # Reshape: (C, T, H, W) -> (T, H, W, C)
        gate_kernel_f = gate_kernel_f.transpose(1, 2, 3, 0)
        hidden_kernel_f = hidden_kernel_f.transpose(1, 2, 3, 0)

        # FFT of projections
        gate_f = jnp.fft.fftn(gate_proj, axes=(1, 2, 3))  # (B, T, H, W, C)
        hidden_f = jnp.fft.fftn(hidden_proj, axes=(1, 2, 3))  # (B, T, H, W, C)

        # Apply convolution in frequency domain
        gate_conv_f = gate_f * gate_kernel_f[None, ...]  # (B, T, H, W, C)
        hidden_conv_f = hidden_f * hidden_kernel_f[None, ...]  # (B, T, H, W, C)

        # IFFT back to spatial domain
        gate_spatial = jnp.fft.ifftn(gate_conv_f, axes=(1, 2, 3)).real  # (B, T, H, W, C)
        hidden_spatial = jnp.fft.ifftn(hidden_conv_f, axes=(1, 2, 3)).real  # (B, T, H, W, C)

        # === minGRU gating in log-space ===
        # z = sigmoid(gate) -> update gate
        # 1 - z = sigmoid(-gate) -> forget coefficient

        # Log-space coefficients for numerical stability:
        # log(1 - z) = -softplus(gate) = log(sigmoid(-gate))
        # log(z) = -softplus(-gate) = log(sigmoid(gate))
        log_forget = -jax.nn.softplus(gate_spatial)  # log(1 - z)
        log_update = -jax.nn.softplus(-gate_spatial)  # log(z)

        # For the hidden candidate, we need it to be positive for log-space
        # Following minGRU: h̃ = exp(hidden) for the parallel scan version
        # Or: h̃ = relu(hidden) + epsilon, then take log
        # Using exp ensures positivity but may have range issues
        # Alternative: squared activation (always positive)

        # Use squared activation for positivity (like minGRU's g(x) = x^2)
        hidden_positive = hidden_spatial ** 2 + 1e-6
        log_hidden = jnp.log(hidden_positive)

        # Combined log-value: log(z * h̃) = log(z) + log(h̃)
        log_values = log_update + log_hidden  # (B, T, H, W, C)

        # Reshape for scan: (B, T, H, W, C) -> (T, B, H, W, C)
        log_forget_seq = log_forget.transpose(1, 0, 2, 3, 4)
        log_values_seq = log_values.transpose(1, 0, 2, 3, 4)

        # Run Heinsen's associative scan in log-space
        log_h_seq = heinsen_associative_scan_log(log_forget_seq, log_values_seq)

        # Reshape back: (T, B, H, W, C) -> (B, T, H, W, C)
        log_h = log_h_seq.transpose(1, 0, 2, 3, 4)

        # Convert from log-space back to regular space
        h = jnp.exp(log_h)

        # Final output projection
        output = nn.Dense(C, dtype=self.dtype, name='output_proj')(h)

        return output.astype(self.dtype)


# =============================================================================
# LayerNorm for 3D
# =============================================================================

class LayerNorm3D(nn.Module):
    """Layer Normalization for 3D inputs."""
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
# Gated ConvNeXt Block
# =============================================================================

class GatedConvNeXt3DBlock(nn.Module):
    """ConvNeXt block with Gated ConvSSM (minGRU-style).

    Architecture:
    1. GatedConvSSM3D (handles spatial/temporal mixing)
    2. LayerNorm -> MLP -> LayerScale -> Residual
    """
    dim: int
    kernel_size: int = 7
    kernel_size_t: int = 5
    expansion_ratio: int = 4
    layer_scale_init: float = 1e-6
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        B, T, H, W, C = x.shape

        # 1. Gated ConvSSM (minGRU-style)
        x = GatedConvSSM3D(
            dim=C,
            kernel_size=self.kernel_size,
            kernel_size_t=self.kernel_size_t,
            dtype=self.dtype,
            name='gated_convssm3d'
        )(x)

        # 2. LayerNorm
        x = LayerNorm3D(dtype=self.dtype)(x)

        # 3. Pointwise MLP
        hidden_dim = int(C * self.expansion_ratio)
        x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(C, dtype=self.dtype)(x)

        # 4. Layer scale
        gamma = self.param(
            'layer_scale',
            nn.initializers.constant(self.layer_scale_init),
            (C,),
            self.dtype
        )
        x = x * gamma

        # 5. Stochastic depth
        if train and self.drop_path_rate > 0:
            keep_prob = 1.0 - self.drop_path_rate
            rng = self.make_rng('dropout')
            mask = jax.random.bernoulli(rng, keep_prob, (B, 1, 1, 1, 1))
            x = x / keep_prob * mask

        return x + residual


# =============================================================================
# Downsampling for 3D
# =============================================================================

class Downsample3D(nn.Module):
    """Spatial downsampling (H, W only, preserves T)."""
    out_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, H, W, C = x.shape
        x = LayerNorm3D(dtype=self.dtype)(x)
        x = x.reshape(B * T, H, W, C)
        x = nn.Conv(
            self.out_dim,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='VALID',
            dtype=self.dtype
        )(x)
        _, H_new, W_new, C_new = x.shape
        x = x.reshape(B, T, H_new, W_new, C_new)
        return x


# =============================================================================
# Full Gated ConvNeXt-SSM Model
# =============================================================================

class GatedConvNeXt3DSSM(nn.Module):
    """Gated ConvNeXt with 3D ConvSSM (minGRU-style).

    Uses gated SSM with log-space computation for numerical stability.
    Input image is repeated T times to create pseudo-video.

    Attributes:
        num_classes: Number of output classes
        T: Number of times to repeat input (temporal depth)
        depths: Number of blocks per stage
        dims: Channel dimensions per stage
        kernel_size: Spatial kernel size for ConvSSM
        kernel_size_t: Temporal kernel size for ConvSSM
        drop_path_rate: Stochastic depth rate
        dtype: Compute dtype
    """
    num_classes: int = 1000
    T: int = 8
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    kernel_size: int = 7
    kernel_size_t: int = 5
    drop_path_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        B, H, W, C_in = x.shape

        # 1. Repeat input T times
        x = jnp.tile(x[:, None, ...], (1, self.T, 1, 1, 1))

        # 2. 3D Stem
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
        x = LayerNorm3D(dtype=self.dtype, name='stem_norm')(x)

        # 3. Stochastic depth schedule
        total_blocks = sum(self.depths)
        dpr = [self.drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0

        # 4. Four stages
        for stage_idx, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            if stage_idx > 0:
                x = Downsample3D(dim, dtype=self.dtype, name=f'downsample_{stage_idx}')(x)

            for block_i in range(depth):
                x = GatedConvNeXt3DBlock(
                    dim=dim,
                    kernel_size=self.kernel_size,
                    kernel_size_t=self.kernel_size_t,
                    drop_path_rate=dpr[block_idx],
                    dtype=self.dtype,
                    name=f'stage_{stage_idx}_block_{block_i}'
                )(x, train=train)
                block_idx += 1

        # 5. Global pooling
        x = jnp.mean(x, axis=(1, 2, 3))

        # 6. Head
        x = LayerNorm3D(dtype=self.dtype, name='head_norm')(x[..., None, None, None, :])
        x = x.squeeze(axis=(1, 2, 3))
        x = nn.Dense(self.num_classes, dtype=self.dtype, name='head')(x)

        return x


# =============================================================================
# Model Constructors
# =============================================================================

def gated_convnext_3d_ssm_tiny(
    num_classes: int = 1000,
    T: int = 8,
    kernel_size: int = 7,
    kernel_size_t: int = 5,
    **kwargs
) -> GatedConvNeXt3DSSM:
    """Gated ConvNeXt-3D-SSM-Tiny with minGRU-style gating."""
    return GatedConvNeXt3DSSM(
        num_classes=num_classes,
        T=T,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        kernel_size=kernel_size,
        kernel_size_t=kernel_size_t,
        **kwargs
    )


def gated_convnext_3d_ssm_small(
    num_classes: int = 1000,
    T: int = 8,
    kernel_size: int = 7,
    kernel_size_t: int = 5,
    **kwargs
) -> GatedConvNeXt3DSSM:
    """Gated ConvNeXt-3D-SSM-Small with minGRU-style gating."""
    return GatedConvNeXt3DSSM(
        num_classes=num_classes,
        T=T,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        kernel_size=kernel_size,
        kernel_size_t=kernel_size_t,
        **kwargs
    )


# =============================================================================
# Tests
# =============================================================================

if __name__ == '__main__':
    import jax.random as random
    import time

    print("=" * 70)
    print("TEST: Gated ConvNeXt-3D-SSM (minGRU-style)")
    print("=" * 70)

    key = random.PRNGKey(0)

    # Test with T=4
    print("\nT=4, kernel_size=7 (spatial), kernel_size_t=5 (temporal):")
    model = gated_convnext_3d_ssm_tiny(num_classes=10, T=4, kernel_size=7, kernel_size_t=5)

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

    # Check for NaN/Inf
    print(f"Output contains NaN: {jnp.any(jnp.isnan(logits))}")
    print(f"Output contains Inf: {jnp.any(jnp.isinf(logits))}")
    print(f"Output range: [{float(jnp.min(logits)):.4f}, {float(jnp.max(logits)):.4f}]")

    # Runtime benchmark
    print("\nRuntime benchmark (5 iterations)...")
    times = []
    for i in range(5):
        t0 = time.time()
        logits = forward(params, dummy)
        logits.block_until_ready()
        times.append(time.time() - t0)

    avg_time = np.mean(times[2:])
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
    print(f"Gradient contains NaN: {any(jnp.any(jnp.isnan(g)) for g in jax.tree_util.tree_leaves(grads))}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
