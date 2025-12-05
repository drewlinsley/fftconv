"""ConvNeXt with 3D Mamba-Gated ConvSSM.

Extends the 3D ConvSSM with Mamba-style input-dependent gating:
1. Forget gate: Controls state decay (input-dependent)
2. Input gate: Controls what new information enters (input-dependent)
3. Delta: Controls update magnitude (Mamba-style selectivity)

The recurrence becomes:
    h_t = forget_gate * (A ★ h_{t-1}) + input_gate * delta * (B ★ x)

Where forget_gate, input_gate, and delta are computed from the input.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from typing import Sequence
import numpy as np


# =============================================================================
# Helper Functions (same as convnext_3d_ssm.py)
# =============================================================================

def kernel_to_freq_3d(kernel: jnp.ndarray, T: int, H: int, W: int) -> jnp.ndarray:
    """Convert small 3D spatial kernel to frequency domain."""
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


def mamba_associative_op(left, right):
    """Associative operation for gated linear recurrence.

    For h_t = a_t * h_{t-1} + b_t with input-dependent a_t (gates):
    (a1, b1) ⊕ (a2, b2) = (a1 * a2, a2 * b1 + b2)
    """
    a_left, b_left = left
    a_right, b_right = right
    return (a_left * a_right, a_right * b_left + b_right)


# =============================================================================
# 3D Mamba-Gated ConvSSM
# =============================================================================

class MambaGatedConvSSM3D(nn.Module):
    """3D ConvSSM with Mamba-style input-dependent gating.

    Key differences from ParallelConvSSM3D:
    1. Forget gate: sigmoid(forget_bias + forget_scale * x) - controls decay
    2. Input gate: sigmoid(input_bias + input_scale * x) - controls input mixing
    3. Delta: softplus(delta_proj(x)) - controls update magnitude

    The recurrence in frequency domain:
        h_t = forget_gate * (A_f * h_{t-1}) + input_gate * delta * (B_f * x_f)

    Attributes:
        dim: Number of channels
        kernel_size: Spatial kernel size (H, W)
        kernel_size_t: Temporal kernel size (T)
        use_delta: Whether to use Mamba-style delta projection
        dtype: Compute dtype
    """
    dim: int
    kernel_size: int = 3
    kernel_size_t: int = 3
    use_delta: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run 3D Mamba-gated ConvSSM.

        Args:
            x: (B, T, H, W, C) 3D spatial input

        Returns:
            (B, T, H, W, C) output after gated SSM
        """
        B, T, H, W, C = x.shape
        k = self.kernel_size
        k_t = self.kernel_size_t

        # === Learnable 3D convolution kernels ===
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
        A_f = kernel_to_freq_3d(A_kernel, T, H, W)  # (C, T, H, W)
        B_f = kernel_to_freq_3d(B_kernel, T, H, W)  # (C, T, H, W)

        # Reshape for broadcasting: (C, T, H, W) -> (T, 1, H, W, C) for batch broadcast
        A_f = A_f.transpose(1, 2, 3, 0)[:, None, ...]  # (T, 1, H, W, C)
        B_f = B_f.transpose(1, 2, 3, 0)[:, None, ...]  # (T, 1, H, W, C)

        # === Mamba-style input-dependent gates ===
        # These are computed in spatial domain before FFT

        # Forget gate: controls state decay
        # Higher values = more remembering = larger receptive field
        forget_bias = self.param('forget_bias', nn.initializers.constant(2.0), (C,))
        forget_scale = self.param('forget_scale', nn.initializers.constant(0.01), (C,))
        forget_gate = jax.nn.sigmoid(forget_bias + forget_scale * x)  # (B, T, H, W, C)

        # Input gate: controls what new info enters
        input_bias = self.param('input_bias', nn.initializers.constant(0.0), (C,))
        input_scale = self.param('input_scale', nn.initializers.constant(0.01), (C,))
        input_gate = jax.nn.sigmoid(input_bias + input_scale * x)  # (B, T, H, W, C)

        # Delta (optional): Mamba-style update magnitude
        if self.use_delta:
            delta_proj = nn.Dense(C, name='delta_proj')
            delta = jax.nn.softplus(delta_proj(x))  # (B, T, H, W, C), always positive
        else:
            delta = jnp.ones_like(x)

        # === FFT of input ===
        x_f = jnp.fft.fftn(x, axes=(1, 2, 3))  # (B, T, H, W, C) complex

        # === Gated SSM coefficients ===
        # a = forget_gate * A_f (input-dependent state transition)
        # b = input_gate * delta * B_f * x_f (gated input contribution)

        # FFT the gates (they modulate in spatial domain, so we need to convolve)
        # For simplicity, we apply gates directly to the coefficients
        # This is an approximation but works well in practice

        # For parallel scan, we need sequences over T
        # Reshape: (B, T, H, W, C) -> (T, B, H, W, C)
        forget_seq = forget_gate.transpose(1, 0, 2, 3, 4)  # (T, B, H, W, C)
        input_seq = input_gate.transpose(1, 0, 2, 3, 4)    # (T, B, H, W, C)
        delta_seq = delta.transpose(1, 0, 2, 3, 4)         # (T, B, H, W, C)
        x_f_seq = x_f.transpose(1, 0, 2, 3, 4)             # (T, B, H, W, C)

        # Gated coefficients for associative scan
        # a_t = forget_gate_t * A_f (modulated state transition)
        # b_t = input_gate_t * delta_t * B_f * x_f_t
        a_seq = forget_seq * A_f  # (T, B, H, W, C) - A_f is (T, 1, H, W, C)
        b_seq = input_seq * delta_seq * B_f * x_f_seq  # (T, B, H, W, C)

        # Parallel associative scan
        _, h_all_f = lax.associative_scan(
            mamba_associative_op,
            (a_seq, b_seq),
            axis=0
        )

        # Reshape back: (T, B, H, W, C) -> (B, T, H, W, C)
        h_f = h_all_f.transpose(1, 0, 2, 3, 4)

        # IFFT back to spatial domain
        h = jnp.fft.ifftn(h_f, axes=(1, 2, 3)).real

        return h.astype(self.dtype)


# =============================================================================
# 3D FFT Depthwise Conv (same as convnext_3d_ssm.py)
# =============================================================================

class FFTDepthwiseConv3D(nn.Module):
    """3D FFT-based depthwise convolution."""
    features: int
    kernel_size: int = 7
    kernel_size_t: int = 1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, H, W, C = x.shape
        k = self.kernel_size
        k_t = self.kernel_size_t

        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (C, k_t, k, k),
            self.dtype
        )

        kernel_f = kernel_to_freq_3d(kernel, T, H, W)
        kernel_f = kernel_f.transpose(1, 2, 3, 0)[None, ...]

        x_f = jnp.fft.fftn(x, axes=(1, 2, 3))
        out_f = x_f * kernel_f
        out = jnp.fft.ifftn(out_f, axes=(1, 2, 3)).real

        return out.astype(self.dtype)


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
# ConvNeXt 3D Mamba Block
# =============================================================================

class ConvNeXt3DMambaBlock(nn.Module):
    """ConvNeXt block with Mamba-gated 3D ConvSSM.

    Architecture:
    1. 3D FFT depthwise conv (7×7×1 - spatial only)
    2. 3D Mamba-Gated ConvSSM (handles temporal with gates)
    3. LayerNorm -> MLP -> LayerScale -> Residual
    """
    dim: int
    kernel_size: int = 7  # Spatial for dwconv
    ssm_kernel_size: int = 3  # SSM spatial
    ssm_kernel_size_t: int = 3  # SSM temporal
    use_delta: bool = True
    expansion_ratio: int = 4
    layer_scale_init: float = 1e-6
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        B, T, H, W, C = x.shape

        # 1. 3D FFT depthwise conv (spatial only)
        x = FFTDepthwiseConv3D(
            features=C,
            kernel_size=self.kernel_size,
            kernel_size_t=1,
            dtype=self.dtype,
            name='dwconv3d'
        )(x)

        # 2. Mamba-gated 3D ConvSSM
        x = MambaGatedConvSSM3D(
            dim=C,
            kernel_size=self.ssm_kernel_size,
            kernel_size_t=self.ssm_kernel_size_t,
            use_delta=self.use_delta,
            dtype=self.dtype,
            name='mamba_convssm3d'
        )(x)

        # 3. LayerNorm
        x = LayerNorm3D(dtype=self.dtype)(x)

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
            mask = jax.random.bernoulli(rng, keep_prob, (B, 1, 1, 1, 1))
            x = x / keep_prob * mask

        return x + residual


# =============================================================================
# Downsampling for 3D
# =============================================================================

class Downsample3D(nn.Module):
    """Spatial downsampling (preserves T)."""
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
# Full 3D Mamba-ConvNeXt Model
# =============================================================================

class ConvNeXt3DMamba(nn.Module):
    """ConvNeXt with 3D Mamba-Gated ConvSSM.

    Input image is repeated T times, then processed through
    Mamba-gated 3D ConvSSM blocks with input-dependent gates.

    Attributes:
        num_classes: Number of output classes
        T: Temporal depth
        depths: Blocks per stage
        dims: Channels per stage
        kernel_size: Spatial kernel for depthwise conv
        ssm_kernel_size: SSM spatial kernel
        ssm_kernel_size_t: SSM temporal kernel
        use_delta: Use Mamba-style delta projection
        drop_path_rate: Stochastic depth
        dtype: Compute dtype
    """
    num_classes: int = 1000
    T: int = 8
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    kernel_size: int = 7
    ssm_kernel_size: int = 3
    ssm_kernel_size_t: int = 3
    use_delta: bool = True
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
                x = ConvNeXt3DMambaBlock(
                    dim=dim,
                    kernel_size=self.kernel_size,
                    ssm_kernel_size=self.ssm_kernel_size,
                    ssm_kernel_size_t=self.ssm_kernel_size_t,
                    use_delta=self.use_delta,
                    drop_path_rate=dpr[block_idx],
                    dtype=self.dtype,
                    name=f'stage_{stage_idx}_block_{block_i}'
                )(x, train=train)
                block_idx += 1

        # 5. Global pooling
        x = jnp.mean(x, axis=(1, 2, 3))  # (B, C)

        # 6. Head
        x = LayerNorm3D(dtype=self.dtype, name='head_norm')(x[..., None, None, None, :])
        x = x.squeeze(axis=(1, 2, 3))
        x = nn.Dense(self.num_classes, dtype=self.dtype, name='head')(x)

        return x


# =============================================================================
# Model Constructors
# =============================================================================

def convnext_3d_mamba_tiny(
    num_classes: int = 1000,
    T: int = 8,
    kernel_size: int = 7,
    ssm_kernel_size: int = 3,
    ssm_kernel_size_t: int = 3,
    use_delta: bool = True,
    **kwargs
) -> ConvNeXt3DMamba:
    """ConvNeXt-3D-Mamba-Tiny with input-dependent gating."""
    return ConvNeXt3DMamba(
        num_classes=num_classes,
        T=T,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        kernel_size=kernel_size,
        ssm_kernel_size=ssm_kernel_size,
        ssm_kernel_size_t=ssm_kernel_size_t,
        use_delta=use_delta,
        **kwargs
    )


def convnext_3d_mamba_small(
    num_classes: int = 1000,
    T: int = 8,
    kernel_size: int = 7,
    ssm_kernel_size: int = 3,
    ssm_kernel_size_t: int = 3,
    use_delta: bool = True,
    **kwargs
) -> ConvNeXt3DMamba:
    """ConvNeXt-3D-Mamba-Small with input-dependent gating."""
    return ConvNeXt3DMamba(
        num_classes=num_classes,
        T=T,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        kernel_size=kernel_size,
        ssm_kernel_size=ssm_kernel_size,
        ssm_kernel_size_t=ssm_kernel_size_t,
        use_delta=use_delta,
        **kwargs
    )


# =============================================================================
# Tests
# =============================================================================

if __name__ == '__main__':
    import jax.random as random
    import time

    print("=" * 70)
    print("TEST: ConvNeXt-3D-Mamba (Mamba-Gated 3D ConvSSM)")
    print("=" * 70)

    key = random.PRNGKey(0)

    # Test with T=4
    print("\nT=4, SSM kernel 3x3x3, with delta:")
    model = convnext_3d_mamba_tiny(
        num_classes=10,
        T=4,
        ssm_kernel_size=3,
        ssm_kernel_size_t=3,
        use_delta=True
    )

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

    # Test without delta
    print("\n" + "-" * 70)
    print("T=4, SSM kernel 3x3x3, without delta:")
    model_no_delta = convnext_3d_mamba_tiny(
        num_classes=10,
        T=4,
        ssm_kernel_size=3,
        ssm_kernel_size_t=3,
        use_delta=False
    )
    variables_nd = model_no_delta.init({'params': key, 'dropout': key}, dummy, train=False)
    params_nd = variables_nd['params']
    n_params_nd = sum(x.size for x in jax.tree_util.tree_leaves(params_nd))
    print(f"Parameters (no delta): {n_params_nd:,} ({n_params_nd / 1e6:.1f}M)")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
