"""Mamba-style SSM for Fourier-domain receptive field growth.

Key differences from simple linear SSM:
1. Input-dependent gating (Δ, B(x), C(x))
2. Frequency-selective receptive field expansion
3. Reset mechanism via forget gate

Design goals:
- T=0: small receptive field (high frequencies)
- T=N: large receptive field (all frequencies)
- Gates control growth rate and can reset
"""

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from typing import Optional


class MambaFourierSSM(nn.Module):
    """Mamba-style SSM operating in Fourier domain.

    For each spatial location (i,j) in Fourier space, we run:
        h_t = sigmoid(forget_gate) * h_{t-1} + sigmoid(input_gate) * (B_f * x_f)
        y = sigmoid(output_gate) * h_t

    Where gates are input-dependent (Mamba-style selectivity).

    The SSM grows receptive field by:
    - Starting with state h_0 = 0 (no context)
    - Each step accumulates more spatial context
    - Forget gate can reset the state (low freq = global reset)
    - Input gate controls what new info enters (high freq = local)

    Attributes:
        dim: Number of channels (depthwise)
        T: Number of SSM iterations
        kernel_size: Spatial kernel size for A, B
        d_state: Hidden state dimension expansion (default 1, i.e., same as input)
    """
    dim: int
    T: int = 8
    kernel_size: int = 7
    d_state: int = 1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_f: jnp.ndarray) -> jnp.ndarray:
        """Apply Mamba-style SSM in Fourier domain.

        Args:
            x_f: Input in Fourier domain (batch, H, W, C) complex

        Returns:
            Output in Fourier domain (batch, H, W, C) complex
        """
        B, H, W, C = x_f.shape
        k = self.kernel_size

        # === Mamba-style input-dependent projections ===
        # Project input to get gate modulations (in real domain of Fourier coeffs)
        x_real = jnp.concatenate([x_f.real, x_f.imag], axis=-1)  # (B, H, W, 2C)

        # Input gate: controls what new information enters
        # Computed per-channel from input
        input_gate_proj = nn.Dense(C, name='input_gate_proj')(x_real)
        input_gate = jax.nn.sigmoid(input_gate_proj)  # (B, H, W, C)

        # Forget gate: controls state decay / reset
        # Lower values = more forgetting = smaller receptive field
        forget_gate_proj = nn.Dense(C, name='forget_gate_proj')(x_real)
        forget_gate = jax.nn.sigmoid(forget_gate_proj + 1.0)  # bias towards remembering

        # Delta (Δ): controls discretization step / update rate
        # This is Mamba's key selectivity mechanism
        delta_proj = nn.Dense(C, name='delta_proj')(x_real)
        delta = jax.nn.softplus(delta_proj)  # Always positive, (B, H, W, C)

        # === Learnable spatial convolution kernels ===
        # A: state transition kernel (small, local)
        A_kernel = self.param(
            'A_kernel',
            nn.initializers.normal(0.02),
            (C, k, k),
            self.dtype
        )

        # B: input projection kernel
        B_kernel = self.param(
            'B_kernel',
            nn.initializers.normal(0.02),
            (C, k, k),
            self.dtype
        )

        # C: output projection kernel
        C_kernel = self.param(
            'C_kernel',
            nn.initializers.normal(0.02),
            (C, k, k),
            self.dtype
        )

        # Pad kernels to match spatial size and convert to Fourier
        def pad_and_fft(kernel):
            if H >= k and W >= k:
                pad_h = (H - k) // 2
                pad_w = (W - k) // 2
                padded = jnp.pad(
                    kernel,
                    ((0, 0), (pad_h, H - k - pad_h), (pad_w, W - k - pad_w)),
                    mode='constant', constant_values=0
                )
            else:
                start_h = (k - H) // 2
                start_w = (k - W) // 2
                padded = kernel[:, start_h:start_h+H, start_w:start_w+W]
            return jnp.fft.fft2(padded, axes=(-2, -1))

        A_f = pad_and_fft(A_kernel)  # (C, H, W) complex
        B_f = pad_and_fft(B_kernel)
        C_f = pad_and_fft(C_kernel)

        # Transpose for broadcasting: (C, H, W) -> (1, H, W, C)
        A_f = A_f.transpose(1, 2, 0)[None, ...]
        B_f = B_f.transpose(1, 2, 0)[None, ...]
        C_f = C_f.transpose(1, 2, 0)[None, ...]

        # === Mamba-style discretization ===
        # Convert continuous A to discrete A_bar using zero-order hold
        # A_bar = exp(delta * A) ≈ 1 + delta * A for small delta
        # We simplify to: A_bar = sigmoid(forget_gate) for stability

        # B_bar = delta * B (scaled input)
        # input_contribution = input_gate * B_f * x_f

        # === Run SSM with lax.scan ===
        def step_fn(h, t):
            """One SSM step with Mamba-style gating."""
            # Forget/decay the state (input-dependent)
            h_decayed = forget_gate * h

            # Input contribution (input-dependent gating)
            # delta controls the magnitude of update
            input_contrib = delta * input_gate * (B_f * x_f)

            # State update
            h_new = A_f * h_decayed + input_contrib

            return h_new, None

        # Initialize state
        h_init = jnp.zeros_like(x_f)

        # Run T steps
        h_final, _ = lax.scan(step_fn, h_init, jnp.arange(self.T))

        # Output projection with C kernel
        y_f = C_f * h_final

        return y_f


class MambaFourierSSMSimple(nn.Module):
    """Simplified Mamba-style SSM for debugging - REAL DOMAIN ONLY.

    Key features:
    - Input-dependent forget gate (controls receptive field growth)
    - Input-dependent input gate (selective attention)
    - Works in REAL domain only (no complex instabilities)
    - Minimal parameters for fast compilation

    Attributes:
        dim: Number of channels
        T: Number of SSM iterations
    """
    dim: int
    T: int = 8
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply simplified Mamba SSM in REAL domain.

        Args:
            x: (batch, H, W, C) real spatial features

        Returns:
            (batch, H, W, C) real spatial features
        """
        B, H, W, C = x.shape

        # === Mamba-style input-dependent gates ===
        # Forget gate: controls decay / receptive field size
        # Learned per-channel scalar + input modulation
        forget_bias = self.param('forget_bias', nn.initializers.constant(2.0), (C,))
        forget_scale = self.param('forget_scale', nn.initializers.constant(0.01), (C,))
        forget_gate = jax.nn.sigmoid(forget_bias + forget_scale * x)  # (B, H, W, C)

        # Input gate: controls what new info enters
        input_bias = self.param('input_bias', nn.initializers.constant(0.0), (C,))
        input_scale = self.param('input_scale', nn.initializers.constant(0.01), (C,))
        input_gate = jax.nn.sigmoid(input_bias + input_scale * x)

        # === Simple SSM transition ===
        # A: just a learned decay (no spatial kernel for speed)
        A_log = self.param('A_log', nn.initializers.constant(-1.0), (C,))
        A = jax.nn.sigmoid(A_log)  # (C,) in [0, 1]

        # B: learned input scale (small init for stability)
        B_scale = self.param('B_scale', nn.initializers.constant(0.1), (C,))

        # === Run SSM with lax.scan ===
        def step_fn(h, t):
            """One SSM step."""
            # State decay with input-dependent forget gate
            h_decayed = (A * forget_gate) * h

            # Input contribution with input-dependent gate
            input_contrib = (B_scale * input_gate) * x

            # State update
            h_new = h_decayed + input_contrib

            return h_new, None

        # Initialize and run
        h_init = jnp.zeros_like(x)
        h_final, _ = lax.scan(step_fn, h_init, jnp.arange(self.T))

        return h_final


class MambaFourierBlock(nn.Module):
    """A single block with Mamba-style SSM - REAL DOMAIN.

    Structure: x -> LayerNorm -> MambaSSM -> MLP -> + residual

    Attributes:
        dim: Number of channels
        T: SSM iterations
        expansion: MLP expansion ratio
    """
    dim: int
    T: int = 8
    expansion: int = 4
    use_simple_ssm: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass in REAL domain.

        Args:
            x: (batch, H, W, C) real features
            train: Training mode

        Returns:
            (batch, H, W, C) real features
        """
        residual = x

        # Layer norm
        x_norm = nn.LayerNorm()(x)

        # Mamba SSM in real domain
        h = MambaFourierSSMSimple(self.dim, T=self.T)(x_norm)

        # Pointwise MLP
        h_mlp = nn.Dense(self.dim * self.expansion)(h)
        h_mlp = nn.gelu(h_mlp)
        h_mlp = nn.Dense(self.dim)(h_mlp)

        # Residual connection
        return residual + h_mlp


class SimpleMambaConvNeXt(nn.Module):
    """Simplified ConvNeXt with Mamba SSM for debugging - REAL DOMAIN.

    Architecture:
    - Stem: 4x4 conv with stride 4 (224 -> 56)
    - 4 stages with downsampling
    - Mamba SSM blocks in real domain (no FFT instabilities)

    Attributes:
        num_classes: Number of output classes
        dims: Channel dimensions per stage
        depths: Number of blocks per stage
        T: SSM iterations
    """
    num_classes: int = 10
    dims: tuple = (64, 128, 256, 512)
    depths: tuple = (2, 2, 4, 2)
    T: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass in REAL domain.

        Args:
            x: (batch, H, W, 3) RGB image
            train: Training mode

        Returns:
            (batch, num_classes) logits
        """
        B = x.shape[0]

        # Stem: 4x4 conv stride 4
        x = nn.Conv(self.dims[0], (4, 4), strides=(4, 4), padding='VALID')(x)
        x = nn.LayerNorm()(x)  # (B, 56, 56, dims[0])

        # Stages
        for stage_idx, (dim, depth) in enumerate(zip(self.dims, self.depths)):
            # Downsample between stages (except first)
            if stage_idx > 0:
                x = nn.LayerNorm()(x)
                x = nn.Conv(dim, (2, 2), strides=(2, 2), padding='VALID')(x)

            # Apply Mamba blocks in REAL domain (no FFT)
            for block_idx in range(depth):
                x = MambaFourierBlock(
                    dim=dim,
                    T=self.T,
                    use_simple_ssm=True
                )(x, train=train)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, dims[-1])

        # Classifier
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.num_classes)(x)

        return logits


def create_simple_mamba_model(num_classes: int = 10, T: int = 4):
    """Create a simple Mamba-ConvNeXt model for debugging."""
    return SimpleMambaConvNeXt(
        num_classes=num_classes,
        dims=(48, 96, 192, 384),  # Smaller dims for faster debug
        depths=(1, 1, 2, 1),       # Fewer blocks
        T=T
    )
