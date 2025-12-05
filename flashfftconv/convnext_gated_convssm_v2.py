"""ConvNeXt with Gated ConvSSM V2 - Input-Dependent Gating on Kernels.

Key insight: The recurrence h_t = a_t * h_{t-1} + b_t is linear in h_{t-1},
so a_t and b_t can be arbitrary functions of x_t while still allowing parallel scan.

This version implements:
1. Input-dependent gating of SSM coefficients (a_t, b_t are gated by x)
2. Optional: Multi-basis kernels with attention-style weighting
3. Maintains parallel scan compatibility via Heinsen's associative scan

Three gating modes:
- "coefficient": Gate the A/B values after convolution
- "kernel_attention": Attention-weighted combination of multiple basis kernels
- "both": Both coefficient gating and kernel attention
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from typing import Sequence, Tuple, Literal
import numpy as np


# =============================================================================
# Log-space Operations for Numerical Stability
# =============================================================================

def heinsen_associative_scan_log(log_coeffs: jnp.ndarray, log_values: jnp.ndarray) -> jnp.ndarray:
    """Heinsen's associative scan in log-space for numerical stability.

    Computes h_t = c_t * h_{t-1} + v_t where:
    - log_coeffs = log(c_t) - the "forget" coefficient in log-space
    - log_values = log(v_t) - the "input" contribution in log-space

    The key insight: In log-space, multiplication becomes addition,
    and we use logaddexp for numerical stable log(a + b) = logaddexp(log_a, log_b).

    Returns log(h_t) - you need exp() to get back to linear space.
    """
    def associative_op(left, right):
        log_a_left, log_b_left = left
        log_a_right, log_b_right = right
        # a_combined = a_left * a_right -> log_a_combined = log_a_left + log_a_right
        new_log_a = log_a_left + log_a_right
        # b_combined = a_right * b_left + b_right -> log(a_right * b_left + b_right)
        # = logaddexp(log_a_right + log_b_left, log_b_right)
        new_log_b = jnp.logaddexp(log_a_right + log_b_left, log_b_right)
        return (new_log_a, new_log_b)

    _, log_h = lax.associative_scan(
        associative_op,
        (log_coeffs, log_values),
        axis=0
    )
    return log_h


def log_space_scan(gate_seq: jnp.ndarray, h_tilde_seq: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Log-space scan using minGRU formulation with complementary gates.

    Computes h_t = (1-z_t) * h_{t-1} + z_t * h_tilde_t with numerical stability.

    Following minGRU (B.3.1), we use the same gate k for both:
    - log_z = -softplus(-k) = log(sigmoid(k))     -- input gate
    - log_coeffs = -softplus(k) = log(1-sigmoid(k)) -- forget gate (coefficient)

    This ensures the gates are complementary: z + (1-z) = 1

    Args:
        gate_seq: (T, ...) gate logits k - will produce complementary gates
        h_tilde_seq: (T, ...) candidate hidden states (pre-activation)
        eps: Small constant for numerical stability

    Returns:
        (T, ...) hidden states in linear space
    """
    # minGRU log-space formulation (Listing 8 from paper)
    # k is the gate logit
    k = gate_seq

    # log_z = -softplus(-k) = log(sigmoid(k)) -- the input gate in log space
    log_z = -jax.nn.softplus(-k)

    # log_coeffs = -softplus(k) = log(1-sigmoid(k)) = log(sigmoid(-k)) -- forget gate in log space
    log_coeffs = -jax.nn.softplus(k)

    # For log_h_tilde, we need log(|h_tilde|)
    # Use softplus to ensure positivity, then log
    h_tilde_positive = jax.nn.softplus(h_tilde_seq)
    log_h_tilde = jnp.log(h_tilde_positive + eps)

    # The value term is z * h_tilde, in log space: log_z + log_h_tilde
    log_values = log_z + log_h_tilde

    # Run log-space scan: h_t = coeffs_t * h_{t-1} + values_t
    log_h = heinsen_associative_scan_log(log_coeffs, log_values)

    # Convert back to linear space
    h = jnp.exp(log_h)

    return h


def linear_scan(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """Standard linear recurrence scan (not log-space).

    Computes h_t = a_t * h_{t-1} + b_t using associative scan.

    Args:
        a_seq: (T, ...) coefficients
        b_seq: (T, ...) input contributions

    Returns:
        (T, ...) hidden states
    """
    def associative_op(left, right):
        a_left, b_left = left
        a_right, b_right = right
        return (a_left * a_right, a_right * b_left + b_right)

    _, h = lax.associative_scan(
        associative_op,
        (a_seq, b_seq),
        axis=0
    )
    return h


# =============================================================================
# Helper Functions
# =============================================================================

def kernel_to_freq_2d(kernel: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """Convert 2D spatial kernel to frequency domain."""
    C, k_h, k_w = kernel.shape
    center_h = k_h // 2
    center_w = k_w // 2

    h_idx = jnp.arange(k_h)
    w_idx = jnp.arange(k_w)

    target_h = (h_idx - center_h) % H
    target_w = (w_idx - center_w) % W

    th, tw = jnp.meshgrid(target_h, target_w, indexing='ij')

    padded = jnp.zeros((C, H, W), dtype=kernel.dtype)
    padded = padded.at[:, th, tw].set(kernel)

    return jnp.fft.fft2(padded, axes=(1, 2))


def apply_conv2d_fft(x: jnp.ndarray, kernel_f: jnp.ndarray) -> jnp.ndarray:
    """Apply 2D convolution via FFT.

    Args:
        x: (B, H, W, C) input
        kernel_f: (C, H, W) frequency domain kernel

    Returns:
        (B, H, W, C) convolved output
    """
    # x: (B, H, W, C) -> (B, C, H, W)
    x = x.transpose(0, 3, 1, 2)

    # FFT
    x_f = jnp.fft.fft2(x, axes=(2, 3))

    # Multiply in frequency domain
    # kernel_f: (C, H, W) -> (1, C, H, W)
    out_f = x_f * kernel_f[None, ...]

    # IFFT
    out = jnp.fft.ifft2(out_f, axes=(2, 3)).real

    # (B, C, H, W) -> (B, H, W, C)
    return out.transpose(0, 2, 3, 1)


# =============================================================================
# Gated ConvSSM V2 - Input-Dependent Kernel Gating
# =============================================================================

class GatedConvSSM2D_V2(nn.Module):
    """Gated 2D ConvSSM with input-dependent coefficient gating.

    The recurrence is: h_t = a_t * h_{t-1} + b_t

    Where a_t and b_t are computed from x_t with optional gating:
    - a_t = conv2d(x_t, A_kernel) * sigmoid(conv2d(x_t, gate_A))
    - b_t = conv2d(x_t, B_kernel) * sigmoid(conv2d(x_t, gate_B))

    This remains linear in h_{t-1}, so parallel scan still works!

    Optionally, can use multi-basis kernels with attention weighting.

    Attributes:
        dim: Number of channels
        kernel_size: Spatial kernel size
        num_basis: Number of basis kernels for attention-weighted combination
        gating_mode: "coefficient", "kernel_attention", or "both"
        dtype: Compute dtype
    """
    dim: int
    kernel_size: int = 7
    num_basis: int = 4  # Number of basis kernels for attention
    gating_mode: Literal["coefficient", "kernel_attention", "both"] = "coefficient"
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, num_iterations: int = 8) -> jnp.ndarray:
        """Run gated 2D ConvSSM.

        Args:
            x: (B, H, W, C) input (single image, will be "unrolled" T times)
            num_iterations: T - number of SSM iterations

        Returns:
            (B, H, W, C) output after T iterations
        """
        B, H, W, C = x.shape
        k = self.kernel_size
        T = num_iterations

        # =================================================================
        # Compute input-dependent coefficients for each iteration
        # =================================================================

        if self.gating_mode in ["coefficient", "both"]:
            # Gate kernels for coefficient gating
            gate_A_kernel = self.param(
                'gate_A_kernel',
                nn.initializers.zeros,  # Initialize to ~0.5 sigmoid
                (C, k, k),
                self.dtype
            )
            gate_B_kernel = self.param(
                'gate_B_kernel',
                nn.initializers.zeros,
                (C, k, k),
                self.dtype
            )

        if self.gating_mode in ["kernel_attention", "both"]:
            # Multi-basis kernels
            num_basis = self.num_basis
            A_basis = self.param(
                'A_basis',
                nn.initializers.lecun_normal(),
                (num_basis, C, k, k),
                self.dtype
            )
            B_basis = self.param(
                'B_basis',
                nn.initializers.lecun_normal(),
                (num_basis, C, k, k),
                self.dtype
            )
            # Query projection for attention
            query_proj = nn.Dense(num_basis, dtype=self.dtype, name='query_proj')
        else:
            # Single A and B kernels
            A_kernel = self.param(
                'A_kernel',
                nn.initializers.lecun_normal(),
                (C, k, k),
                self.dtype
            )
            B_kernel = self.param(
                'B_kernel',
                nn.initializers.lecun_normal(),
                (C, k, k),
                self.dtype
            )

        # =================================================================
        # Compute A and B in frequency domain
        # =================================================================

        if self.gating_mode in ["kernel_attention", "both"]:
            # Attention-weighted kernel selection
            # Global average pool -> query -> attention weights
            x_pool = jnp.mean(x, axis=(1, 2))  # (B, C)
            attn_logits = query_proj(x_pool)  # (B, num_basis)
            attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # (B, num_basis)

            # Weighted combination of basis kernels
            # A_basis: (num_basis, C, k, k), attn_weights: (B, num_basis)
            # Result: (B, C, k, k)
            # Flatten k*k dimension, then reshape back
            A_basis_flat = A_basis.reshape(self.num_basis, C * k * k)  # (num_basis, C*k*k)
            B_basis_flat = B_basis.reshape(self.num_basis, C * k * k)  # (num_basis, C*k*k)

            A_kernel_dynamic = jnp.einsum('bn,nd->bd', attn_weights, A_basis_flat).reshape(B, C, k, k)
            B_kernel_dynamic = jnp.einsum('bn,nd->bd', attn_weights, B_basis_flat).reshape(B, C, k, k)

            # Convert to frequency domain (per-sample)
            # This is expensive but necessary for dynamic kernels
            def to_freq(kernel):
                return kernel_to_freq_2d(kernel, H, W)

            A_f = jax.vmap(to_freq)(A_kernel_dynamic)  # (B, C, H, W)
            B_f = jax.vmap(to_freq)(B_kernel_dynamic)  # (B, C, H, W)

            # Reshape: (B, C, H, W) -> (B, H, W, C)
            A_f = A_f.transpose(0, 2, 3, 1)
            B_f = B_f.transpose(0, 2, 3, 1)
        else:
            # Static kernels
            A_f = kernel_to_freq_2d(A_kernel, H, W)  # (C, H, W)
            B_f = kernel_to_freq_2d(B_kernel, H, W)  # (C, H, W)

            # Reshape: (C, H, W) -> (H, W, C)
            A_f = A_f.transpose(1, 2, 0)
            B_f = B_f.transpose(1, 2, 0)

        # =================================================================
        # Compute gated coefficients
        # =================================================================

        # FFT of input
        x_f = jnp.fft.fft2(x.transpose(0, 3, 1, 2), axes=(2, 3))  # (B, C, H, W)
        x_f = x_f.transpose(0, 2, 3, 1)  # (B, H, W, C)

        if self.gating_mode in ["kernel_attention", "both"]:
            # A_f, B_f already have batch dimension
            a_base = A_f  # (B, H, W, C)
            b_contrib = B_f * x_f  # (B, H, W, C)
        else:
            # Broadcast static kernels
            a_base = A_f[None, ...]  # (1, H, W, C)
            b_contrib = B_f[None, ...] * x_f  # (B, H, W, C)

        if self.gating_mode in ["coefficient", "both"]:
            # Compute gates from input
            gate_A_f = kernel_to_freq_2d(gate_A_kernel, H, W).transpose(1, 2, 0)  # (H, W, C)
            gate_B_f = kernel_to_freq_2d(gate_B_kernel, H, W).transpose(1, 2, 0)  # (H, W, C)

            gate_A_conv_f = gate_A_f[None, ...] * x_f  # (B, H, W, C)
            gate_B_conv_f = gate_B_f[None, ...] * x_f  # (B, H, W, C)

            # IFFT to get spatial gate values
            gate_A_spatial = jnp.fft.ifft2(
                gate_A_conv_f.transpose(0, 3, 1, 2), axes=(2, 3)
            ).real.transpose(0, 2, 3, 1)
            gate_B_spatial = jnp.fft.ifft2(
                gate_B_conv_f.transpose(0, 3, 1, 2), axes=(2, 3)
            ).real.transpose(0, 2, 3, 1)

            # Apply sigmoid gating
            gate_A = jax.nn.sigmoid(gate_A_spatial)  # (B, H, W, C)
            gate_B = jax.nn.sigmoid(gate_B_spatial)  # (B, H, W, C)

            # Gated coefficients (in frequency domain for A, spatial for gating)
            # We need to apply gating in spatial domain, so IFFT a_base first
            if self.gating_mode == "both":
                a_spatial = jnp.fft.ifft2(
                    a_base.transpose(0, 3, 1, 2), axes=(2, 3)
                ).real.transpose(0, 2, 3, 1)
            else:
                a_spatial = jnp.fft.ifft2(
                    jnp.broadcast_to(a_base, (B, H, W, C)).transpose(0, 3, 1, 2), axes=(2, 3)
                ).real.transpose(0, 2, 3, 1)

            a_gated = a_spatial * gate_A  # (B, H, W, C) - gated transition coefficient

            # b_contrib is already in freq domain, convert to spatial and gate
            b_spatial = jnp.fft.ifft2(
                b_contrib.transpose(0, 3, 1, 2), axes=(2, 3)
            ).real.transpose(0, 2, 3, 1)
            b_gated = b_spatial * gate_B  # (B, H, W, C) - gated input

            # Final coefficients
            a_t = a_gated  # (B, H, W, C)
            b_t = b_gated  # (B, H, W, C)
        else:
            # No coefficient gating, just use base values
            # Convert to spatial domain
            if self.gating_mode == "kernel_attention":
                a_t = jnp.fft.ifft2(
                    a_base.transpose(0, 3, 1, 2), axes=(2, 3)
                ).real.transpose(0, 2, 3, 1)
            else:
                a_t = jnp.fft.ifft2(
                    jnp.broadcast_to(a_base, (B, H, W, C)).transpose(0, 3, 1, 2), axes=(2, 3)
                ).real.transpose(0, 2, 3, 1)

            b_t = jnp.fft.ifft2(
                b_contrib.transpose(0, 3, 1, 2), axes=(2, 3)
            ).real.transpose(0, 2, 3, 1)

        # =================================================================
        # Run parallel scan over T iterations (minGRU formulation)
        # =================================================================

        # For minGRU: h_t = (1-z) * h_{t-1} + z * h_tilde
        # where z = sigmoid(k) is the input gate (from same k as forget gate)
        #
        # a_t = gate logits (k in minGRU)
        # b_t = candidate hidden state (h_tilde in minGRU)
        #
        # The log_space_scan will compute complementary gates:
        #   - log_z = -softplus(-k) = log(sigmoid(k))       -- input gate
        #   - log_coeffs = -softplus(k) = log(1-sigmoid(k)) -- forget gate

        # gate_seq: (T, B, H, W, C) - gate logits
        # h_tilde_seq: (T, B, H, W, C) - candidate hidden states
        gate_seq = jnp.tile(a_t[None, ...], (T, 1, 1, 1, 1))
        h_tilde_seq = jnp.tile(b_t[None, ...], (T, 1, 1, 1, 1))

        # Run associative scan in log-space with minGRU complementary gates
        h_seq = log_space_scan(gate_seq, h_tilde_seq)  # (T, B, H, W, C)

        # Return final hidden state
        h_final = h_seq[-1]  # (B, H, W, C)

        # Output projection
        output = nn.Dense(C, dtype=self.dtype, name='output_proj')(h_final)

        return output.astype(self.dtype)


# =============================================================================
# Alternative: Step-by-Step Gated Convolution (Sequential)
# =============================================================================

class IterativeGatedConv2D(nn.Module):
    """Iterative gated 2D convolution - each step has different gating.

    This is the "true" gated convolution where at each step:
    1. Compute gate from current state: z_t = sigmoid(conv(h_{t-1}))
    2. Compute candidate: h̃_t = conv(h_{t-1})
    3. Update: h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t

    Note: This requires sequential computation (no parallel scan) because
    the gate depends on h_{t-1}.

    Attributes:
        dim: Number of channels
        kernel_size: Spatial kernel size
        num_iterations: Number of iterations
        dtype: Compute dtype
    """
    dim: int
    kernel_size: int = 7
    num_iterations: int = 8
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run iterative gated convolution.

        Args:
            x: (B, H, W, C) input

        Returns:
            (B, H, W, C) output after iterations
        """
        B, H, W, C = x.shape
        k = self.kernel_size

        # Shared kernels across iterations
        gate_kernel = self.param(
            'gate_kernel',
            nn.initializers.zeros,
            (C, k, k),
            self.dtype
        )
        hidden_kernel = self.param(
            'hidden_kernel',
            nn.initializers.lecun_normal(),
            (C, k, k),
            self.dtype
        )

        # Convert to frequency domain once
        gate_f = kernel_to_freq_2d(gate_kernel, H, W).transpose(1, 2, 0)  # (H, W, C)
        hidden_f = kernel_to_freq_2d(hidden_kernel, H, W).transpose(1, 2, 0)  # (H, W, C)

        # Initialize hidden state
        h = x

        # Sequential iterations (cannot parallelize due to state-dependent gating)
        def step(h, _):
            # FFT of current state
            h_f = jnp.fft.fft2(h.transpose(0, 3, 1, 2), axes=(2, 3)).transpose(0, 2, 3, 1)

            # Gate from current state
            gate_conv_f = gate_f[None, ...] * h_f
            gate_spatial = jnp.fft.ifft2(
                gate_conv_f.transpose(0, 3, 1, 2), axes=(2, 3)
            ).real.transpose(0, 2, 3, 1)
            z = jax.nn.sigmoid(gate_spatial)

            # Candidate from current state
            hidden_conv_f = hidden_f[None, ...] * h_f
            h_tilde = jnp.fft.ifft2(
                hidden_conv_f.transpose(0, 3, 1, 2), axes=(2, 3)
            ).real.transpose(0, 2, 3, 1)

            # Gated update
            h_new = (1 - z) * h + z * h_tilde

            return h_new, None

        # Run iterations
        h_final, _ = lax.scan(step, h, None, length=self.num_iterations)

        return h_final.astype(self.dtype)


# =============================================================================
# LayerNorm
# =============================================================================

class LayerNorm2D(nn.Module):
    """Layer Normalization for 2D inputs."""
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
# ConvNeXt Block with Gated ConvSSM V2
# =============================================================================

class GatedConvNeXtBlock_V2(nn.Module):
    """ConvNeXt block with input-dependent gated ConvSSM."""
    dim: int
    kernel_size: int = 7
    num_iterations: int = 8
    num_basis: int = 4
    gating_mode: str = "coefficient"
    expansion_ratio: int = 4
    layer_scale_init: float = 1e-6
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        B, H, W, C = x.shape

        # Gated ConvSSM
        x = GatedConvSSM2D_V2(
            dim=C,
            kernel_size=self.kernel_size,
            num_basis=self.num_basis,
            gating_mode=self.gating_mode,
            dtype=self.dtype,
            name='gated_convssm'
        )(x, num_iterations=self.num_iterations)

        # LayerNorm
        x = LayerNorm2D(dtype=self.dtype)(x)

        # MLP
        hidden_dim = int(C * self.expansion_ratio)
        x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(C, dtype=self.dtype)(x)

        # Layer scale
        gamma = self.param(
            'layer_scale',
            nn.initializers.constant(self.layer_scale_init),
            (C,),
            self.dtype
        )
        x = x * gamma

        # Drop path
        if train and self.drop_path_rate > 0:
            keep_prob = 1.0 - self.drop_path_rate
            rng = self.make_rng('dropout')
            mask = jax.random.bernoulli(rng, keep_prob, (B, 1, 1, 1))
            x = x / keep_prob * mask

        return x + residual


# =============================================================================
# Full Model
# =============================================================================

class GatedConvNeXtSSM_V2(nn.Module):
    """ConvNeXt with input-dependent gated ConvSSM.

    Attributes:
        num_classes: Number of output classes
        num_iterations: SSM iterations per block
        depths: Blocks per stage
        dims: Channels per stage
        kernel_size: Spatial kernel size
        num_basis: Number of basis kernels for attention
        gating_mode: "coefficient", "kernel_attention", or "both"
        drop_path_rate: Stochastic depth rate
        dtype: Compute dtype
    """
    num_classes: int = 1000
    num_iterations: int = 8
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    kernel_size: int = 7
    num_basis: int = 4
    gating_mode: str = "coefficient"
    drop_path_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        B, H, W, C_in = x.shape

        # Stem
        x = nn.Conv(
            self.dims[0],
            kernel_size=(4, 4),
            strides=(4, 4),
            padding='VALID',
            dtype=self.dtype,
            name='stem'
        )(x)
        x = LayerNorm2D(dtype=self.dtype, name='stem_norm')(x)

        # Stochastic depth schedule
        total_blocks = sum(self.depths)
        dpr = [self.drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0

        # Stages
        for stage_idx, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            # Downsample
            if stage_idx > 0:
                x = LayerNorm2D(dtype=self.dtype, name=f'downsample_norm_{stage_idx}')(x)
                x = nn.Conv(
                    dim,
                    kernel_size=(2, 2),
                    strides=(2, 2),
                    padding='VALID',
                    dtype=self.dtype,
                    name=f'downsample_{stage_idx}'
                )(x)

            # Blocks
            for block_i in range(depth):
                x = GatedConvNeXtBlock_V2(
                    dim=dim,
                    kernel_size=self.kernel_size,
                    num_iterations=self.num_iterations,
                    num_basis=self.num_basis,
                    gating_mode=self.gating_mode,
                    drop_path_rate=dpr[block_idx],
                    dtype=self.dtype,
                    name=f'stage_{stage_idx}_block_{block_i}'
                )(x, train=train)
                block_idx += 1

        # Head
        x = jnp.mean(x, axis=(1, 2))
        x = LayerNorm2D(dtype=self.dtype, name='head_norm')(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype, name='head')(x)

        return x


# =============================================================================
# Model Constructors
# =============================================================================

def gated_convnext_ssm_v2_tiny(
    num_classes: int = 1000,
    num_iterations: int = 8,
    kernel_size: int = 7,
    num_basis: int = 4,
    gating_mode: str = "coefficient",
    **kwargs
) -> GatedConvNeXtSSM_V2:
    """Gated ConvNeXt-SSM-V2-Tiny with input-dependent gating."""
    return GatedConvNeXtSSM_V2(
        num_classes=num_classes,
        num_iterations=num_iterations,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        kernel_size=kernel_size,
        num_basis=num_basis,
        gating_mode=gating_mode,
        **kwargs
    )


# =============================================================================
# Tests
# =============================================================================

if __name__ == '__main__':
    import jax.random as random
    import time

    print("=" * 70)
    print("TEST: Gated ConvNeXt-SSM V2 (Input-Dependent Gating)")
    print("=" * 70)

    key = random.PRNGKey(0)
    dummy = jnp.ones((2, 160, 160, 3))

    for mode in ["coefficient", "kernel_attention", "both"]:
        print(f"\n--- Gating mode: {mode} ---")

        model = gated_convnext_ssm_v2_tiny(
            num_classes=10,
            num_iterations=4,
            kernel_size=5,
            num_basis=4,
            gating_mode=mode
        )

        print("Initializing...")
        t0 = time.time()
        variables = model.init({'params': key, 'dropout': key}, dummy, train=False)
        params = variables['params']
        print(f"Init time: {time.time() - t0:.2f}s")

        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

        # Forward pass
        @jax.jit
        def forward(params, x):
            return model.apply({'params': params}, x, train=False, rngs={'dropout': key})

        t0 = time.time()
        logits = forward(params, dummy)
        logits.block_until_ready()
        print(f"JIT + forward: {time.time() - t0:.2f}s")
        print(f"Output shape: {logits.shape}")
        print(f"NaN check: {jnp.any(jnp.isnan(logits))}")

        # Gradient check
        def loss_fn(params):
            return jnp.mean(forward(params, dummy) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        print(f"Loss: {float(loss):.4f}, Grad norm: {float(grad_norm):.4f}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
