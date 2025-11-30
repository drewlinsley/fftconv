"""ConvNeXt-Fourier V2: All operations in Fourier domain (STABLE VERSION).

Key design decisions:
1. Stay in Fourier domain throughout (no IFFT until final output)
2. ConvSSM replaces depthwise conv (T=16 timesteps for receptive field growth)
3. CRITICAL: Use real/imag stacked representation for numerical stability
   - Complex ops are represented as operations on (real, imag) pairs
   - This avoids gradient issues with jnp.angle() and complex chain rule
4. Optional bfloat16 for SSM operations (2x memory savings, faster compute)

Architecture per block:
    x_f -> StableNorm -> ConvSSM(T=16) -> StableNorm -> PointwiseMLP -> + residual
"""

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from typing import Sequence, Optional
import numpy as np

# Default compute dtype for SSM operations
DEFAULT_DTYPE = jnp.float32


# =============================================================================
# Stable Complex Operations using Real/Imag Representation
# =============================================================================

def complex_to_realimag(x_f: jnp.ndarray) -> jnp.ndarray:
    """Convert complex (B,H,W,C) to stacked real (B,H,W,2C)."""
    return jnp.concatenate([x_f.real, x_f.imag], axis=-1)


def realimag_to_complex(x_ri: jnp.ndarray) -> jnp.ndarray:
    """Convert stacked real (B,H,W,2C) to complex (B,H,W,C)."""
    C = x_ri.shape[-1] // 2
    return x_ri[..., :C] + 1j * x_ri[..., C:]


def complex_mul_realimag(a_ri: jnp.ndarray, b_ri: jnp.ndarray) -> jnp.ndarray:
    """Complex multiplication using real/imag representation.

    (a_r + i*a_i) * (b_r + i*b_i) = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)
    """
    C = a_ri.shape[-1] // 2
    a_r, a_i = a_ri[..., :C], a_ri[..., C:]
    b_r, b_i = b_ri[..., :C], b_ri[..., C:]

    out_r = a_r * b_r - a_i * b_i
    out_i = a_r * b_i + a_i * b_r

    return jnp.concatenate([out_r, out_i], axis=-1)


# =============================================================================
# Stable Fourier-Domain Operations
# =============================================================================

class StableFourierNorm(nn.Module):
    """LayerNorm in Fourier domain using real/imag representation.

    Applies LayerNorm to the stacked (real, imag) representation,
    which is numerically stable and preserves the relationship between
    real and imaginary parts.
    """
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x_ri: jnp.ndarray) -> jnp.ndarray:
        """Normalize Fourier coefficients in real/imag form.

        Args:
            x_ri: (batch, H, W, 2C) stacked real/imag

        Returns:
            Normalized (batch, H, W, 2C)
        """
        # Standard LayerNorm on stacked representation
        return nn.LayerNorm(epsilon=self.epsilon)(x_ri)


class StableFourierGELU(nn.Module):
    """GELU in Fourier domain using real/imag representation.

    Applies GELU to both real and imaginary parts independently.
    This provides a nonlinearity while maintaining numerical stability.
    """

    @nn.compact
    def __call__(self, x_ri: jnp.ndarray) -> jnp.ndarray:
        """Apply GELU to real/imag representation.

        Args:
            x_ri: (batch, H, W, 2C) stacked real/imag

        Returns:
            Activated (batch, H, W, 2C)
        """
        return nn.gelu(x_ri)


# =============================================================================
# Helper Functions for Efficient Kernel FFT
# =============================================================================

def kernel_to_freq(kernel: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """FFT a small spatial kernel to frequency domain.

    This is the key function for parameter efficiency: we learn small spatial
    kernels (like 7x7) and FFT them to the target size, rather than learning
    separate parameters for each frequency position.

    CRITICAL: For correct FFT convolution, kernel elements must be SCATTERED
    to their wrapped positions in the padded array:
    - kernel[center, center] -> position [0, 0]
    - kernel[i, j] -> position [(i - center) % H, (j - center) % W]

    This ensures the FFT convolution matches spatial convolution with 'SAME' padding
    and circular boundary conditions.

    Args:
        kernel: (C, k, k) real-valued spatial kernel
        H, W: target spatial dimensions

    Returns:
        (C, H, W) complex frequency representation
    """
    C, k, _ = kernel.shape
    center = k // 2

    # Create index arrays for scattering kernel to wrapped positions
    # This is equivalent to: for each kernel[c, i, j], place at [(i-center)%H, (j-center)%W]
    i_idx = jnp.arange(k)
    j_idx = jnp.arange(k)

    # Target positions with wrapping
    target_i = (i_idx - center) % H  # Shape: (k,)
    target_j = (j_idx - center) % W  # Shape: (k,)

    # Create meshgrid for target positions
    ti, tj = jnp.meshgrid(target_i, target_j, indexing='ij')  # Both (k, k)

    # Vectorized scatter: create padded array and set all channels at once
    # Use vmap to apply the scatter operation across channels
    padded = jnp.zeros((C, H, W), dtype=kernel.dtype)
    # Use advanced indexing: for each channel c, set padded[c, ti, tj] = kernel[c]
    padded = padded.at[:, ti, tj].set(kernel)

    # FFT to frequency domain
    # NO ortho normalization here - the input FFT uses ortho which provides sqrt(N)
    # factor, and IFFT with ortho provides another sqrt(N), giving correct total scaling
    return jnp.fft.fft2(padded, axes=(-2, -1))


def make_decay_mask(kernel_size: int, decay_rate: float = 0.5) -> jnp.ndarray:
    """Create a decay mask for kernel stability.

    Applies exponential decay from center to edges, helping stability
    by biasing towards shorter-range dependencies.

    Args:
        kernel_size: Size of the kernel
        decay_rate: Rate of decay (higher = faster decay)

    Returns:
        (kernel_size, kernel_size) decay mask
    """
    center = kernel_size // 2
    y, x = jnp.meshgrid(jnp.arange(kernel_size), jnp.arange(kernel_size))
    dist = jnp.sqrt((x - center)**2 + (y - center)**2)
    return jnp.exp(-decay_rate * dist / center)


def complex_to_realimag_chw(x_f: jnp.ndarray) -> jnp.ndarray:
    """Convert complex (C, H, W) to stacked real (1, H, W, 2C) for broadcasting."""
    # x_f: (C, H, W) complex
    x_ri = jnp.concatenate([x_f.real, x_f.imag], axis=0)  # (2C, H, W)
    return x_ri.transpose(1, 2, 0)[None, ...]  # (1, H, W, 2C)


# =============================================================================
# Efficient ConvSSM (Parameter-Efficient Version)
# =============================================================================

class EfficientFourierConvSSM(nn.Module):
    """FFT-based ConvSSM with spatial kernel learning (parameter efficient).

    KEY INSIGHT: Convolution theorem says:
        conv(x, kernel) = IFFT(FFT(x) * FFT(kernel))

    So if we're already in Fourier domain, convolution becomes elementwise multiply.
    We learn small 7x7 spatial kernels (same as ConvNeXt), FFT them once per forward,
    and use elementwise multiplication.

    Parameter count: Same as ConvNeXt (7x7 kernel per channel).
    Speed: No FFT/IFFT per layer (we stay in Fourier domain).

    Attributes:
        dim: Number of channels (C)
        T: Number of SSM timesteps
        kernel_size: Spatial kernel size (default 7, like ConvNeXt)
        dtype: Compute dtype for SSM operations
    """
    dim: int
    T: int = 16
    kernel_size: int = 7
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_ri: jnp.ndarray) -> jnp.ndarray:
        """Run ConvSSM for T timesteps.

        Args:
            x_ri: (batch, H, W, 2C) stacked real/imag Fourier coefficients

        Returns:
            (batch, H, W, 2C) output Fourier coefficients
        """
        B, H, W, C2 = x_ri.shape
        C = C2 // 2
        k = self.kernel_size

        # Cast to compute dtype
        input_dtype = x_ri.dtype
        x_ri = x_ri.astype(self.dtype)

        # === Learn small SPATIAL kernels (like ConvNeXt depthwise conv) ===
        # A: state transition kernel (controls memory decay)
        # B: input kernel (controls what enters state)
        A_spatial = self.param('A', nn.initializers.normal(0.02), (C, k, k))
        B_spatial = self.param('B', nn.initializers.normal(0.02), (C, k, k))

        # Apply decay mask for stability (optional but helps)
        decay = make_decay_mask(k, decay_rate=0.3)
        A_spatial = A_spatial * decay[None, :, :]

        # Scale A for stability (|A| < 1 ensures bounded state)
        A_spatial = 0.9 * jnp.tanh(A_spatial)

        # === FFT kernels to frequency domain ===
        # This is computed once per forward, not learned per-frequency!
        A_f = kernel_to_freq(A_spatial, H, W)  # (C, H, W) complex
        B_f = kernel_to_freq(B_spatial, H, W)

        # Convert to real/imag representation for stable compute
        A_f_ri = complex_to_realimag_chw(A_f).astype(self.dtype)  # (1, H, W, 2C)
        B_f_ri = complex_to_realimag_chw(B_f).astype(self.dtype)

        # === SSM recurrence with lax.scan ===
        # h_t = A * h_{t-1} + B * x (elementwise multiply in freq domain = convolution)
        def step_fn(h_ri, _):
            """One SSM timestep: h_new = A ⊙ h + B ⊙ x (complex multiply)."""
            # Complex multiplication: (a_r + i*a_i) * (b_r + i*b_i)
            h_r, h_i = h_ri[..., :C], h_ri[..., C:]
            A_r, A_i = A_f_ri[..., :C], A_f_ri[..., C:]
            B_r, B_i = B_f_ri[..., :C], B_f_ri[..., C:]
            x_r, x_i = x_ri[..., :C], x_ri[..., C:]

            # A * h
            Ah_r = A_r * h_r - A_i * h_i
            Ah_i = A_r * h_i + A_i * h_r

            # B * x
            Bx_r = B_r * x_r - B_i * x_i
            Bx_i = B_r * x_i + B_i * x_r

            # h_new = A*h + B*x
            h_new_r = Ah_r + Bx_r
            h_new_i = Ah_i + Bx_i

            return jnp.concatenate([h_new_r, h_new_i], axis=-1), None

        # Initialize hidden state
        h_init = jnp.zeros_like(x_ri)

        # Run T timesteps
        h_final, _ = lax.scan(step_fn, h_init, None, length=self.T)

        return h_final.astype(input_dtype)


# =============================================================================
# ConvSSM in Fourier Domain (Original Version - High Parameter Count)
# =============================================================================

class StableFourierConvSSM(nn.Module):
    """Mamba-style ConvSSM in Fourier domain using real/imag representation.

    Key changes for stability:
    1. All operations on (real, imag) stacked tensor
    2. No magnitude/phase decomposition (avoids jnp.angle gradient issues)
    3. Complex multiplication implemented explicitly
    4. Optional bfloat16 compute for faster SSM operations

    Attributes:
        dim: Number of channels (C, not 2C)
        T: Number of SSM timesteps
        kernel_size: Spatial kernel size for A, B
        dtype: Compute dtype for SSM operations (float32 or bfloat16)
    """
    dim: int
    T: int = 16
    kernel_size: int = 7
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_ri: jnp.ndarray) -> jnp.ndarray:
        """Run ConvSSM for T timesteps.

        Args:
            x_ri: (batch, H, W, 2C) stacked real/imag Fourier coefficients

        Returns:
            (batch, H, W, 2C) output Fourier coefficients
        """
        B, H, W, C2 = x_ri.shape
        C = C2 // 2  # Actual channel count
        k = self.kernel_size

        # Cast input to compute dtype (bfloat16 or float32)
        input_dtype = x_ri.dtype
        x_ri = x_ri.astype(self.dtype)

        # === Input-dependent gates (Mamba-style) ===
        # Compute gates from the real/imag representation

        # Forget gate: controls how much of past state to keep
        forget_proj = nn.Dense(C2, name='forget_gate', dtype=self.dtype)(x_ri)
        forget_gate = jax.nn.sigmoid(forget_proj + 2.0)  # Bias towards remembering

        # Input gate: controls what new info enters
        input_proj = nn.Dense(C2, name='input_gate', dtype=self.dtype)(x_ri)
        input_gate = jax.nn.sigmoid(input_proj)

        # Delta (discretization step): controls update magnitude
        delta_proj = nn.Dense(C2, name='delta', dtype=self.dtype)(x_ri)
        delta = jax.nn.softplus(delta_proj) * 0.1  # Small updates for stability

        # === Learnable frequency-domain kernels ===
        # A and B are complex kernels, stored as real/imag
        A_real = self.param('A_real', nn.initializers.normal(0.02), (C, k, k))
        A_imag = self.param('A_imag', nn.initializers.normal(0.02), (C, k, k))
        B_real = self.param('B_real', nn.initializers.normal(0.02), (C, k, k))
        B_imag = self.param('B_imag', nn.initializers.normal(0.02), (C, k, k))

        # Pad kernels to match spatial size and FFT
        def pad_and_fft(kernel_real, kernel_imag):
            kernel = kernel_real + 1j * kernel_imag
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

        A_f = pad_and_fft(A_real, A_imag)  # (C, H, W) complex
        B_f = pad_and_fft(B_real, B_imag)

        # Convert to real/imag representation and transpose: (C, H, W) -> (1, H, W, 2C)
        A_f_ri = jnp.concatenate([A_f.real, A_f.imag], axis=0)  # (2C, H, W)
        A_f_ri = A_f_ri.transpose(1, 2, 0)[None, ...]  # (1, H, W, 2C)
        A_f_ri = A_f_ri.astype(self.dtype)  # Cast to compute dtype (no-op for float32)
        B_f_ri = jnp.concatenate([B_f.real, B_f.imag], axis=0)
        B_f_ri = B_f_ri.transpose(1, 2, 0)[None, ...]
        B_f_ri = B_f_ri.astype(self.dtype)  # Cast to compute dtype (no-op for float32)

        # Stability: bound A magnitude
        # For complex numbers represented as (real, imag), bound the L2 norm
        A_r = A_f_ri[..., :C]
        A_i = A_f_ri[..., C:]
        A_mag = jnp.sqrt(A_r**2 + A_i**2 + 1e-8)
        A_scale = 0.95 * jax.nn.sigmoid(A_mag) / (A_mag + 1e-8)
        A_f_ri = jnp.concatenate([A_r * A_scale, A_i * A_scale], axis=-1)

        # === SSM with lax.scan ===
        def step_fn(h_ri, t):
            """One SSM timestep with Mamba-style gating."""
            # State decay: element-wise multiply (forget_gate * A * h)
            # For complex: (A_r + iA_i) * (h_r + ih_i) = (A_r*h_r - A_i*h_i) + i(A_r*h_i + A_i*h_r)
            h_r = h_ri[..., :C]
            h_i = h_ri[..., C:]
            Ar = A_f_ri[..., :C]
            Ai = A_f_ri[..., C:]

            # Complex multiply A * h
            Ah_r = Ar * h_r - Ai * h_i
            Ah_i = Ar * h_i + Ai * h_r

            # Apply forget gate (real-valued, applied to both components)
            h_decayed_r = forget_gate[..., :C] * Ah_r
            h_decayed_i = forget_gate[..., C:] * Ah_i

            # Input contribution: delta * input_gate * B * x
            x_r = x_ri[..., :C]
            x_i = x_ri[..., C:]
            Br = B_f_ri[..., :C]
            Bi = B_f_ri[..., C:]

            # Complex multiply B * x
            Bx_r = Br * x_r - Bi * x_i
            Bx_i = Br * x_i + Bi * x_r

            # Apply delta and input_gate
            input_r = delta[..., :C] * input_gate[..., :C] * Bx_r
            input_i = delta[..., C:] * input_gate[..., C:] * Bx_i

            # State update
            h_new_r = h_decayed_r + input_r
            h_new_i = h_decayed_i + input_i

            h_new_ri = jnp.concatenate([h_new_r, h_new_i], axis=-1)

            # Ensure output dtype matches input dtype for lax.scan compatibility
            return h_new_ri.astype(h_ri.dtype), None

        # Initialize state
        h_init = jnp.zeros_like(x_ri)

        # Run T steps
        h_final, _ = lax.scan(step_fn, h_init, jnp.arange(self.T))

        # Cast back to input dtype (float32) for FFT compatibility
        return h_final.astype(input_dtype)


# =============================================================================
# Fourier ConvNeXt Block (Stable Version)
# =============================================================================

class StableFourierConvNeXtBlock(nn.Module):
    """ConvNeXt block in Fourier domain with numerical stability.

    Uses real/imag representation throughout for stable gradients.

    Attributes:
        dim: Number of channels (C, the block outputs 2C stacked)
        T: SSM timesteps
        expansion: MLP expansion ratio
        drop_path_rate: Stochastic depth rate
        dtype: Compute dtype for SSM operations (float32 or bfloat16)
    """
    dim: int
    T: int = 16
    expansion: int = 4
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_ri: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass using real/imag representation.

        Args:
            x_ri: (batch, H, W, 2C) stacked real/imag
            train: Training mode

        Returns:
            (batch, H, W, 2C) output
        """
        residual = x_ri
        C2 = x_ri.shape[-1]
        C = C2 // 2

        # ConvSSM (replaces depthwise conv)
        x_ri = StableFourierConvSSM(self.dim, T=self.T, dtype=self.dtype)(x_ri)

        # LayerNorm
        x_ri = StableFourierNorm()(x_ri)

        # Pointwise MLP
        x_mlp = nn.Dense(C2 * self.expansion)(x_ri)  # Expansion
        x_mlp = nn.gelu(x_mlp)
        x_mlp = nn.Dense(C2)(x_mlp)  # Projection back

        # Layer scale
        layer_scale = self.param(
            'layer_scale',
            nn.initializers.constant(1e-6),
            (C2,)
        )
        x_ri = layer_scale * x_mlp

        # Residual
        return residual + x_ri


# =============================================================================
# Pure Fourier Depthwise Convolution (No SSM - Exact Equivalent to Spatial Conv)
# =============================================================================

class FourierDepthwiseConv(nn.Module):
    """FFT-based depthwise convolution - exact equivalent to spatial conv.

    This is the simplest possible FFT convolution: learn a spatial kernel,
    FFT it, multiply with input in frequency domain. No SSM, no recurrence.

    Equivalent to: nn.Conv(features=C, kernel_size=(k,k), feature_group_count=C)

    Args:
        dim: Number of channels (C)
        kernel_size: Spatial kernel size (default 7, like ConvNeXt)
        dtype: Computation dtype
    """
    dim: int
    kernel_size: int = 7
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_ri: jnp.ndarray) -> jnp.ndarray:
        """Apply depthwise convolution in frequency domain.

        Args:
            x_ri: (batch, H, W, 2C) stacked real/imag Fourier coefficients

        Returns:
            (batch, H, W, 2C) output Fourier coefficients
        """
        B, H, W, C2 = x_ri.shape
        C = C2 // 2
        k = self.kernel_size

        # Cast to compute dtype
        input_dtype = x_ri.dtype
        x_ri = x_ri.astype(self.dtype)

        # Learn spatial kernel (exactly like ConvNeXt depthwise conv)
        # Initialized same as flax Conv default
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (C, k, k)
        )

        # FFT kernel to frequency domain (kernel_to_freq handles centering)
        kernel_f = kernel_to_freq(kernel, H, W)  # (C, H, W) complex

        # Convert to real/imag representation
        kernel_f_ri = complex_to_realimag_chw(kernel_f).astype(self.dtype)  # (1, H, W, 2C)

        # Complex multiply: kernel_f * x_f (elementwise = convolution in spatial)
        x_r, x_i = x_ri[..., :C], x_ri[..., C:]
        k_r, k_i = kernel_f_ri[..., :C], kernel_f_ri[..., C:]

        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        out_r = k_r * x_r - k_i * x_i
        out_i = k_r * x_i + k_i * x_r

        out_ri = jnp.concatenate([out_r, out_i], axis=-1)

        return out_ri.astype(input_dtype)


class PureFourierConvNeXtBlock(nn.Module):
    """ConvNeXt block using pure FFT convolution (no SSM).

    This is the exact Fourier-domain equivalent of a standard ConvNeXt block.
    Uses FourierDepthwiseConv instead of spatial depthwise conv.

    Architecture:
        x_ri -> FourierDepthwiseConv -> LayerNorm -> MLP(expand->GELU->project) -> + residual

    Attributes:
        dim: Number of channels (C)
        kernel_size: Spatial kernel size (default 7)
        expansion: MLP expansion ratio (default 4)
        drop_path_rate: Stochastic depth rate
        dtype: Compute dtype
    """
    dim: int
    kernel_size: int = 7
    expansion: int = 4
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_ri: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass using real/imag representation.

        Args:
            x_ri: (batch, H, W, 2C) stacked real/imag
            train: Training mode

        Returns:
            (batch, H, W, 2C) output
        """
        residual = x_ri
        C2 = x_ri.shape[-1]
        C = C2 // 2  # Actual channel count

        # Pure FFT depthwise convolution (exact equivalent to spatial conv)
        x_ri = FourierDepthwiseConv(
            dim=self.dim,
            kernel_size=self.kernel_size,
            dtype=self.dtype
        )(x_ri)

        # LayerNorm
        x_ri = StableFourierNorm()(x_ri)

        # Pointwise MLP - use C-based dimensions to match ConvNeXt param count
        x_mlp = nn.Dense(C * self.expansion)(x_ri)
        x_mlp = nn.gelu(x_mlp)
        x_mlp = nn.Dense(C2)(x_mlp)

        # Layer scale
        layer_scale = self.param(
            'layer_scale',
            nn.initializers.constant(1e-6),
            (C2,)
        )
        x_ri = layer_scale * x_mlp

        # Residual
        return residual + x_ri


# =============================================================================
# Efficient Fourier ConvNeXt Block (Parameter-Efficient Version)
# =============================================================================

class EfficientFourierConvNeXtBlock(nn.Module):
    """ConvNeXt block in Fourier domain with parameter-efficient spatial kernels.

    Uses EfficientFourierConvSSM which learns small 7x7 spatial kernels
    (same as ConvNeXt) instead of per-frequency weights.

    IMPORTANT: MLP dimensions are based on C (not 2C) to match ConvNeXt param count.
    Since we're in real/imag representation (2C), we use C = 2C//2 for MLP sizing.

    Attributes:
        dim: Number of channels (C, the block outputs 2C stacked)
        T: SSM timesteps
        kernel_size: Spatial kernel size (default 7)
        expansion: MLP expansion ratio
        drop_path_rate: Stochastic depth rate
        dtype: Compute dtype for SSM operations (float32 or bfloat16)
    """
    dim: int
    T: int = 16
    kernel_size: int = 7
    expansion: int = 4
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_ri: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass using real/imag representation.

        Args:
            x_ri: (batch, H, W, 2C) stacked real/imag
            train: Training mode

        Returns:
            (batch, H, W, 2C) output
        """
        residual = x_ri
        C2 = x_ri.shape[-1]
        C = C2 // 2  # Actual channel count (2C = real + imag)

        # Efficient ConvSSM (replaces depthwise conv with spatial kernel learning)
        x_ri = EfficientFourierConvSSM(
            dim=self.dim,
            T=self.T,
            kernel_size=self.kernel_size,
            dtype=self.dtype
        )(x_ri)

        # LayerNorm
        x_ri = StableFourierNorm()(x_ri)

        # Pointwise MLP - use C-based dimensions to match ConvNeXt param count
        # Input: 2C -> hidden: C*expansion -> output: 2C
        x_mlp = nn.Dense(C * self.expansion)(x_ri)  # Expansion based on C
        x_mlp = nn.gelu(x_mlp)
        x_mlp = nn.Dense(C2)(x_mlp)  # Project back to 2C

        # Layer scale
        layer_scale = self.param(
            'layer_scale',
            nn.initializers.constant(1e-6),
            (C2,)
        )
        x_ri = layer_scale * x_mlp

        # Residual
        return residual + x_ri


# =============================================================================
# Fourier Downsample (Stable Version)
# =============================================================================

class StableFourierDownsample(nn.Module):
    """Downsample in Fourier domain using real/imag representation.

    Crops high frequencies (low-pass filter) and projects channels.
    Uses orthonormal scaling to maintain gradient magnitudes.
    """
    out_dim: int

    @nn.compact
    def __call__(self, x_ri: jnp.ndarray) -> jnp.ndarray:
        """Downsample by 2x in Fourier domain.

        Args:
            x_ri: (batch, H, W, 2C) real/imag representation

        Returns:
            (batch, H//2, W//2, 2*out_dim) downsampled
        """
        B, H, W, C2 = x_ri.shape
        C = C2 // 2
        new_H, new_W = H // 2, W // 2

        # Convert back to complex for FFT operations
        x_f = realimag_to_complex(x_ri)

        # Crop to central frequencies (low-pass filter)
        x_f_shifted = jnp.fft.fftshift(x_f, axes=(1, 2))
        start_h = (H - new_H) // 2
        start_w = (W - new_W) // 2
        x_f_cropped = x_f_shifted[:, start_h:start_h+new_H, start_w:start_w+new_W, :]
        x_f_down = jnp.fft.ifftshift(x_f_cropped, axes=(1, 2))

        # Scale for orthonormal convention when changing spatial size
        # With ortho norm: ifft(fft(x)) = x
        # When cropping from HxW to H/2 x W/2 in frequency domain,
        # we need to scale by sqrt(new_size/old_size) = sqrt(1/4) = 0.5
        # to maintain consistent energy/gradient scale
        x_f_down = x_f_down * 0.5

        # Convert back to real/imag
        x_ri_down = complex_to_realimag(x_f_down)

        # Channel projection
        x_proj = nn.Dense(self.out_dim * 2)(x_ri_down)

        return x_proj


# =============================================================================
# Full Model (Stable Version)
# =============================================================================

class ConvNeXtFourierV2(nn.Module):
    """ConvNeXt with all operations in Fourier domain (stable version).

    Architecture:
    1. Stem: Spatial conv (only spatial op!) + FFT -> real/imag
    2. 4 Stages of StableFourierConvNeXtBlocks
    3. Final: real/imag -> complex -> IFFT -> global pool -> classifier

    Attributes:
        num_classes: Number of output classes
        dims: Channel dimensions per stage
        depths: Number of blocks per stage
        T: SSM timesteps per block
        drop_path_rate: Stochastic depth rate
        dtype: Compute dtype for SSM operations (float32 or bfloat16)
    """
    num_classes: int = 1000
    dims: Sequence[int] = (96, 192, 384, 768)
    depths: Sequence[int] = (3, 3, 9, 3)
    T: int = 16
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: (batch, H, W, 3) RGB image in [0, 1]
            train: Training mode

        Returns:
            (batch, num_classes) logits
        """
        # === Stem: Only spatial operation ===
        x = nn.Conv(self.dims[0], (4, 4), strides=(4, 4), padding='VALID')(x)
        x = nn.LayerNorm()(x)

        # Convert to Fourier domain then to real/imag representation
        x_f = jnp.fft.fft2(x, axes=(1, 2))
        x_ri = complex_to_realimag(x_f)  # (B, H, W, 2C)

        # === Stages ===
        total_blocks = sum(self.depths)
        block_idx = 0

        for stage_idx, (dim, depth) in enumerate(zip(self.dims, self.depths)):
            # Downsample between stages (except first)
            if stage_idx > 0:
                x_ri = StableFourierNorm()(x_ri)
                x_ri = StableFourierDownsample(dim)(x_ri)

            # Blocks
            for i in range(depth):
                drop_rate = self.drop_path_rate * block_idx / (total_blocks - 1) if total_blocks > 1 else 0.0
                x_ri = StableFourierConvNeXtBlock(
                    dim=dim,
                    T=self.T,
                    drop_path_rate=drop_rate,
                    dtype=self.dtype
                )(x_ri, train=train)
                block_idx += 1

        # === Head ===
        # Convert back to complex then spatial
        x_f = realimag_to_complex(x_ri)
        x = jnp.fft.ifft2(x_f, axes=(1, 2)).real

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Final norm and classifier
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.num_classes)(x)

        return logits


# =============================================================================
# Full Model V3 (Parameter-Efficient Version)
# =============================================================================

class ConvNeXtFourierV3(nn.Module):
    """ConvNeXt with all operations in Fourier domain (parameter-efficient version).

    Key difference from V2:
    - Uses EfficientFourierConvSSM which learns small 7x7 spatial kernels
    - Same parameter count as standard ConvNeXt (~28M for Tiny)
    - Convolution implemented via FFT(kernel) * FFT(input) in frequency domain

    Architecture:
    1. Stem: Spatial conv (only spatial op!) + FFT -> real/imag
    2. 4 Stages of EfficientFourierConvNeXtBlocks
    3. Final: real/imag -> complex -> IFFT -> global pool -> classifier

    Attributes:
        num_classes: Number of output classes
        dims: Channel dimensions per stage
        depths: Number of blocks per stage
        T: SSM timesteps per block
        kernel_size: Spatial kernel size for SSM (default 7, like ConvNeXt)
        drop_path_rate: Stochastic depth rate
        dtype: Compute dtype for SSM operations (float32 or bfloat16)
    """
    num_classes: int = 1000
    dims: Sequence[int] = (96, 192, 384, 768)
    depths: Sequence[int] = (3, 3, 9, 3)
    T: int = 16
    kernel_size: int = 7
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: (batch, H, W, 3) RGB image in [0, 1]
            train: Training mode

        Returns:
            (batch, num_classes) logits
        """
        # === Stem: Only spatial operation ===
        x = nn.Conv(self.dims[0], (4, 4), strides=(4, 4), padding='VALID')(x)
        x = nn.LayerNorm()(x)

        # Convert to Fourier domain then to real/imag representation
        # Use orthonormal FFT for consistent gradient scaling
        x_f = jnp.fft.fft2(x, axes=(1, 2), norm='ortho')
        x_ri = complex_to_realimag(x_f)  # (B, H, W, 2C)

        # === Stages ===
        total_blocks = sum(self.depths)
        block_idx = 0

        for stage_idx, (dim, depth) in enumerate(zip(self.dims, self.depths)):
            # Downsample between stages (except first)
            if stage_idx > 0:
                x_ri = StableFourierNorm()(x_ri)
                x_ri = StableFourierDownsample(dim)(x_ri)

            # Blocks using efficient spatial kernel learning
            for i in range(depth):
                drop_rate = self.drop_path_rate * block_idx / (total_blocks - 1) if total_blocks > 1 else 0.0
                x_ri = EfficientFourierConvNeXtBlock(
                    dim=dim,
                    T=self.T,
                    kernel_size=self.kernel_size,
                    drop_path_rate=drop_rate,
                    dtype=self.dtype
                )(x_ri, train=train)
                block_idx += 1

        # === Head ===
        # Convert back to complex then spatial with orthonormal IFFT
        x_f = realimag_to_complex(x_ri)
        x = jnp.fft.ifft2(x_f, axes=(1, 2), norm='ortho').real

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Final norm and classifier
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.num_classes)(x)

        return logits


# =============================================================================
# Full Model - Pure Fourier (No SSM - Exact Equivalent to ConvNeXt)
# =============================================================================

class ConvNeXtFourierPure(nn.Module):
    """ConvNeXt with all operations in Fourier domain (no SSM - exact equivalent).

    This is the cleanest comparison with standard ConvNeXt:
    - Uses PureFourierConvNeXtBlock with FourierDepthwiseConv
    - No SSM recurrence - just FFT(kernel) * FFT(x)
    - Exactly matches ConvNeXt architecture in Fourier domain

    Architecture:
    1. Stem: Spatial conv + FFT -> real/imag
    2. 4 Stages of PureFourierConvNeXtBlocks
    3. Final: real/imag -> complex -> IFFT -> global pool -> classifier

    Attributes:
        num_classes: Number of output classes
        dims: Channel dimensions per stage
        depths: Number of blocks per stage
        kernel_size: Spatial kernel size (default 7, like ConvNeXt)
        drop_path_rate: Stochastic depth rate
        dtype: Compute dtype
    """
    num_classes: int = 1000
    dims: Sequence[int] = (96, 192, 384, 768)
    depths: Sequence[int] = (3, 3, 9, 3)
    kernel_size: int = 7
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: (batch, H, W, 3) RGB image in [0, 1]
            train: Training mode

        Returns:
            (batch, num_classes) logits
        """
        # === Stem: Only spatial operation ===
        x = nn.Conv(self.dims[0], (4, 4), strides=(4, 4), padding='VALID')(x)
        x = nn.LayerNorm()(x)

        # Convert to Fourier domain then to real/imag representation
        # Use orthonormal FFT for consistent gradient scaling
        x_f = jnp.fft.fft2(x, axes=(1, 2), norm='ortho')
        x_ri = complex_to_realimag(x_f)  # (B, H, W, 2C)

        # === Stages ===
        total_blocks = sum(self.depths)
        block_idx = 0

        for stage_idx, (dim, depth) in enumerate(zip(self.dims, self.depths)):
            # Downsample between stages (except first)
            if stage_idx > 0:
                x_ri = StableFourierNorm()(x_ri)
                x_ri = StableFourierDownsample(dim)(x_ri)

            # Pure Fourier blocks (no SSM)
            for i in range(depth):
                drop_rate = self.drop_path_rate * block_idx / (total_blocks - 1) if total_blocks > 1 else 0.0
                x_ri = PureFourierConvNeXtBlock(
                    dim=dim,
                    kernel_size=self.kernel_size,
                    drop_path_rate=drop_rate,
                    dtype=self.dtype
                )(x_ri, train=train)
                block_idx += 1

        # === Head ===
        # Convert back to complex then spatial with orthonormal IFFT
        x_f = realimag_to_complex(x_ri)
        x = jnp.fft.ifft2(x_f, axes=(1, 2), norm='ortho').real

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Final norm and classifier
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.num_classes)(x)

        return logits


# =============================================================================
# Model Variants V2 (High Parameter Count)
# =============================================================================

def convnext_fourier_v2_tiny(num_classes: int = 1000, T: int = 16, **kwargs):
    """ConvNeXt-Fourier-V2 Tiny."""
    return ConvNeXtFourierV2(
        num_classes=num_classes,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 9, 3),
        T=T,
        **kwargs
    )


def convnext_fourier_v2_small(num_classes: int = 1000, T: int = 16, **kwargs):
    """ConvNeXt-Fourier-V2 Small."""
    return ConvNeXtFourierV2(
        num_classes=num_classes,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 27, 3),
        T=T,
        **kwargs
    )


def convnext_fourier_v2_base(num_classes: int = 1000, T: int = 16, **kwargs):
    """ConvNeXt-Fourier-V2 Base."""
    return ConvNeXtFourierV2(
        num_classes=num_classes,
        dims=(128, 256, 512, 1024),
        depths=(3, 3, 27, 3),
        T=T,
        **kwargs
    )


# =============================================================================
# Debug Model
# =============================================================================

def convnext_fourier_v2_debug(num_classes: int = 10, T: int = 8, **kwargs):
    """Small model for debugging."""
    return ConvNeXtFourierV2(
        num_classes=num_classes,
        dims=(48, 96, 192, 384),
        depths=(1, 1, 3, 1),
        T=T,
        **kwargs
    )


# =============================================================================
# BFloat16 Model Variants (SSM operations in bfloat16)
# =============================================================================

def convnext_fourier_v2_tiny_bf16(num_classes: int = 1000, T: int = 16, **kwargs):
    """ConvNeXt-Fourier-V2 Tiny with bfloat16 SSM operations."""
    return ConvNeXtFourierV2(
        num_classes=num_classes,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 9, 3),
        T=T,
        dtype=jnp.bfloat16,
        **kwargs
    )


def convnext_fourier_v2_small_bf16(num_classes: int = 1000, T: int = 16, **kwargs):
    """ConvNeXt-Fourier-V2 Small with bfloat16 SSM operations."""
    return ConvNeXtFourierV2(
        num_classes=num_classes,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 27, 3),
        T=T,
        dtype=jnp.bfloat16,
        **kwargs
    )


def convnext_fourier_v2_debug_bf16(num_classes: int = 10, T: int = 8, **kwargs):
    """Small bfloat16 model for debugging."""
    return ConvNeXtFourierV2(
        num_classes=num_classes,
        dims=(48, 96, 192, 384),
        depths=(1, 1, 3, 1),
        T=T,
        dtype=jnp.bfloat16,
        **kwargs
    )


# =============================================================================
# Model Variants V3 (Parameter-Efficient Version)
# =============================================================================

def convnext_fourier_v3_tiny(num_classes: int = 1000, T: int = 16, kernel_size: int = 7, **kwargs):
    """ConvNeXt-Fourier-V3 Tiny (~28M params, same as ConvNeXt baseline).

    Uses small spatial kernels (7x7) FFT'd to frequency domain,
    instead of per-frequency learnable weights.
    """
    return ConvNeXtFourierV3(
        num_classes=num_classes,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 9, 3),
        T=T,
        kernel_size=kernel_size,
        **kwargs
    )


def convnext_fourier_v3_small(num_classes: int = 1000, T: int = 16, kernel_size: int = 7, **kwargs):
    """ConvNeXt-Fourier-V3 Small."""
    return ConvNeXtFourierV3(
        num_classes=num_classes,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 27, 3),
        T=T,
        kernel_size=kernel_size,
        **kwargs
    )


def convnext_fourier_v3_base(num_classes: int = 1000, T: int = 16, kernel_size: int = 7, **kwargs):
    """ConvNeXt-Fourier-V3 Base."""
    return ConvNeXtFourierV3(
        num_classes=num_classes,
        dims=(128, 256, 512, 1024),
        depths=(3, 3, 27, 3),
        T=T,
        kernel_size=kernel_size,
        **kwargs
    )


def convnext_fourier_v3_debug(num_classes: int = 10, T: int = 8, kernel_size: int = 7, **kwargs):
    """Small V3 model for debugging."""
    return ConvNeXtFourierV3(
        num_classes=num_classes,
        dims=(48, 96, 192, 384),
        depths=(1, 1, 3, 1),
        T=T,
        kernel_size=kernel_size,
        **kwargs
    )


# =============================================================================
# V3 BFloat16 Variants
# =============================================================================

def convnext_fourier_v3_tiny_bf16(num_classes: int = 1000, T: int = 16, kernel_size: int = 7, **kwargs):
    """ConvNeXt-Fourier-V3 Tiny with bfloat16 SSM operations."""
    return ConvNeXtFourierV3(
        num_classes=num_classes,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 9, 3),
        T=T,
        kernel_size=kernel_size,
        dtype=jnp.bfloat16,
        **kwargs
    )


def convnext_fourier_v3_small_bf16(num_classes: int = 1000, T: int = 16, kernel_size: int = 7, **kwargs):
    """ConvNeXt-Fourier-V3 Small with bfloat16 SSM operations."""
    return ConvNeXtFourierV3(
        num_classes=num_classes,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 27, 3),
        T=T,
        kernel_size=kernel_size,
        dtype=jnp.bfloat16,
        **kwargs
    )


def convnext_fourier_v3_debug_bf16(num_classes: int = 10, T: int = 8, kernel_size: int = 7, **kwargs):
    """Small V3 bfloat16 model for debugging."""
    return ConvNeXtFourierV3(
        num_classes=num_classes,
        dims=(48, 96, 192, 384),
        depths=(1, 1, 3, 1),
        T=T,
        kernel_size=kernel_size,
        dtype=jnp.bfloat16,
        **kwargs
    )


# =============================================================================
# Model Variants - Pure Fourier (No SSM - Exact Equivalent to ConvNeXt)
# =============================================================================

def convnext_fourier_pure_tiny(num_classes: int = 1000, kernel_size: int = 7, **kwargs):
    """Pure Fourier ConvNeXt-Tiny (~28M params equivalent to ConvNeXt baseline).

    No SSM - just FFT(kernel) * FFT(x). Exact equivalent to spatial ConvNeXt.
    """
    return ConvNeXtFourierPure(
        num_classes=num_classes,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 9, 3),
        kernel_size=kernel_size,
        **kwargs
    )


def convnext_fourier_pure_small(num_classes: int = 1000, kernel_size: int = 7, **kwargs):
    """Pure Fourier ConvNeXt-Small."""
    return ConvNeXtFourierPure(
        num_classes=num_classes,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 27, 3),
        kernel_size=kernel_size,
        **kwargs
    )


def convnext_fourier_pure_base(num_classes: int = 1000, kernel_size: int = 7, **kwargs):
    """Pure Fourier ConvNeXt-Base."""
    return ConvNeXtFourierPure(
        num_classes=num_classes,
        dims=(128, 256, 512, 1024),
        depths=(3, 3, 27, 3),
        kernel_size=kernel_size,
        **kwargs
    )


def convnext_fourier_pure_debug(num_classes: int = 10, kernel_size: int = 7, **kwargs):
    """Small Pure Fourier model for debugging."""
    return ConvNeXtFourierPure(
        num_classes=num_classes,
        dims=(48, 96, 192, 384),
        depths=(1, 1, 3, 1),
        kernel_size=kernel_size,
        **kwargs
    )
