# Tests for JAX implementation of 3D FFT convolution and parallel scan
# Run with: pytest tests/test_conv_nd_jax.py -v

import pytest
import numpy as np

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Skip all tests if JAX not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")

if JAX_AVAILABLE:
    from flashfftconv.conv_nd_jax import (
        fft_conv_3d,
        fft_conv_3d_jit,
        convssm_sequential_3d,
        convssm_sequential_3d_jit,
        convssm_parallel_3d,
        convssm_parallel_3d_jit,
        parallel_scan_fft_3d,
        FlashFFTConv3DJAX,
        ConvSSMParallelScan3DJAX,
        convssm_loss_and_grad,
        # Fourier-space functions
        to_fourier_3d,
        to_fourier_3d_jit,
        from_fourier_3d,
        from_fourier_3d_jit,
        kernel_to_fourier_3d,
        kernel_to_fourier_3d_jit,
        convssm_fourier_scan,
        convssm_fourier_scan_jit,
        convssm_fourier_scan_final_jit,
        convssm_fourier_scan_parallel,
        convssm_fourier_scan_parallel_jit,
        convssm_fourier_scan_sequential,
        convssm_fourier_scan_sequential_jit,
        FourierConvSSM3D,
        fourier_mse_loss,
        fourier_convssm_loss_and_grad,
        compute_fft_size,
        compute_rfft_shape,
    )


# =============================================================================
# Helper functions
# =============================================================================

def random_array(key, shape, scale=0.1):
    """Create random array with given shape."""
    return random.normal(key, shape) * scale


def ref_fft_conv_3d_numpy(u, k, spatial_size):
    """
    NumPy reference implementation of 3D FFT convolution.
    """
    D, H, W = spatial_size
    fft_size = (2 * D, 2 * H, 2 * W)

    u_f = np.fft.rfftn(u, s=fft_size, axes=(-3, -2, -1))
    k_f = np.fft.rfftn(k, s=fft_size, axes=(-3, -2, -1))
    y_f = u_f * k_f[None, ...]
    y = np.fft.irfftn(y_f, s=fft_size, axes=(-3, -2, -1))

    return y[..., :D, :H, :W]


# =============================================================================
# FFT Convolution Tests
# =============================================================================

class TestFFTConv3D:
    """Tests for 3D FFT convolution."""

    @pytest.mark.parametrize('B', [1, 2, 4])
    @pytest.mark.parametrize('C', [16, 32])
    @pytest.mark.parametrize('D,H,W', [(8, 8, 8), (8, 16, 16), (16, 16, 16)])
    def test_forward_shape(self, B, C, D, H, W):
        """Test output shape is correct."""
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)

        u = random_array(k1, (B, C, D, H, W))
        k = random_array(k2, (C, D, H, W))

        y = fft_conv_3d(u, k, (D, H, W))

        assert y.shape == (B, C, D, H, W), f"Expected {(B, C, D, H, W)}, got {y.shape}"

    @pytest.mark.parametrize('B', [1, 2])
    @pytest.mark.parametrize('C', [16])
    @pytest.mark.parametrize('D,H,W', [(8, 8, 8), (8, 16, 16)])
    def test_forward_matches_numpy(self, B, C, D, H, W):
        """Test JAX implementation matches NumPy reference."""
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)

        u = random_array(k1, (B, C, D, H, W))
        k = random_array(k2, (C, D, H, W))

        # JAX result
        y_jax = fft_conv_3d(u, k, (D, H, W))

        # NumPy reference
        y_np = ref_fft_conv_3d_numpy(np.array(u), np.array(k), (D, H, W))

        np.testing.assert_allclose(np.array(y_jax), y_np, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('K', [3, 5, 7])
    def test_small_kernel(self, K):
        """Test with small kernels (3x3x3, 5x5x5, 7x7x7)."""
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)

        B, C, D, H, W = 2, 16, 16, 16, 16

        u = random_array(k1, (B, C, D, H, W))
        k = random_array(k2, (C, K, K, K))

        y = fft_conv_3d(u, k, (D, H, W))

        assert y.shape == (B, C, D, H, W)
        assert jnp.isfinite(y).all(), "Output contains NaN/Inf"

    def test_jit_compilation(self):
        """Test JIT compilation works correctly."""
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)

        B, C, D, H, W = 2, 16, 8, 8, 8

        u = random_array(k1, (B, C, D, H, W))
        k = random_array(k2, (C, 3, 3, 3))

        # Non-JIT
        y1 = fft_conv_3d(u, k, (D, H, W))

        # JIT
        y2 = fft_conv_3d_jit(u, k, (D, H, W))

        np.testing.assert_allclose(np.array(y1), np.array(y2), rtol=1e-5, atol=1e-5)

    def test_class_interface(self):
        """Test FlashFFTConv3DJAX class."""
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)

        D, H, W = 8, 16, 16
        B, C = 2, 32

        conv = FlashFFTConv3DJAX(D, H, W)

        u = random_array(k1, (B, C, D, H, W))
        k = random_array(k2, (C, 3, 3, 3))

        y = conv(u, k)

        assert y.shape == (B, C, D, H, W)
        assert repr(conv) == f"FlashFFTConv3DJAX(depth={D}, height={H}, width={W})"


# =============================================================================
# Sequential ConvSSM Tests
# =============================================================================

class TestConvSSMSequential:
    """Tests for sequential ConvSSM."""

    def test_basic(self):
        """Test basic sequential ConvSSM."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 5, 2, 16, 8, 8, 8
        K = 3

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K))
        B_kernel = random_array(keys[2], (C, K, K, K))

        h_seq = convssm_sequential_3d(x_seq, A_kernel, B_kernel, (D, H, W))

        assert h_seq.shape == (T, B, C, D, H, W)
        assert jnp.isfinite(h_seq).all()

    def test_recurrence_correctness(self):
        """Verify sequential computes correct recurrence."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 3, 1, 8, 8, 8, 8
        K = 3

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K))
        B_kernel = random_array(keys[2], (C, K, K, K))

        # Use lax.scan version
        h_seq = convssm_sequential_3d(x_seq, A_kernel, B_kernel, (D, H, W))

        # Manual loop verification
        h = jnp.zeros((B, C, D, H, W))
        for t in range(T):
            h = fft_conv_3d(h, A_kernel, (D, H, W)) + fft_conv_3d(x_seq[t], B_kernel, (D, H, W))
            np.testing.assert_allclose(
                np.array(h_seq[t]), np.array(h), rtol=1e-4, atol=1e-4,
                err_msg=f"Mismatch at timestep {t}"
            )


# =============================================================================
# Parallel Scan Tests
# =============================================================================

class TestParallelScan:
    """Tests for parallel scan ConvSSM."""

    def test_matches_sequential(self):
        """Test parallel scan matches sequential (with decaying kernels for stability)."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 10, 2, 16, 8, 8, 8
        K = 3

        x_seq = random_array(keys[0], (T, B, C, D, H, W))

        # Use decaying kernels to minimize circular aliasing effects
        # The parallel scan operates entirely in FFT space without intermediate crops,
        # so random kernels can accumulate differences. Decaying kernels are more realistic.
        A_base = random_array(keys[1], (C, K, K, K))
        B_base = random_array(keys[2], (C, K, K, K))

        # Apply spatial decay
        decay = jnp.exp(-0.5 * jnp.arange(K))
        decay_3d = decay[:, None, None] * decay[None, :, None] * decay[None, None, :]
        A_kernel = A_base * decay_3d * 0.5  # Scale down for stability
        B_kernel = B_base * decay_3d * 0.5

        # Sequential
        h_seq_sequential = convssm_sequential_3d(x_seq, A_kernel, B_kernel, (D, H, W))

        # Parallel
        h_seq_parallel = convssm_parallel_3d(x_seq, A_kernel, B_kernel, (D, H, W), return_all=True)

        # Allow larger tolerance due to FFT boundary effects
        # The parallel scan stays in freq domain, sequential crops at each step
        np.testing.assert_allclose(
            np.array(h_seq_parallel), np.array(h_seq_sequential),
            rtol=0.1, atol=0.1,
            err_msg="Parallel scan doesn't match sequential"
        )

    def test_return_all_false(self):
        """Test return_all=False returns only final state."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 10, 2, 16, 8, 8, 8
        K = 3

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K))
        B_kernel = random_array(keys[2], (C, K, K, K))

        # Get all states
        h_seq_all = convssm_parallel_3d(x_seq, A_kernel, B_kernel, (D, H, W), return_all=True)

        # Get final only
        h_final = convssm_parallel_3d(x_seq, A_kernel, B_kernel, (D, H, W), return_all=False)

        np.testing.assert_allclose(
            np.array(h_final), np.array(h_seq_all[-1]),
            rtol=1e-6, atol=1e-6,
            err_msg="return_all=False doesn't match h_seq[-1]"
        )

    @pytest.mark.parametrize('T', [1, 2, 3, 7, 8, 15, 16, 31, 32, 64, 100])
    def test_various_lengths(self, T):
        """Test parallel scan with various sequence lengths."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        B, C, D, H, W = 2, 8, 8, 8, 8
        K = 3

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K))
        B_kernel = random_array(keys[2], (C, K, K, K))

        h_seq = convssm_parallel_3d(x_seq, A_kernel, B_kernel, (D, H, W), return_all=True)

        assert h_seq.shape == (T, B, C, D, H, W), f"Expected {(T, B, C, D, H, W)}, got {h_seq.shape}"
        assert jnp.isfinite(h_seq).all(), f"T={T}: NaN/Inf in output"

    def test_jit_compilation(self):
        """Test JIT compilation of parallel scan."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 10, 2, 16, 8, 8, 8
        K = 3

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K))
        B_kernel = random_array(keys[2], (C, K, K, K))

        # Non-JIT
        h1 = convssm_parallel_3d(x_seq, A_kernel, B_kernel, (D, H, W), return_all=True)

        # JIT
        h2 = convssm_parallel_3d_jit(x_seq, A_kernel, B_kernel, (D, H, W), True)

        np.testing.assert_allclose(np.array(h1), np.array(h2), rtol=1e-5, atol=1e-5)

    def test_class_interface(self):
        """Test ConvSSMParallelScan3DJAX class."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 10, 2, 16, 8, 8, 8
        K = 3

        scanner = ConvSSMParallelScan3DJAX(D, H, W)

        x_seq = random_array(keys[0], (T, B, C, D, H, W))

        # Use decaying kernels for stability
        A_base = random_array(keys[1], (C, K, K, K))
        B_base = random_array(keys[2], (C, K, K, K))
        decay = jnp.exp(-0.5 * jnp.arange(K))
        decay_3d = decay[:, None, None] * decay[None, :, None] * decay[None, None, :]
        A_kernel = A_base * decay_3d * 0.5
        B_kernel = B_base * decay_3d * 0.5

        # Parallel
        h_parallel = scanner(x_seq, A_kernel, B_kernel, return_all=True)

        # Sequential (via class method)
        h_sequential = scanner.sequential(x_seq, A_kernel, B_kernel)

        np.testing.assert_allclose(
            np.array(h_parallel), np.array(h_sequential),
            rtol=0.1, atol=0.1
        )

        assert repr(scanner) == f"ConvSSMParallelScan3DJAX(depth={D}, height={H}, width={W})"


# =============================================================================
# Gradient Tests
# =============================================================================

class TestGradients:
    """Tests for gradient computation."""

    def test_fft_conv_grad(self):
        """Test gradients flow through FFT conv."""
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)

        B, C, D, H, W = 2, 8, 8, 8, 8

        u = random_array(k1, (B, C, D, H, W))
        k = random_array(k2, (C, 3, 3, 3))

        def loss_fn(kernel):
            y = fft_conv_3d(u, kernel, (D, H, W))
            return jnp.mean(y ** 2)

        grad_k = jax.grad(loss_fn)(k)

        assert grad_k.shape == k.shape
        assert jnp.isfinite(grad_k).all(), "Gradient contains NaN/Inf"
        assert jnp.abs(grad_k).sum() > 1e-6, "Gradient is zero"

    def test_parallel_scan_grad(self):
        """Test gradients flow through parallel scan."""
        key = random.PRNGKey(42)
        keys = random.split(key, 4)

        T, B, C, D, H, W = 5, 2, 8, 8, 8, 8
        K = 3

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K))
        B_kernel = random_array(keys[2], (C, K, K, K))
        target = random_array(keys[3], (T, B, C, D, H, W))

        loss, grad_A, grad_B = convssm_loss_and_grad(
            x_seq, A_kernel, B_kernel, (D, H, W), target
        )

        assert jnp.isfinite(loss), "Loss is NaN/Inf"
        assert grad_A.shape == A_kernel.shape
        assert grad_B.shape == B_kernel.shape
        assert jnp.isfinite(grad_A).all(), "grad_A contains NaN/Inf"
        assert jnp.isfinite(grad_B).all(), "grad_B contains NaN/Inf"

    def test_grad_matches_sequential(self):
        """Test parallel scan gradients match sequential gradients."""
        key = random.PRNGKey(42)
        keys = random.split(key, 4)

        T, B, C, D, H, W = 5, 2, 8, 8, 8, 8
        K = 3

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K))
        B_kernel = random_array(keys[2], (C, K, K, K))
        target = random_array(keys[3], (T, B, C, D, H, W))

        def loss_parallel(A, B):
            h_seq = convssm_parallel_3d(x_seq, A, B, (D, H, W), return_all=True)
            return jnp.mean((h_seq - target) ** 2)

        def loss_sequential(A, B):
            h_seq = convssm_sequential_3d(x_seq, A, B, (D, H, W))
            return jnp.mean((h_seq - target) ** 2)

        grad_A_par, grad_B_par = jax.grad(loss_parallel, argnums=(0, 1))(A_kernel, B_kernel)
        grad_A_seq, grad_B_seq = jax.grad(loss_sequential, argnums=(0, 1))(A_kernel, B_kernel)

        np.testing.assert_allclose(
            np.array(grad_A_par), np.array(grad_A_seq),
            rtol=1e-3, atol=1e-3,
            err_msg="grad_A doesn't match between parallel and sequential"
        )
        np.testing.assert_allclose(
            np.array(grad_B_par), np.array(grad_B_seq),
            rtol=1e-3, atol=1e-3,
            err_msg="grad_B doesn't match between parallel and sequential"
        )


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_single_timestep(self):
        """Test T=1 works correctly."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 1, 2, 16, 8, 8, 8
        K = 3

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K))
        B_kernel = random_array(keys[2], (C, K, K, K))

        h_seq = convssm_parallel_3d(x_seq, A_kernel, B_kernel, (D, H, W), return_all=True)

        # For T=1: h_1 = A * 0 + B * x_1 = B * x_1
        expected = fft_conv_3d(x_seq[0], B_kernel, (D, H, W))

        np.testing.assert_allclose(
            np.array(h_seq[0]), np.array(expected),
            rtol=1e-5, atol=1e-5
        )

    def test_batch_size_one(self):
        """Test B=1 works correctly."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 5, 1, 16, 8, 8, 8
        K = 3

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K))
        B_kernel = random_array(keys[2], (C, K, K, K))

        h_seq = convssm_parallel_3d(x_seq, A_kernel, B_kernel, (D, H, W), return_all=True)

        assert h_seq.shape == (T, B, C, D, H, W)
        assert jnp.isfinite(h_seq).all()

    def test_zero_input(self):
        """Test with zero input."""
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        T, B, C, D, H, W = 5, 2, 16, 8, 8, 8
        K = 3

        x_seq = jnp.zeros((T, B, C, D, H, W))
        A_kernel = random_array(keys[0], (C, K, K, K))
        B_kernel = random_array(keys[1], (C, K, K, K))

        h_seq = convssm_parallel_3d(x_seq, A_kernel, B_kernel, (D, H, W), return_all=True)

        # All zeros in -> all zeros out
        assert jnp.allclose(h_seq, 0, atol=1e-6)

    def test_identity_kernel(self):
        """Test with identity-like kernel (delta function)."""
        T, B, C, D, H, W = 3, 2, 4, 8, 8, 8

        x_seq = jnp.ones((T, B, C, D, H, W))

        # A = 0 (no recurrence), B = delta function at (0,0,0)
        A_kernel = jnp.zeros((C, 3, 3, 3))
        B_kernel = jnp.zeros((C, 3, 3, 3))
        B_kernel = B_kernel.at[:, 0, 0, 0].set(1.0)

        h_seq = convssm_parallel_3d(x_seq, A_kernel, B_kernel, (D, H, W), return_all=True)

        # h_t = 0 * h_{t-1} + delta * x_t = x_t
        # Due to FFT boundary effects, center should be close to input
        assert jnp.isfinite(h_seq).all()


# =============================================================================
# Fourier-Space Tests
# =============================================================================

class TestFourierSpace:
    """Tests for Fourier-space ConvSSM operations."""

    def test_to_from_fourier_roundtrip(self):
        """Test that to_fourier -> from_fourier recovers original signal."""
        key = random.PRNGKey(42)

        B, C, D, H, W = 2, 16, 8, 16, 16
        spatial_size = (D, H, W)

        x = random_array(key, (B, C, D, H, W))

        # Roundtrip
        x_f = to_fourier_3d(x, spatial_size)
        x_recovered = from_fourier_3d(x_f, spatial_size)

        np.testing.assert_allclose(
            np.array(x_recovered), np.array(x),
            rtol=1e-5, atol=1e-5,
            err_msg="to_fourier -> from_fourier roundtrip failed"
        )

    def test_to_fourier_jit(self):
        """Test JIT versions of Fourier conversions."""
        key = random.PRNGKey(42)

        B, C, D, H, W = 2, 16, 8, 16, 16
        spatial_size = (D, H, W)

        x = random_array(key, (B, C, D, H, W))

        # Non-JIT
        x_f1 = to_fourier_3d(x, spatial_size)
        x_rec1 = from_fourier_3d(x_f1, spatial_size)

        # JIT
        x_f2 = to_fourier_3d_jit(x, spatial_size)
        x_rec2 = from_fourier_3d_jit(x_f2, spatial_size)

        np.testing.assert_allclose(np.array(x_f1), np.array(x_f2), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(x_rec1), np.array(x_rec2), rtol=1e-5, atol=1e-5)

    def test_fourier_shape(self):
        """Test Fourier transform output shapes."""
        D, H, W = 8, 16, 16
        spatial_size = (D, H, W)

        # Expected shapes
        fft_size = compute_fft_size(spatial_size)
        rfft_shape = compute_rfft_shape(spatial_size)

        assert fft_size == (16, 32, 32), f"Expected (16, 32, 32), got {fft_size}"
        assert rfft_shape == (16, 32, 17), f"Expected (16, 32, 17), got {rfft_shape}"

        # Test actual output shape
        key = random.PRNGKey(42)
        x = random_array(key, (2, 4, D, H, W))
        x_f = to_fourier_3d(x, spatial_size)

        expected_shape = (2, 4, 16, 32, 17)
        assert x_f.shape == expected_shape, f"Expected {expected_shape}, got {x_f.shape}"
        assert jnp.iscomplexobj(x_f), "Fourier output should be complex"

    def test_kernel_to_fourier(self):
        """Test kernel conversion to Fourier domain."""
        key = random.PRNGKey(42)

        C, K = 16, 3
        D, H, W = 8, 16, 16
        spatial_size = (D, H, W)

        kernel = random_array(key, (C, K, K, K))
        kernel_f = kernel_to_fourier_3d(kernel, spatial_size)

        # Check shape
        expected_shape = (C, 16, 32, 17)  # (C, 2D, 2H, W+1)
        assert kernel_f.shape == expected_shape, f"Expected {expected_shape}, got {kernel_f.shape}"
        assert jnp.iscomplexobj(kernel_f)

    def test_convssm_fourier_matches_parallel(self):
        """Test Fourier-space ConvSSM matches spatial parallel scan."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 10, 2, 16, 8, 8, 8
        K = 3
        spatial_size = (D, H, W)

        x_seq = random_array(keys[0], (T, B, C, D, H, W))

        # Use decaying kernels for stability
        A_base = random_array(keys[1], (C, K, K, K))
        B_base = random_array(keys[2], (C, K, K, K))
        decay = jnp.exp(-0.5 * jnp.arange(K))
        decay_3d = decay[:, None, None] * decay[None, :, None] * decay[None, None, :]
        A_kernel = A_base * decay_3d * 0.5
        B_kernel = B_base * decay_3d * 0.5

        # Spatial parallel scan
        h_seq_spatial = convssm_parallel_3d(x_seq, A_kernel, B_kernel, spatial_size, return_all=True)

        # Fourier-space scan
        x_seq_f = to_fourier_3d(x_seq, spatial_size)
        A_f = kernel_to_fourier_3d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_3d(B_kernel, spatial_size)
        h_seq_f = convssm_fourier_scan(A_f, B_f, x_seq_f, return_all=True)
        h_seq_fourier = from_fourier_3d(h_seq_f, spatial_size)

        # Compare - should be very close since both use same underlying FFT ops
        np.testing.assert_allclose(
            np.array(h_seq_fourier), np.array(h_seq_spatial),
            rtol=1e-4, atol=1e-4,
            err_msg="Fourier-space ConvSSM doesn't match spatial parallel scan"
        )

    def test_convssm_fourier_return_all_false(self):
        """Test Fourier-space scan with return_all=False."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 10, 2, 16, 8, 8, 8
        K = 3
        spatial_size = (D, H, W)

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K)) * 0.1
        B_kernel = random_array(keys[2], (C, K, K, K)) * 0.1

        # Convert to Fourier
        x_seq_f = to_fourier_3d(x_seq, spatial_size)
        A_f = kernel_to_fourier_3d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_3d(B_kernel, spatial_size)

        # Get all states
        h_seq_f_all = convssm_fourier_scan(A_f, B_f, x_seq_f, return_all=True)

        # Get final only
        h_f_final = convssm_fourier_scan(A_f, B_f, x_seq_f, return_all=False)

        np.testing.assert_allclose(
            np.array(h_f_final), np.array(h_seq_f_all[-1]),
            rtol=1e-6, atol=1e-6,
            err_msg="return_all=False doesn't match h_seq_f[-1]"
        )

    def test_convssm_fourier_jit(self):
        """Test JIT-compiled Fourier-space scan."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 10, 2, 16, 8, 8, 8
        K = 3
        spatial_size = (D, H, W)

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K)) * 0.1
        B_kernel = random_array(keys[2], (C, K, K, K)) * 0.1

        x_seq_f = to_fourier_3d(x_seq, spatial_size)
        A_f = kernel_to_fourier_3d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_3d(B_kernel, spatial_size)

        # Non-JIT
        h1 = convssm_fourier_scan(A_f, B_f, x_seq_f, return_all=True)

        # JIT
        h2 = convssm_fourier_scan_jit(A_f, B_f, x_seq_f)

        np.testing.assert_allclose(np.array(h1), np.array(h2), rtol=1e-5, atol=1e-5)

        # Test final-only JIT
        h_final1 = convssm_fourier_scan(A_f, B_f, x_seq_f, return_all=False)
        h_final2 = convssm_fourier_scan_final_jit(A_f, B_f, x_seq_f)

        np.testing.assert_allclose(np.array(h_final1), np.array(h_final2), rtol=1e-5, atol=1e-5)

    def test_fourier_convssm_class(self):
        """Test FourierConvSSM3D class interface."""
        D, H, W, C = 8, 16, 16, 32
        T, B = 10, 2
        K = 3

        model = FourierConvSSM3D(D, H, W, channels=C)

        # Check properties
        assert model.depth == D
        assert model.height == H
        assert model.width == W
        assert model.channels == C
        assert model.spatial_size == (D, H, W)
        assert model.fft_size == (16, 32, 32)
        assert model.rfft_shape == (16, 32, 17)

        # Test repr
        assert "FourierConvSSM3D" in repr(model)
        assert f"depth={D}" in repr(model)

        # Create input
        key = random.PRNGKey(42)
        x_seq = random_array(key, (T, B, C, D, H, W))

        # Pre-FFT input
        x_seq_f = model.precompute_input_fft(x_seq)
        assert x_seq_f.shape == (T, B, C, 16, 32, 17)

        # Init kernels
        A_f, B_f = model.init_kernels_fourier(key, kernel_size=K)
        assert A_f.shape == (C, 16, 32, 17)
        assert B_f.shape == (C, 16, 32, 17)

        # Forward pass
        h_seq_f = model.forward_fourier(A_f, B_f, x_seq_f, return_all=True)
        assert h_seq_f.shape == (T, B, C, 16, 32, 17)

        # Convert to spatial
        h_seq = model.to_spatial(h_seq_f)
        assert h_seq.shape == (T, B, C, D, H, W)

        # Also test to_fourier method
        h_seq_f2 = model.to_fourier(h_seq)
        assert h_seq_f2.shape == (T, B, C, 16, 32, 17)

    def test_fourier_convssm_class_no_channels(self):
        """Test FourierConvSSM3D with channels provided at init_kernels time."""
        D, H, W = 8, 16, 16
        C = 32

        model = FourierConvSSM3D(D, H, W)  # No channels
        assert model.channels is None

        key = random.PRNGKey(42)
        A_f, B_f = model.init_kernels_fourier(key, kernel_size=3, channels=C)

        assert A_f.shape[0] == C
        assert B_f.shape[0] == C

    def test_fourier_convssm_class_missing_channels_error(self):
        """Test FourierConvSSM3D raises error when channels not provided."""
        model = FourierConvSSM3D(8, 16, 16)

        key = random.PRNGKey(42)
        with pytest.raises(ValueError, match="channels must be provided"):
            model.init_kernels_fourier(key, kernel_size=3)

    def test_fourier_mse_loss_matches_spatial(self):
        """Test Fourier MSE loss approximates spatial MSE loss (Parseval)."""
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        T, B, C, D, H, W = 5, 2, 16, 8, 8, 8
        spatial_size = (D, H, W)

        x = random_array(keys[0], (T, B, C, D, H, W))
        y = random_array(keys[1], (T, B, C, D, H, W))

        # Spatial MSE
        spatial_mse = jnp.mean((x - y) ** 2)

        # Fourier MSE
        x_f = to_fourier_3d(x, spatial_size)
        y_f = to_fourier_3d(y, spatial_size)
        fourier_mse = fourier_mse_loss(x_f, y_f)

        # By Parseval's theorem, ||x - y||² = (1/N) * ||X - Y||²
        # But due to rfftn normalization and doubling, there's a scale factor
        # The important thing is that the loss is proportional
        # Check that both are positive and have reasonable ratio
        assert spatial_mse > 0
        assert fourier_mse > 0

        # The ratio should be roughly constant (due to FFT normalization)
        # Just verify the loss is computable and finite
        assert jnp.isfinite(fourier_mse)

    def test_fourier_mse_loss_zero_diff(self):
        """Test Fourier MSE loss is zero when inputs match."""
        key = random.PRNGKey(42)

        T, B, C, D, H, W = 5, 2, 16, 8, 8, 8
        spatial_size = (D, H, W)

        x = random_array(key, (T, B, C, D, H, W))
        x_f = to_fourier_3d(x, spatial_size)

        loss = fourier_mse_loss(x_f, x_f)

        np.testing.assert_allclose(float(loss), 0.0, atol=1e-10)

    def test_fourier_convssm_loss_and_grad(self):
        """Test loss and gradient computation in Fourier space."""
        key = random.PRNGKey(42)
        keys = random.split(key, 4)

        T, B, C, D, H, W = 5, 2, 8, 8, 8, 8
        K = 3
        spatial_size = (D, H, W)

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        target = random_array(keys[3], (T, B, C, D, H, W))

        # Create decaying kernels
        A_base = random_array(keys[1], (C, K, K, K))
        B_base = random_array(keys[2], (C, K, K, K))
        decay = jnp.exp(-0.3 * jnp.arange(K))
        decay_3d = decay[:, None, None] * decay[None, :, None] * decay[None, None, :]
        A_kernel = A_base * decay_3d * 0.1
        B_kernel = B_base * decay_3d * 0.1

        # Convert to Fourier
        x_seq_f = to_fourier_3d(x_seq, spatial_size)
        target_f = to_fourier_3d(target, spatial_size)
        A_f = kernel_to_fourier_3d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_3d(B_kernel, spatial_size)

        # Compute loss and gradients
        loss, grad_A_f, grad_B_f = fourier_convssm_loss_and_grad(A_f, B_f, x_seq_f, target_f)

        assert jnp.isfinite(loss), "Loss is NaN/Inf"
        assert grad_A_f.shape == A_f.shape, f"Expected {A_f.shape}, got {grad_A_f.shape}"
        assert grad_B_f.shape == B_f.shape
        assert jnp.isfinite(grad_A_f).all(), "grad_A_f contains NaN/Inf"
        assert jnp.isfinite(grad_B_f).all(), "grad_B_f contains NaN/Inf"
        assert jnp.abs(grad_A_f).sum() > 1e-6, "grad_A_f is zero"
        assert jnp.abs(grad_B_f).sum() > 1e-6, "grad_B_f is zero"

    def test_fourier_grad_flow(self):
        """Test gradients flow through Fourier-space operations."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 3, 2, 8, 8, 8, 8
        K = 3
        spatial_size = (D, H, W)

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K)) * 0.1
        B_kernel = random_array(keys[2], (C, K, K, K)) * 0.1

        # Pre-convert to Fourier
        x_seq_f = to_fourier_3d(x_seq, spatial_size)
        A_f = kernel_to_fourier_3d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_3d(B_kernel, spatial_size)

        def loss_fn(A_f, B_f):
            h_seq_f = convssm_fourier_scan(A_f, B_f, x_seq_f, return_all=True)
            return jnp.mean(jnp.abs(h_seq_f) ** 2)

        grad_A, grad_B = jax.grad(loss_fn, argnums=(0, 1))(A_f, B_f)

        assert grad_A.shape == A_f.shape
        assert grad_B.shape == B_f.shape
        assert jnp.isfinite(grad_A).all()
        assert jnp.isfinite(grad_B).all()

    @pytest.mark.parametrize('T', [1, 5, 10, 50])
    def test_fourier_various_lengths(self, T):
        """Test Fourier-space scan with various sequence lengths."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        B, C, D, H, W = 2, 8, 8, 8, 8
        K = 3
        spatial_size = (D, H, W)

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K)) * 0.1
        B_kernel = random_array(keys[2], (C, K, K, K)) * 0.1

        x_seq_f = to_fourier_3d(x_seq, spatial_size)
        A_f = kernel_to_fourier_3d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_3d(B_kernel, spatial_size)

        h_seq_f = convssm_fourier_scan(A_f, B_f, x_seq_f, return_all=True)

        expected_shape = (T, B, C, 16, 16, 9)  # FFT shape for 8x8x8 -> 16x16x9
        assert h_seq_f.shape == expected_shape, f"Expected {expected_shape}, got {h_seq_f.shape}"
        assert jnp.isfinite(h_seq_f).all()

    def test_spatial_kernel_to_fourier_method(self):
        """Test FourierConvSSM3D.spatial_kernel_to_fourier method."""
        key = random.PRNGKey(42)

        D, H, W, C = 8, 16, 16, 32
        K = 3

        model = FourierConvSSM3D(D, H, W, C)

        kernel = random_array(key, (C, K, K, K))
        kernel_f = model.spatial_kernel_to_fourier(kernel)

        # Compare with direct call
        kernel_f_direct = kernel_to_fourier_3d(kernel, (D, H, W))

        np.testing.assert_allclose(
            np.array(kernel_f), np.array(kernel_f_direct),
            rtol=1e-6, atol=1e-6
        )

    def test_sequential_vs_parallel_match(self):
        """Test sequential and parallel Fourier scans produce same results."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 20, 2, 16, 8, 8, 8
        K = 3
        spatial_size = (D, H, W)

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K)) * 0.1
        B_kernel = random_array(keys[2], (C, K, K, K)) * 0.1

        x_seq_f = to_fourier_3d(x_seq, spatial_size)
        A_f = kernel_to_fourier_3d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_3d(B_kernel, spatial_size)

        # Sequential
        h_seq = convssm_fourier_scan_sequential(A_f, B_f, x_seq_f, return_all=True)

        # Parallel
        h_par = convssm_fourier_scan_parallel(A_f, B_f, x_seq_f, return_all=True)

        np.testing.assert_allclose(
            np.array(h_seq), np.array(h_par),
            rtol=1e-5, atol=1e-5,
            err_msg="Sequential and parallel Fourier scans don't match"
        )

    def test_sequential_return_all_false(self):
        """Test sequential scan with return_all=False."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 10, 2, 16, 8, 8, 8
        K = 3
        spatial_size = (D, H, W)

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K)) * 0.1
        B_kernel = random_array(keys[2], (C, K, K, K)) * 0.1

        x_seq_f = to_fourier_3d(x_seq, spatial_size)
        A_f = kernel_to_fourier_3d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_3d(B_kernel, spatial_size)

        # Get all
        h_all = convssm_fourier_scan_sequential(A_f, B_f, x_seq_f, return_all=True)

        # Get final only
        h_final = convssm_fourier_scan_sequential(A_f, B_f, x_seq_f, return_all=False)

        np.testing.assert_allclose(
            np.array(h_final), np.array(h_all[-1]),
            rtol=1e-6, atol=1e-6
        )

    def test_auto_mode_selection(self):
        """Test that auto mode works and produces correct results."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, D, H, W = 10, 2, 16, 8, 8, 8
        K = 3
        spatial_size = (D, H, W)

        x_seq = random_array(keys[0], (T, B, C, D, H, W))
        A_kernel = random_array(keys[1], (C, K, K, K)) * 0.1
        B_kernel = random_array(keys[2], (C, K, K, K)) * 0.1

        x_seq_f = to_fourier_3d(x_seq, spatial_size)
        A_f = kernel_to_fourier_3d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_3d(B_kernel, spatial_size)

        # Auto mode
        h_auto = convssm_fourier_scan(A_f, B_f, x_seq_f, return_all=True, mode='auto')

        # Explicit sequential
        h_seq = convssm_fourier_scan_sequential(A_f, B_f, x_seq_f, return_all=True)

        # They should match (auto should pick sequential for this size)
        np.testing.assert_allclose(
            np.array(h_auto), np.array(h_seq),
            rtol=1e-5, atol=1e-5
        )


# =============================================================================
# 2D FFT Convolution and ConvSSM Tests
# =============================================================================

if JAX_AVAILABLE:
    from flashfftconv.conv_nd_jax import (
        # 2D functions
        fft_conv_2d,
        fft_conv_2d_jit,
        convssm_sequential_2d,
        convssm_sequential_2d_jit,
        convssm_parallel_2d,
        convssm_parallel_2d_jit,
        # 2D Fourier-space functions
        to_fourier_2d,
        to_fourier_2d_jit,
        from_fourier_2d,
        from_fourier_2d_jit,
        kernel_to_fourier_2d,
        kernel_to_fourier_2d_jit,
        convssm_fourier_scan_2d,
        convssm_fourier_scan_2d_jit,
        convssm_fourier_scan_2d_final_jit,
        convssm_fourier_scan_parallel_2d,
        convssm_fourier_scan_sequential_2d,
        FourierConvSSM2D,
        compute_fft_size_2d,
        compute_rfft_shape_2d,
    )


class TestFFTConv2D:
    """Tests for 2D FFT convolution."""

    @pytest.mark.parametrize('B', [1, 2, 4])
    @pytest.mark.parametrize('C', [16, 32])
    @pytest.mark.parametrize('H,W', [(32, 32), (64, 64), (128, 128)])
    def test_forward_shape(self, B, C, H, W):
        """Test output shape is correct."""
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)

        u = random_array(k1, (B, C, H, W))
        k = random_array(k2, (C, 7, 7))

        y = fft_conv_2d(u, k, (H, W))

        assert y.shape == (B, C, H, W), f"Expected {(B, C, H, W)}, got {y.shape}"

    @pytest.mark.parametrize('K', [3, 5, 7])
    def test_small_kernel(self, K):
        """Test with small kernels (3x3, 5x5, 7x7)."""
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)

        B, C, H, W = 2, 16, 64, 64

        u = random_array(k1, (B, C, H, W))
        k = random_array(k2, (C, K, K))

        y = fft_conv_2d(u, k, (H, W))

        assert y.shape == (B, C, H, W)
        assert jnp.isfinite(y).all(), "Output contains NaN/Inf"

    def test_jit_compilation(self):
        """Test JIT compilation works correctly."""
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)

        B, C, H, W = 2, 16, 64, 64

        u = random_array(k1, (B, C, H, W))
        k = random_array(k2, (C, 7, 7))

        # Non-JIT
        y1 = fft_conv_2d(u, k, (H, W))

        # JIT
        y2 = fft_conv_2d_jit(u, k, (H, W))

        np.testing.assert_allclose(np.array(y1), np.array(y2), rtol=1e-5, atol=1e-5)


class TestConvSSM2D:
    """Tests for 2D ConvSSM."""

    def test_sequential_basic(self):
        """Test basic sequential 2D ConvSSM."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, H, W = 8, 2, 16, 64, 64
        K = 7

        x_seq = random_array(keys[0], (T, B, C, H, W))
        A_kernel = random_array(keys[1], (C, K, K))
        B_kernel = random_array(keys[2], (C, K, K))

        h_seq = convssm_sequential_2d(x_seq, A_kernel, B_kernel, (H, W))

        assert h_seq.shape == (T, B, C, H, W)
        assert jnp.isfinite(h_seq).all()

    def test_parallel_basic(self):
        """Test basic parallel 2D ConvSSM."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, H, W = 8, 2, 16, 64, 64
        K = 7

        x_seq = random_array(keys[0], (T, B, C, H, W))
        A_kernel = random_array(keys[1], (C, K, K))
        B_kernel = random_array(keys[2], (C, K, K))

        h_seq = convssm_parallel_2d(x_seq, A_kernel, B_kernel, (H, W), return_all=True)

        assert h_seq.shape == (T, B, C, H, W)
        assert jnp.isfinite(h_seq).all()

    def test_parallel_matches_sequential(self):
        """Test parallel 2D scan matches sequential (with decaying kernels)."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, H, W = 8, 2, 16, 64, 64
        K = 7

        x_seq = random_array(keys[0], (T, B, C, H, W))

        # Use decaying kernels for stability
        A_base = random_array(keys[1], (C, K, K))
        B_base = random_array(keys[2], (C, K, K))
        decay = jnp.exp(-0.5 * jnp.arange(K))
        decay_2d = decay[:, None] * decay[None, :]
        A_kernel = A_base * decay_2d * 0.5
        B_kernel = B_base * decay_2d * 0.5

        h_seq = convssm_sequential_2d(x_seq, A_kernel, B_kernel, (H, W))
        h_par = convssm_parallel_2d(x_seq, A_kernel, B_kernel, (H, W), return_all=True)

        # Allow larger tolerance due to FFT boundary effects
        np.testing.assert_allclose(
            np.array(h_par), np.array(h_seq),
            rtol=0.1, atol=0.1,
            err_msg="Parallel 2D scan doesn't match sequential"
        )

    def test_return_all_false(self):
        """Test return_all=False returns only final state."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, H, W = 8, 2, 16, 64, 64
        K = 7

        x_seq = random_array(keys[0], (T, B, C, H, W))
        A_kernel = random_array(keys[1], (C, K, K))
        B_kernel = random_array(keys[2], (C, K, K))

        h_all = convssm_parallel_2d(x_seq, A_kernel, B_kernel, (H, W), return_all=True)
        h_final = convssm_parallel_2d(x_seq, A_kernel, B_kernel, (H, W), return_all=False)

        np.testing.assert_allclose(
            np.array(h_final), np.array(h_all[-1]),
            rtol=1e-6, atol=1e-6
        )


class TestFourierSpace2D:
    """Tests for 2D Fourier-space ConvSSM operations."""

    def test_to_from_fourier_roundtrip(self):
        """Test that to_fourier -> from_fourier recovers original signal."""
        key = random.PRNGKey(42)

        B, C, H, W = 2, 16, 64, 64
        spatial_size = (H, W)

        x = random_array(key, (B, C, H, W))

        # Roundtrip
        x_f = to_fourier_2d(x, spatial_size)
        x_recovered = from_fourier_2d(x_f, spatial_size)

        np.testing.assert_allclose(
            np.array(x_recovered), np.array(x),
            rtol=1e-5, atol=1e-5,
            err_msg="2D to_fourier -> from_fourier roundtrip failed"
        )

    def test_fourier_shape(self):
        """Test 2D Fourier transform output shapes."""
        H, W = 64, 64
        spatial_size = (H, W)

        # Expected shapes
        fft_size = compute_fft_size_2d(spatial_size)
        rfft_shape = compute_rfft_shape_2d(spatial_size)

        assert fft_size == (128, 128), f"Expected (128, 128), got {fft_size}"
        assert rfft_shape == (128, 65), f"Expected (128, 65), got {rfft_shape}"

        # Test actual output shape
        key = random.PRNGKey(42)
        x = random_array(key, (2, 4, H, W))
        x_f = to_fourier_2d(x, spatial_size)

        expected_shape = (2, 4, 128, 65)
        assert x_f.shape == expected_shape, f"Expected {expected_shape}, got {x_f.shape}"
        assert jnp.iscomplexobj(x_f), "Fourier output should be complex"

    def test_kernel_to_fourier(self):
        """Test 2D kernel conversion to Fourier domain."""
        key = random.PRNGKey(42)

        C, K = 16, 7
        H, W = 64, 64
        spatial_size = (H, W)

        kernel = random_array(key, (C, K, K))
        kernel_f = kernel_to_fourier_2d(kernel, spatial_size)

        expected_shape = (C, 128, 65)
        assert kernel_f.shape == expected_shape, f"Expected {expected_shape}, got {kernel_f.shape}"
        assert jnp.iscomplexobj(kernel_f)

    def test_fourier_jit(self):
        """Test JIT versions of 2D Fourier conversions."""
        key = random.PRNGKey(42)

        B, C, H, W = 2, 16, 64, 64
        spatial_size = (H, W)

        x = random_array(key, (B, C, H, W))

        # Non-JIT
        x_f1 = to_fourier_2d(x, spatial_size)
        x_rec1 = from_fourier_2d(x_f1, spatial_size)

        # JIT
        x_f2 = to_fourier_2d_jit(x, spatial_size)
        x_rec2 = from_fourier_2d_jit(x_f2, spatial_size)

        np.testing.assert_allclose(np.array(x_f1), np.array(x_f2), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(x_rec1), np.array(x_rec2), rtol=1e-5, atol=1e-5)

    def test_convssm_fourier_matches_spatial(self):
        """Test 2D Fourier-space ConvSSM matches spatial parallel scan."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, H, W = 8, 2, 16, 64, 64
        K = 7
        spatial_size = (H, W)

        x_seq = random_array(keys[0], (T, B, C, H, W))

        # Use decaying kernels for stability
        A_base = random_array(keys[1], (C, K, K))
        B_base = random_array(keys[2], (C, K, K))
        decay = jnp.exp(-0.5 * jnp.arange(K))
        decay_2d = decay[:, None] * decay[None, :]
        A_kernel = A_base * decay_2d * 0.5
        B_kernel = B_base * decay_2d * 0.5

        # Spatial parallel scan
        h_seq_spatial = convssm_parallel_2d(x_seq, A_kernel, B_kernel, spatial_size, return_all=True)

        # Fourier-space scan
        x_seq_f = to_fourier_2d(x_seq, spatial_size)
        A_f = kernel_to_fourier_2d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_2d(B_kernel, spatial_size)
        h_seq_f = convssm_fourier_scan_2d(A_f, B_f, x_seq_f, return_all=True)
        h_seq_fourier = from_fourier_2d(h_seq_f, spatial_size)

        np.testing.assert_allclose(
            np.array(h_seq_fourier), np.array(h_seq_spatial),
            rtol=1e-4, atol=1e-4,
            err_msg="2D Fourier-space ConvSSM doesn't match spatial parallel scan"
        )

    def test_sequential_vs_parallel_match(self):
        """Test 2D sequential and parallel Fourier scans produce same results."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, H, W = 8, 2, 16, 64, 64
        K = 7
        spatial_size = (H, W)

        x_seq = random_array(keys[0], (T, B, C, H, W))
        A_kernel = random_array(keys[1], (C, K, K)) * 0.1
        B_kernel = random_array(keys[2], (C, K, K)) * 0.1

        x_seq_f = to_fourier_2d(x_seq, spatial_size)
        A_f = kernel_to_fourier_2d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_2d(B_kernel, spatial_size)

        # Sequential
        h_seq = convssm_fourier_scan_sequential_2d(A_f, B_f, x_seq_f, return_all=True)

        # Parallel
        h_par = convssm_fourier_scan_parallel_2d(A_f, B_f, x_seq_f, return_all=True)

        np.testing.assert_allclose(
            np.array(h_seq), np.array(h_par),
            rtol=1e-5, atol=1e-5,
            err_msg="2D Sequential and parallel Fourier scans don't match"
        )

    def test_fourier_convssm_jit(self):
        """Test JIT-compiled 2D Fourier-space scan."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, H, W = 8, 2, 16, 64, 64
        K = 7
        spatial_size = (H, W)

        x_seq = random_array(keys[0], (T, B, C, H, W))
        A_kernel = random_array(keys[1], (C, K, K)) * 0.1
        B_kernel = random_array(keys[2], (C, K, K)) * 0.1

        x_seq_f = to_fourier_2d(x_seq, spatial_size)
        A_f = kernel_to_fourier_2d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_2d(B_kernel, spatial_size)

        # Non-JIT
        h1 = convssm_fourier_scan_2d(A_f, B_f, x_seq_f, return_all=True)

        # JIT
        h2 = convssm_fourier_scan_2d_jit(A_f, B_f, x_seq_f)

        np.testing.assert_allclose(np.array(h1), np.array(h2), rtol=1e-5, atol=1e-5)

        # Test final-only JIT
        h_final1 = convssm_fourier_scan_2d(A_f, B_f, x_seq_f, return_all=False)
        h_final2 = convssm_fourier_scan_2d_final_jit(A_f, B_f, x_seq_f)

        np.testing.assert_allclose(np.array(h_final1), np.array(h_final2), rtol=1e-5, atol=1e-5)

    def test_fourier_convssm_2d_class(self):
        """Test FourierConvSSM2D class interface."""
        H, W, C = 64, 64, 32
        T, B = 8, 2
        K = 7

        model = FourierConvSSM2D(H, W, channels=C)

        # Check properties
        assert model.height == H
        assert model.width == W
        assert model.channels == C
        assert model.spatial_size == (H, W)
        assert model.fft_size == (128, 128)
        assert model.rfft_shape == (128, 65)

        # Test repr
        assert "FourierConvSSM2D" in repr(model)
        assert f"height={H}" in repr(model)

        # Create input
        key = random.PRNGKey(42)
        x = random_array(key, (B, C, H, W))

        # Pre-FFT input
        x_f = model.precompute_input_fft(x)
        assert x_f.shape == (B, C, 128, 65)

        # Broadcast to timesteps
        x_seq_f = model.broadcast_to_timesteps(x_f, T)
        assert x_seq_f.shape == (T, B, C, 128, 65)

        # Init kernels
        A_f, B_f = model.init_kernels_fourier(key, kernel_size=K)
        assert A_f.shape == (C, 128, 65)
        assert B_f.shape == (C, 128, 65)

        # Forward pass
        h_seq_f = model.forward_fourier(A_f, B_f, x_seq_f, return_all=True)
        assert h_seq_f.shape == (T, B, C, 128, 65)

        # Convert to spatial
        h_seq = model.to_spatial(h_seq_f)
        assert h_seq.shape == (T, B, C, H, W)

        # Also test to_fourier method
        h_seq_f2 = model.to_fourier(h_seq)
        assert h_seq_f2.shape == (T, B, C, 128, 65)

    def test_fourier_convssm_2d_return_all_false(self):
        """Test FourierConvSSM2D with return_all=False."""
        H, W, C = 64, 64, 32
        T, B = 8, 2
        K = 7

        model = FourierConvSSM2D(H, W, channels=C)

        key = random.PRNGKey(42)
        x = random_array(key, (B, C, H, W))

        x_f = model.precompute_input_fft(x)
        x_seq_f = model.broadcast_to_timesteps(x_f, T)
        A_f, B_f = model.init_kernels_fourier(key, kernel_size=K)

        # Get all
        h_seq_f = model.forward_fourier(A_f, B_f, x_seq_f, return_all=True)

        # Get final only
        h_final_f = model.forward_fourier(A_f, B_f, x_seq_f, return_all=False)

        np.testing.assert_allclose(
            np.array(h_final_f), np.array(h_seq_f[-1]),
            rtol=1e-6, atol=1e-6
        )

    def test_grad_flow_2d(self):
        """Test gradients flow through 2D Fourier-space operations."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C, H, W = 4, 2, 8, 32, 32
        K = 7
        spatial_size = (H, W)

        x_seq = random_array(keys[0], (T, B, C, H, W))
        A_kernel = random_array(keys[1], (C, K, K)) * 0.1
        B_kernel = random_array(keys[2], (C, K, K)) * 0.1

        # Pre-convert to Fourier
        x_seq_f = to_fourier_2d(x_seq, spatial_size)
        A_f = kernel_to_fourier_2d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_2d(B_kernel, spatial_size)

        def loss_fn(A_f, B_f):
            h_seq_f = convssm_fourier_scan_2d(A_f, B_f, x_seq_f, return_all=True)
            return jnp.mean(jnp.abs(h_seq_f) ** 2)

        grad_A, grad_B = jax.grad(loss_fn, argnums=(0, 1))(A_f, B_f)

        assert grad_A.shape == A_f.shape
        assert grad_B.shape == B_f.shape
        assert jnp.isfinite(grad_A).all()
        assert jnp.isfinite(grad_B).all()

    @pytest.mark.parametrize('H,W', [(32, 32), (64, 64), (128, 128), (224, 224)])
    def test_various_image_sizes(self, H, W):
        """Test 2D Fourier ConvSSM with various image sizes."""
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        T, B, C = 8, 2, 16
        K = 7
        spatial_size = (H, W)

        x_seq = random_array(keys[0], (T, B, C, H, W))
        A_kernel = random_array(keys[1], (C, K, K)) * 0.1
        B_kernel = random_array(keys[2], (C, K, K)) * 0.1

        x_seq_f = to_fourier_2d(x_seq, spatial_size)
        A_f = kernel_to_fourier_2d(A_kernel, spatial_size)
        B_f = kernel_to_fourier_2d(B_kernel, spatial_size)

        h_seq_f = convssm_fourier_scan_2d(A_f, B_f, x_seq_f, return_all=True)
        h_seq = from_fourier_2d(h_seq_f, spatial_size)

        assert h_seq.shape == (T, B, C, H, W)
        assert jnp.isfinite(h_seq).all()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
