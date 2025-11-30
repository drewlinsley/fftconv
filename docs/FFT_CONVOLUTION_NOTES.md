# FFT Convolution Implementation Notes

## The Convolution Theorem

Spatial convolution can be computed in the frequency domain:
```
conv(x, kernel) = IFFT(FFT(x) * FFT(kernel))
```

However, getting this right requires careful attention to:
1. Kernel placement (centering)
2. FFT normalization
3. Complex number handling

## Critical Implementation Details

### 1. Kernel Placement for FFT Convolution

**The Problem**: For FFT convolution to match spatial convolution with "SAME" padding,
the kernel must be placed so that its center is at position (0, 0) with wrap-around.

**WRONG Approach** (what we had before):
```python
# DON'T DO THIS - loses wrap-around elements!
kernel_rolled = jnp.roll(kernel, -center, axis=1)
kernel_rolled = jnp.roll(kernel_rolled, -center, axis=2)
padded = jnp.zeros((C, H, W))
padded = padded.at[:, :k, :k].set(kernel_rolled)  # BUG: loses wrapped elements!
```

**CORRECT Approach** - scatter elements to wrapped positions:
```python
def kernel_to_freq(kernel, H, W):
    """FFT a small spatial kernel to frequency domain.

    Args:
        kernel: (C, k, k) spatial kernel
        H, W: target spatial dimensions

    Returns:
        (C, H, W) complex frequency representation
    """
    C, k, _ = kernel.shape
    center = k // 2

    # Create target position indices with wrapping
    i_idx = jnp.arange(k)
    j_idx = jnp.arange(k)
    target_i = (i_idx - center) % H  # Wrapped positions
    target_j = (j_idx - center) % W

    # Create meshgrid for 2D indexing
    ti, tj = jnp.meshgrid(target_i, target_j, indexing='ij')

    # Scatter kernel values to correct wrapped positions
    padded = jnp.zeros((C, H, W), dtype=kernel.dtype)
    padded = padded.at[:, ti, tj].set(kernel)

    # FFT (NO ortho normalization on kernel!)
    return jnp.fft.fft2(padded, axes=(-2, -1))
```

**Why this matters**: For a 7x7 kernel on 56x56 input:
- Kernel center (3,3) → position (0,0) ✓
- Kernel top-left (0,0) → position (-3,-3) = (53,53) ✓
- With the wrong approach, elements that should be at (53,53) end up at (4,4) - completely wrong!

### 2. FFT Normalization

**The Issue**: Using `norm='ortho'` affects scaling:
- `FFT_ortho(x) = FFT(x) / sqrt(N)` where N = H * W
- `IFFT_ortho(x) = IFFT(x) * sqrt(N)`

**Correct Configuration for Convolution**:
```python
# Input: use ortho (for gradient stability)
x_f = jnp.fft.fft2(x, axes=(1, 2), norm='ortho')

# Kernel: NO ortho
kernel_f = kernel_to_freq(kernel, H, W)  # Uses standard FFT inside

# Multiply in frequency domain
output_f = x_f * kernel_f  # (elementwise per channel)

# Output: use ortho
output = jnp.fft.ifft2(output_f, axes=(1, 2), norm='ortho').real
```

**Math verification**:
- x_f = FFT(x) / sqrt(N)
- kernel_f = FFT(kernel)
- product = FFT(x) * FFT(kernel) / sqrt(N)
- IFFT_ortho(product) = product * sqrt(N) / N = FFT(x) * FFT(kernel) / N = conv(x, kernel) ✓

### 3. Complex Number Handling

For numerical stability in neural networks, store complex numbers as stacked real/imag:
```python
def complex_to_realimag(x_f):
    """(B, H, W, C) complex → (B, H, W, 2C) real"""
    return jnp.concatenate([x_f.real, x_f.imag], axis=-1)

def realimag_to_complex(x_ri):
    """(B, H, W, 2C) real → (B, H, W, C) complex"""
    C = x_ri.shape[-1] // 2
    return x_ri[..., :C] + 1j * x_ri[..., C:]

def complex_mul_realimag(a_ri, b_ri):
    """Complex multiply using real/imag representation."""
    C = a_ri.shape[-1] // 2
    a_r, a_i = a_ri[..., :C], a_ri[..., C:]
    b_r, b_i = b_ri[..., :C], b_ri[..., C:]

    out_r = a_r * b_r - a_i * b_i
    out_i = a_r * b_i + a_i * b_r
    return jnp.concatenate([out_r, out_i], axis=-1)
```

## Complete Pipeline Example

```python
def fft_depthwise_conv(x, kernel):
    """FFT-based depthwise convolution matching spatial 'SAME' padding.

    Args:
        x: (B, H, W, C) spatial input
        kernel: (C, k, k) depthwise kernel

    Returns:
        (B, H, W, C) convolution output
    """
    B, H, W, C = x.shape

    # 1. FFT input with ortho normalization
    x_f = jnp.fft.fft2(x, axes=(1, 2), norm='ortho')
    x_ri = complex_to_realimag(x_f)  # (B, H, W, 2C)

    # 2. FFT kernel (NO ortho) with proper placement
    kernel_f = kernel_to_freq(kernel, H, W)  # (C, H, W)
    kernel_f_ri = complex_to_realimag_chw(kernel_f)  # (1, H, W, 2C)

    # 3. Complex multiply (= convolution in spatial domain)
    out_f_ri = complex_mul_realimag(kernel_f_ri, x_ri)

    # 4. IFFT with ortho normalization
    out_f = realimag_to_complex(out_f_ri)
    output = jnp.fft.ifft2(out_f, axes=(1, 2), norm='ortho').real

    return output
```

## Verification Test

To verify your FFT convolution is correct:
```python
import numpy as np
from scipy import ndimage

# Test data
x = np.random.randn(1, 16, 16, 4).astype(np.float32)
kernel = np.random.randn(4, 7, 7).astype(np.float32)

# Scipy reference (per-channel depthwise)
out_scipy = np.zeros_like(x)
for c in range(4):
    out_scipy[0, :, :, c] = ndimage.convolve(x[0, :, :, c], kernel[c], mode='wrap')

# FFT convolution
out_fft = fft_depthwise_conv(jnp.array(x), jnp.array(kernel))

# Should match!
diff = np.abs(out_scipy - np.array(out_fft))
print(f"Max diff: {diff.max():.10f}")  # Should be < 1e-5
print(f"Match: {diff.max() < 1e-4}")
```

## Common Bugs

1. **Roll then copy to [:k, :k]**: Loses wrap-around elements
2. **Using ortho on kernel FFT**: Causes scaling mismatch
3. **Forgetting to take .real after IFFT**: Output should be real
4. **Wrong axis ordering**: FFT axes should match spatial dimensions (1, 2)
5. **Flipping kernel**: Not needed for our use case (learned kernels)

## Files

- Main implementation: `flashfftconv/convnext_fourier_v2.py`
- Test verification: `tests/test_fft_convolution.py`
