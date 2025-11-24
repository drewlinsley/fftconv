# ConvSSM Complexity Analysis

## Overview

This document explains why FFT convolution is essential for ConvSSMs (Convolutional State Space Models) and provides complexity comparisons with alternative approaches.

---

## Key Clarification: Spatial vs Spatiotemporal Convolution

**ConvSSM performs spatial convolution at each timestep**, NOT one big spatiotemporal convolution:

```
SSM recurrence:     h_t = A · h_{t-1} + B · x_t        (matrix multiply)
ConvSSM recurrence: h_t = A ★ h_{t-1} + B ★ x_t        (★ = spatial convolution)
```

This means:
- **2D ConvSSM** (for video): 2D spatial conv (H×W) applied at each of T timesteps
- **3D ConvSSM** (for volumetric): 3D spatial conv (D×H×W) applied at each of T timesteps

The time dimension T is handled by the SSM recurrence (parallelized via scan), NOT by the convolution.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| `T` | Sequence length (number of timesteps) |
| `H, W` | Spatial dimensions (height, width) |
| `D` | Depth (for 3D spatial data) |
| `N` | Total spatial size: H×W (2D) or D×H×W (3D) |
| `K` | Spatial kernel size (e.g., 3 for 3×3 conv) |
| `C` | Number of channels |

---

## Scenario 1: Video Processing (2D Spatial + Time)

Input shape: `(T, C, H, W)` where T=frames, N=H×W

### Training Complexity

| Model | Description | Training Time | Training Memory |
|-------|-------------|---------------|-----------------|
| **2D Conv (per frame)** | Independent 2D conv on each frame | O(T·N·K²·C²) | O(T·N·C) |
| **3D Conv (spatiotemporal)** | One K×K×K conv over T×H×W | O(T·N·K³·C²) | O(T·N·C) |
| **ConvRNN (ConvLSTM)** | Sequential 2D conv per timestep | O(T·N·K²·C²) ⚠️ | O(T·N·C) |
| **Transformer** | Attention over T×N tokens | O((T·N)²·C) ⚠️ | O((T·N)²) ⚠️ |
| **SSM (no spatial conv)** | Parallel scan, no spatial mixing | O(T·N·C·log T) | O(T·N·C) |
| **2D ConvSSM** | Parallel scan + 2D FFT conv | O(T·N·C·(log T + log N)) | O(T·N·C) |

⚠️ = Major bottleneck

### Inference Complexity

| Model | Inference Time (per step) | Inference Memory |
|-------|---------------------------|------------------|
| **2D Conv (per frame)** | O(N·K²·C²) | O(N·C) |
| **3D Conv** | O(T·N·K³·C²) - needs all frames | O(T·N·C) |
| **ConvRNN** | O(N·K²·C²) | O(N·C) ✓ |
| **Transformer** | O(T·N·C) - attends to all past | O(T·N·C) grows! |
| **SSM** | O(N·C) ✓ | O(N·C) ✓ |
| **2D ConvSSM** | O(N·C) ✓ | O(N·C) ✓ |

**Why ConvRNN training is slow**: Must compute T steps sequentially (backprop through time).

**Why ConvSSM is fast**: Parallel scan over T in FFT space, O(log T) depth.

---

## Scenario 2: Volumetric Sequences (3D Spatial + Time)

Input shape: `(T, C, D, H, W)` where T=timesteps, N=D×H×W

### Training Complexity

| Model | Description | Training Time | Training Memory |
|-------|-------------|---------------|-----------------|
| **3D Conv (per step)** | Independent 3D conv each timestep | O(T·N·K³·C²) | O(T·N·C) |
| **4D Conv (spatiotemporal)** | One K⁴ conv over T×D×H×W | O(T·N·K⁴·C²) | O(T·N·C) |
| **ConvRNN-3D** | Sequential 3D conv per timestep | O(T·N·K³·C²) ⚠️ | O(T·N·C) |
| **Transformer** | Attention over T×N tokens | O((T·N)²·C) ⚠️ | O((T·N)²) ⚠️ |
| **SSM (no spatial conv)** | Parallel scan, no spatial mixing | O(T·N·C·log T) | O(T·N·C) |
| **3D ConvSSM** | Parallel scan + 3D FFT conv | O(T·N·C·(log T + log N)) | O(T·N·C) |

### Inference Complexity

| Model | Inference Time (per step) | Inference Memory |
|-------|---------------------------|------------------|
| **3D Conv (per step)** | O(N·K³·C²) | O(N·C) |
| **ConvRNN-3D** | O(N·K³·C²) | O(N·C) ✓ |
| **Transformer** | O(T·N·C) | O(T·N·C) grows! |
| **SSM** | O(N·C) ✓ | O(N·C) ✓ |
| **3D ConvSSM** | O(N·C) ✓ | O(N·C) ✓ |

---

## Why FFT Convolution? (The Parallel Scan Problem)

ConvSSM uses **small kernels** (e.g., 3×3 or 3×3×3), NOT full-resolution kernels.

For a single convolution, direct conv is faster:
- **Direct conv (K=3)**: O(N·K²) = O(9N) ✓
- **FFT conv**: O(N·log N) ≈ O(16N) for 256×256

**So why use FFT?** Because of the **parallel scan over time**.

---

## The Parallel Scan Problem (Core ConvSSM Insight)

The SSM recurrence `h_t = A ★ h_{t-1} + B ★ x_t` can be parallelized using a scan.

**Problem in pixel space**: The effective kernel grows during the scan!

```
For 3×3 kernel A:
  A¹ = A           → 3×3 kernel
  A² = A ★ A       → 5×5 kernel
  A⁴ = A² ★ A²     → 9×9 kernel
  A⁸               → 17×17 kernel
  ...
  A^T              → O(T×K) × O(T×K) kernel!
```

After log(T) scan stages, kernel size explodes. Can't efficiently compute A^T ★ h₀.

**Solution in FFT space**: Kernel size stays constant!

```
FFT the 3×3 kernel (zero-padded to H×W):
  Â = FFT(A)       → N complex values (where N = H×W)
  Â² = Â ⊙ Â       → still N values (element-wise multiply!)
  Â⁴ = Â² ⊙ Â²     → still N values
  ...
  Â^T              → still N values!
```

**Key insight**: In FFT space, repeated convolution is just repeated element-wise multiplication. The representation size never grows.

---

## Complexity Breakdown

| Operation | Pixel Space | FFT Space |
|-----------|-------------|-----------|
| Single conv (K=3) | O(N·K²) = O(9N) ✓ faster | O(N·log N) |
| Compute A^T (T steps) | O(T·N·K_eff²) where K_eff grows! ❌ | O(N·log N · log T) ✓ |
| Parallel scan depth | O(T) sequential ❌ | O(log T) parallel ✓ |

The FFT overhead per step is worth it because:
1. Kernel size stays constant during scan
2. Enables O(log T) parallel depth instead of O(T) sequential

---

## Concrete Example: Video Processing

T=1000 frames, H=W=256 (N=65,536), C=512 channels

| Model | Training Time | Training Memory | Inference Memory |
|-------|---------------|-----------------|------------------|
| **ConvRNN (K=3)** | ~1000 sequential steps | ~134 GB | ~134 MB ✓ |
| **Transformer** | O((65M)²) ≈ impossible | ~17 TB ❌ | grows with T |
| **2D ConvSSM** | ~10 parallel stages | ~134 GB | ~134 MB ✓ |

---

## Receptive Field Expansion Over Time

One of the powerful properties of ConvSSM is that the effective receptive field grows with timesteps:

```
Timestep 1: 3×3 receptive field
Timestep 2: 5×5 receptive field
Timestep 3: 7×7 receptive field
...
Timestep T: (2T(K-1)+1) × (2T(K-1)+1) receptive field
```

This happens naturally through the recurrence without explicitly computing larger kernels.

---

## Summary: When to Use What

| Data Type | Spatial | Temporal | Best Model | Implementation |
|-----------|---------|----------|------------|----------------|
| Video | 2D (H×W) | Sequence (T) | **2D ConvSSM** | `FlashFFTConv2D` |
| 4D Medical | 3D (D×H×W) | Sequence (T) | **3D ConvSSM** | `FlashFFTConv3D` |
| Single image | 2D (H×W) | None | Standard Conv | - |
| Single volume | 3D (D×H×W) | None | Standard 3D Conv | - |

---

## What ConvSSM Enables

1. **Video Understanding**: Process long videos with spatial context at each frame
   - Weather prediction, climate modeling
   - Video generation and prediction
   - Action recognition with spatial reasoning

2. **Volumetric Sequences**: 3D spatial data evolving over time
   - Medical imaging (4D CT/MRI)
   - Fluid dynamics simulation
   - Molecular dynamics

3. **Efficient Autoregressive Generation**: O(1) inference per frame
   - Real-time video synthesis
   - Interactive 3D content generation
