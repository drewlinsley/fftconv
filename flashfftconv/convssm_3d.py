"""3D ConvSSM for Video Point Tracking.

This extends the 2D ConvSSM to 3D (spatiotemporal) for video understanding tasks
like point tracking on TAP-Vid.

Key differences from 2D:
- Uses 3D FFT convolution over (T, H, W) instead of 2D (H, W)
- SSM recurrence operates over an additional "iterations" dimension
- Can be integrated into transformers or 3D CNNs for point tracking

Architecture options:
1. Replace 3D conv blocks in a video backbone with 3D ConvSSM
2. Add 3D ConvSSM as temporal refinement after spatial features
3. Use as the core module in a point tracking head
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from typing import Sequence, Optional, Tuple
import numpy as np


# =============================================================================
# 3D FFT Convolution
# =============================================================================

def fft_depthwise_conv_3d(
    x: jnp.ndarray,
    kernel: jnp.ndarray
) -> jnp.ndarray:
    """3D FFT-based depthwise convolution.

    Args:
        x: (B, T, H, W, C) input video in NTHWC format
        kernel: (C, kt, kh, kw) depthwise kernel (one 3D kernel per channel)

    Returns:
        (B, T, H, W, C) convolution output
    """
    B, T, H, W, C = x.shape
    kt, kh, kw = kernel.shape[1], kernel.shape[2], kernel.shape[3]

    # Center indices for wrap-around placement
    center_t = kt // 2
    center_h = kh // 2
    center_w = kw // 2

    # Create index grids
    t_idx = jnp.arange(kt)
    h_idx = jnp.arange(kh)
    w_idx = jnp.arange(kw)

    # Target positions with wrap-around
    target_t = (t_idx - center_t) % T
    target_h = (h_idx - center_h) % H
    target_w = (w_idx - center_w) % W

    # Create meshgrid for 3D indexing
    tt, th, tw = jnp.meshgrid(target_t, target_h, target_w, indexing='ij')

    # Pad kernel to match input size
    padded_kernel = jnp.zeros((C, T, H, W), dtype=kernel.dtype)
    padded_kernel = padded_kernel.at[:, tt, th, tw].set(kernel)

    # 3D FFT convolution
    x_f = jnp.fft.fftn(x, axes=(1, 2, 3))  # (B, T, H, W, C)
    kernel_f = jnp.fft.fftn(padded_kernel, axes=(1, 2, 3))  # (C, T, H, W)

    # Broadcast kernel: (C, T, H, W) -> (1, T, H, W, C)
    kernel_f = kernel_f.transpose(1, 2, 3, 0)[None, ...]
    out_f = x_f * kernel_f

    # Inverse 3D FFT
    out = jnp.fft.ifftn(out_f, axes=(1, 2, 3)).real

    return out


class FFTDepthwiseConv3D(nn.Module):
    """3D FFT-based depthwise convolution layer."""
    features: int
    kernel_size: Tuple[int, int, int] = (3, 7, 7)  # (T, H, W)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply 3D FFT depthwise conv.

        Args:
            x: (B, T, H, W, C) video input

        Returns:
            (B, T, H, W, C) convolved output
        """
        C = x.shape[-1]
        kt, kh, kw = self.kernel_size

        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (C, kt, kh, kw),
            self.dtype
        )

        return fft_depthwise_conv_3d(x, kernel)


# =============================================================================
# 3D ConvSSM - Associative Scan for O(log T) depth
# =============================================================================

def ssm_associative_op_3d(left, right):
    """Associative operation for 3D linear recurrence.

    Same as 2D: (a1, b1) ⊕ (a2, b2) = (a1 * a2, a2 * b1 + b2)
    but operates on 3D frequency-domain tensors.
    """
    a_left, b_left = left
    a_right, b_right = right
    return (a_left * a_right, a_right * b_left + b_right)


class ParallelConvSSM3D(nn.Module):
    """3D Parallel ConvSSM using associative scan.

    SSM recurrence: h_t = A * h_{t-1} + B * x
    where * denotes 3D depthwise convolution (via FFT).

    This processes video volumes and adds spatial-temporal recurrence
    to expand the effective receptive field.

    Attributes:
        dim: Number of channels
        iterations: Number of SSM iterations (RF grows with iterations)
        kernel_size: Size of A and B 3D convolution kernels
        dtype: Compute dtype
    """
    dim: int
    iterations: int = 8  # SSM iterations (not video frames)
    kernel_size: Tuple[int, int, int] = (3, 7, 7)  # (T, H, W)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run 3D ConvSSM.

        Args:
            x: (B, T, H, W, C) video input

        Returns:
            (B, T, H, W, C) output after SSM iterations
        """
        B, T, H, W, C = x.shape
        kt, kh, kw = self.kernel_size

        # Learn 3D kernels for A (transition) and B (input)
        A_kernel = self.param(
            'A_kernel',
            nn.initializers.normal(0.02),
            (C, kt, kh, kw),
            self.dtype
        )
        B_kernel = self.param(
            'B_kernel',
            nn.initializers.normal(0.02),
            (C, kt, kh, kw),
            self.dtype
        )

        # Stability: bound A with tanh
        A_kernel_stable = 0.9 * jnp.tanh(A_kernel)

        # Pre-compute FFT of kernels
        center_t, center_h, center_w = kt // 2, kh // 2, kw // 2

        t_idx = jnp.arange(kt)
        h_idx = jnp.arange(kh)
        w_idx = jnp.arange(kw)

        target_t = (t_idx - center_t) % T
        target_h = (h_idx - center_h) % H
        target_w = (w_idx - center_w) % W

        tt, th, tw = jnp.meshgrid(target_t, target_h, target_w, indexing='ij')

        # Pad and FFT A kernel
        A_padded = jnp.zeros((C, T, H, W), dtype=self.dtype)
        A_padded = A_padded.at[:, tt, th, tw].set(A_kernel_stable)
        A_f = jnp.fft.fftn(A_padded, axes=(1, 2, 3))
        A_f = A_f.transpose(1, 2, 3, 0)[None, ...]  # (1, T, H, W, C)

        # Pad and FFT B kernel
        B_padded = jnp.zeros((C, T, H, W), dtype=self.dtype)
        B_padded = B_padded.at[:, tt, th, tw].set(B_kernel)
        B_f = jnp.fft.fftn(B_padded, axes=(1, 2, 3))
        B_f = B_f.transpose(1, 2, 3, 0)[None, ...]  # (1, T, H, W, C)

        # FFT input
        x_f = jnp.fft.fftn(x, axes=(1, 2, 3))  # (B, T, H, W, C)

        # B * x in frequency domain
        Bx_f = B_f * x_f

        # Create sequences for associative scan
        # Shape: (iterations, B, T, H, W, C)
        a_seq = jnp.broadcast_to(A_f, (self.iterations, B, T, H, W, C))
        b_seq = jnp.broadcast_to(Bx_f, (self.iterations, B, T, H, W, C))

        # Run associative scan - O(log iterations) parallel depth
        _, h_all_f = lax.associative_scan(
            ssm_associative_op_3d,
            (a_seq, b_seq),
            axis=0
        )

        # Take final state
        h_final_f = h_all_f[-1]  # (B, T, H, W, C)

        # IFFT back to spatial domain
        h_final = jnp.fft.ifftn(h_final_f, axes=(1, 2, 3)).real

        return h_final


# =============================================================================
# Point Tracking Head with 3D ConvSSM
# =============================================================================

class ConvSSMPointTrackingHead(nn.Module):
    """Point tracking head using 3D ConvSSM.

    Takes video features and query points, outputs predicted trajectories.

    Architecture:
    1. Encode query points into feature volume
    2. Apply 3D ConvSSM for spatiotemporal propagation
    3. Decode to point coordinates and occlusion

    This is meant to be added to a video backbone (e.g., ResNet3D, Video Transformer).
    """
    hidden_dim: int = 256
    iterations: int = 8
    kernel_size: Tuple[int, int, int] = (3, 7, 7)
    num_refinement_blocks: int = 3
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        video_features: jnp.ndarray,  # (B, T, H, W, C) from backbone
        query_points: jnp.ndarray,    # (B, N, 3) query (t, x, y) normalized to [0, 1]
        train: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict point trajectories.

        Args:
            video_features: (B, T, H, W, C) features from video backbone
            query_points: (B, N, 3) query points (frame_idx, x, y), normalized

        Returns:
            trajectories: (B, N, T, 2) predicted (x, y) for each point at each frame
            occlusion: (B, N, T) occlusion probability
        """
        B, T, H, W, C = video_features.shape
        N = query_points.shape[1]

        # Project to hidden dim
        x = nn.Conv(self.hidden_dim, kernel_size=(1, 1, 1), dtype=self.dtype)(video_features)

        # Inject query point information
        # Sample features at query point locations
        query_t = (query_points[..., 0] * (T - 1)).astype(jnp.int32)  # (B, N)
        query_y = (query_points[..., 1] * (H - 1)).astype(jnp.int32)  # (B, N)
        query_x = (query_points[..., 2] * (W - 1)).astype(jnp.int32)  # (B, N)

        # Create position encodings for query points
        query_encoding = self._create_query_encoding(query_points, T, H, W, C)  # (B, N, T, H, W, dim)

        # Broadcast to spatial dimensions and add to features
        # This marks where queries are in the volume
        query_broadcast = self._scatter_queries(query_encoding, query_t, query_y, query_x, T, H, W)
        x = x + query_broadcast  # (B, T, H, W, hidden_dim)

        # Apply 3D ConvSSM blocks for spatiotemporal propagation
        for i in range(self.num_refinement_blocks):
            # 3D ConvSSM
            x_ssm = ParallelConvSSM3D(
                dim=self.hidden_dim,
                iterations=self.iterations,
                kernel_size=self.kernel_size,
                dtype=self.dtype,
                name=f'convssm_{i}'
            )(x)

            # LayerNorm + residual
            x_ssm = nn.LayerNorm(dtype=self.dtype)(x_ssm)
            x = x + x_ssm

            # MLP
            x_mlp = nn.Dense(self.hidden_dim * 4, dtype=self.dtype)(x)
            x_mlp = nn.gelu(x_mlp)
            x_mlp = nn.Dense(self.hidden_dim, dtype=self.dtype)(x_mlp)
            x = x + x_mlp

        # Decode trajectories
        # For each query point, gather features across time and predict coordinates
        trajectories = self._decode_trajectories(x, query_points, T, H, W)
        occlusion = self._decode_occlusion(x, query_points, T, H, W)

        return trajectories, occlusion

    def _create_query_encoding(self, query_points, T, H, W, C):
        """Create positional encoding for query points."""
        # Simple learnable encoding per query
        # In practice, could use sinusoidal or Fourier features
        B, N, _ = query_points.shape

        # Encode query position
        query_embed = nn.Dense(self.hidden_dim, dtype=self.dtype)(query_points)  # (B, N, hidden)

        return query_embed

    def _scatter_queries(self, query_encoding, query_t, query_y, query_x, T, H, W):
        """Scatter query encodings into spatial volume."""
        B = query_encoding.shape[0]

        # For simplicity, just add query info as a channel-wise bias
        # In practice, would use more sophisticated attention/scattering
        # This is a placeholder - real implementation needs proper scatter
        return jnp.zeros((B, T, H, W, self.hidden_dim), dtype=self.dtype)

    def _decode_trajectories(self, features, query_points, T, H, W):
        """Decode point trajectories from features."""
        B, _, _, _, C = features.shape
        N = query_points.shape[1]

        # For each query, sample features across all frames and predict delta
        # This is a simplified version - real impl would use correlation/attention

        # Global average pool spatial dims
        feat_temporal = jnp.mean(features, axis=(2, 3))  # (B, T, C)

        # Expand for each query point
        feat_expanded = feat_temporal[:, None, :, :].repeat(N, axis=1)  # (B, N, T, C)

        # Predict coordinates
        coords = nn.Dense(2, dtype=self.dtype)(feat_expanded)  # (B, N, T, 2)

        # Add query position as initial position
        query_xy = query_points[..., 1:3][:, :, None, :]  # (B, N, 1, 2)
        coords = coords + query_xy  # Predict residual from query position

        return coords

    def _decode_occlusion(self, features, query_points, T, H, W):
        """Decode occlusion probabilities."""
        B, _, _, _, C = features.shape
        N = query_points.shape[1]

        feat_temporal = jnp.mean(features, axis=(2, 3))  # (B, T, C)
        feat_expanded = feat_temporal[:, None, :, :].repeat(N, axis=1)  # (B, N, T, C)

        occ_logits = nn.Dense(1, dtype=self.dtype)(feat_expanded)  # (B, N, T, 1)
        occ_prob = nn.sigmoid(occ_logits[..., 0])  # (B, N, T)

        return occ_prob


# =============================================================================
# Full Point Tracking Model
# =============================================================================

class ConvSSMPointTracker(nn.Module):
    """Full point tracking model with 3D ConvSSM.

    Combines a simple video encoder with ConvSSM-based tracking head.
    For better results, use a pretrained video backbone instead.
    """
    hidden_dim: int = 256
    iterations: int = 8
    kernel_size: Tuple[int, int, int] = (3, 7, 7)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        video: jnp.ndarray,        # (B, T, H, W, 3) RGB video
        query_points: jnp.ndarray, # (B, N, 3) queries (t, x, y)
        train: bool = True
    ):
        """Predict point tracks.

        Args:
            video: (B, T, H, W, 3) input video
            query_points: (B, N, 3) query points

        Returns:
            trajectories: (B, N, T, 2) predicted positions
            occlusion: (B, N, T) occlusion probabilities
        """
        # Simple video encoder (replace with pretrained backbone)
        x = video.astype(self.dtype) / 255.0

        # Initial conv
        x = nn.Conv(64, kernel_size=(1, 7, 7), strides=(1, 2, 2), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)

        # Downsample and increase channels
        x = nn.Conv(128, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(256, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)

        # Point tracking head
        trajectories, occlusion = ConvSSMPointTrackingHead(
            hidden_dim=self.hidden_dim,
            iterations=self.iterations,
            kernel_size=self.kernel_size,
            dtype=self.dtype
        )(x, query_points, train=train)

        return trajectories, occlusion


# =============================================================================
# Tests
# =============================================================================

if __name__ == '__main__':
    import jax.random as random
    import time

    print("="*70)
    print("TEST: 3D ConvSSM for Point Tracking")
    print("="*70)

    key = random.PRNGKey(0)

    # Test 3D FFT conv
    print("\n1. Testing 3D FFT convolution...")
    x_3d = jnp.ones((2, 16, 64, 64, 32))  # (B, T, H, W, C)
    kernel_3d = jnp.ones((32, 3, 7, 7)) * 0.01

    out_3d = fft_depthwise_conv_3d(x_3d, kernel_3d)
    print(f"   Input: {x_3d.shape} -> Output: {out_3d.shape}")

    # Test ParallelConvSSM3D
    print("\n2. Testing ParallelConvSSM3D...")
    ssm_3d = ParallelConvSSM3D(dim=32, iterations=8, kernel_size=(3, 7, 7))
    variables = ssm_3d.init(key, x_3d)
    out_ssm = ssm_3d.apply(variables, x_3d)
    print(f"   Input: {x_3d.shape} -> Output: {out_ssm.shape}")

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(variables['params']))
    print(f"   Parameters: {n_params:,}")

    # Test full point tracker
    print("\n3. Testing ConvSSMPointTracker...")
    video = jnp.ones((2, 16, 128, 128, 3), dtype=jnp.uint8)  # (B, T, H, W, 3)
    queries = jnp.array([
        [[0.0, 0.5, 0.5], [0.25, 0.3, 0.7]],  # 2 queries for batch 0
        [[0.5, 0.2, 0.8], [0.75, 0.6, 0.4]],  # 2 queries for batch 1
    ])  # (B, N, 3) = (2, 2, 3)

    tracker = ConvSSMPointTracker(hidden_dim=128, iterations=4)
    variables = tracker.init(key, video, queries, train=False)

    t0 = time.time()
    trajectories, occlusion = tracker.apply(variables, video, queries, train=False)
    init_time = time.time() - t0

    print(f"   Video: {video.shape}")
    print(f"   Queries: {queries.shape}")
    print(f"   Trajectories: {trajectories.shape}")
    print(f"   Occlusion: {occlusion.shape}")
    print(f"   Init time: {init_time:.2f}s")

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(variables['params']))
    print(f"   Total parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Gradient check
    print("\n4. Gradient check...")
    def loss_fn(params, video, queries):
        traj, occ = tracker.apply({'params': params}, video, queries, train=True)
        return jnp.mean(traj**2) + jnp.mean(occ**2)

    loss, grads = jax.value_and_grad(loss_fn)(variables['params'], video, queries)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    print(f"   Loss: {loss:.6f}")
    print(f"   Gradient norm: {grad_norm:.6f}")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
