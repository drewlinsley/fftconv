"""Train 3D ConvSSM on TAP-Vid Point Tracking.

This script trains the 3D ConvSSM model on the TAP-Vid DAVIS dataset
for point tracking.

Usage:
    python -m flashfftconv.train_point_tracking \
        --data_dir ./data/tapvid/tapvid_davis/tapvid_davis \
        --output_dir ./checkpoints_point_tracking \
        --model_type ssm_3d \
        --epochs 100

Reference:
    - TAP-Vid: https://arxiv.org/abs/2211.03726
    - 3D ConvSSM: convssm_3d.py
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import cv2

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =============================================================================
# TAP-Vid Dataset
# =============================================================================

class TAPVidDataset:
    """TAP-Vid dataset loader for JAX training.

    Each sample contains:
        - video: (T, H, W, 3) uint8 RGB frames
        - points: (N, T, 2) float32 point coordinates (x, y in pixels)
        - occluded: (N, T) bool, True if point is occluded
    """

    def __init__(
        self,
        data_path: str,
        resize: Tuple[int, int] = (256, 256),
        max_frames: int = 32,
        max_points: int = 20,
        split: str = 'all',  # 'train', 'val', 'all'
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.resize = resize
        self.max_frames = max_frames
        self.max_points = max_points

        # Load dataset
        print(f"Loading TAP-Vid dataset from {data_path}...")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        all_names = sorted(list(self.data.keys()))
        print(f"  Total videos: {len(all_names)}")

        # Train/val split
        np.random.seed(seed)
        n_val = int(len(all_names) * val_ratio)
        val_indices = np.random.choice(len(all_names), n_val, replace=False)
        val_names = [all_names[i] for i in val_indices]
        train_names = [n for n in all_names if n not in val_names]

        if split == 'train':
            self.video_names = train_names
            print(f"  Using TRAIN split: {len(self.video_names)} videos")
        elif split == 'val':
            self.video_names = val_names
            print(f"  Using VAL split: {len(self.video_names)} videos")
        else:
            self.video_names = all_names
            print(f"  Using ALL data: {len(self.video_names)} videos")

        # Show example shapes
        example = self.data[self.video_names[0]]
        print(f"  Example video: {example['video'].shape}")
        print(f"  Example points: {example['points'].shape}")

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample, resized and normalized."""
        name = self.video_names[idx]
        sample = self.data[name]

        video = sample['video']  # (T, H, W, 3)
        points = sample['points']  # (N, T, 2)
        occluded = sample['occluded']  # (N, T)

        T, H, W, C = video.shape
        N = points.shape[0]

        # Subsample frames if too long
        if T > self.max_frames:
            indices = np.linspace(0, T-1, self.max_frames, dtype=np.int32)
            video = video[indices]
            points = points[:, indices]
            occluded = occluded[:, indices]
            T = self.max_frames

        # Subsample points if too many
        if N > self.max_points:
            perm = np.random.permutation(N)[:self.max_points]
            points = points[perm]
            occluded = occluded[perm]
            N = self.max_points

        # Resize video
        new_H, new_W = self.resize
        video_resized = np.zeros((T, new_H, new_W, C), dtype=np.uint8)
        for t in range(T):
            video_resized[t] = cv2.resize(video[t], (new_W, new_H))

        # Scale points
        scale_x = new_W / W
        scale_y = new_H / H
        points_scaled = points.copy()
        points_scaled[..., 0] *= scale_x  # x
        points_scaled[..., 1] *= scale_y  # y

        # Normalize points to [0, 1]
        points_norm = points_scaled.copy()
        points_norm[..., 0] /= new_W
        points_norm[..., 1] /= new_H

        # Create query points (sample from first frames where visible)
        query_points = np.zeros((N, 3), dtype=np.float32)  # (t, y, x) normalized
        for i in range(N):
            # Find first visible frame
            visible = ~occluded[i]
            if visible.any():
                first_visible = np.where(visible)[0][0]
            else:
                first_visible = 0

            query_points[i, 0] = first_visible / max(T - 1, 1)  # t normalized
            query_points[i, 1] = points_norm[i, first_visible, 1]  # y
            query_points[i, 2] = points_norm[i, first_visible, 0]  # x

        return {
            'video': video_resized,  # (T, H, W, 3) uint8
            'points': points_scaled.astype(np.float32),  # (N, T, 2) in pixels
            'points_norm': points_norm.astype(np.float32),  # (N, T, 2) normalized
            'occluded': occluded.astype(bool),  # (N, T)
            'query_points': query_points,  # (N, 3) normalized (t, y, x)
            'name': name,
        }


def collate_batch(samples, pad_frames=32, pad_points=20):
    """Collate samples into a batch with padding."""
    B = len(samples)
    H, W = samples[0]['video'].shape[1:3]

    # Pad to same size
    videos = np.zeros((B, pad_frames, H, W, 3), dtype=np.uint8)
    points = np.zeros((B, pad_points, pad_frames, 2), dtype=np.float32)
    points_norm = np.zeros((B, pad_points, pad_frames, 2), dtype=np.float32)
    occluded = np.ones((B, pad_points, pad_frames), dtype=bool)  # Default occluded
    query_points = np.zeros((B, pad_points, 3), dtype=np.float32)
    valid_frames = np.zeros(B, dtype=np.int32)
    valid_points = np.zeros(B, dtype=np.int32)

    for i, sample in enumerate(samples):
        T = sample['video'].shape[0]
        N = sample['points'].shape[0]

        videos[i, :T] = sample['video']
        points[i, :N, :T] = sample['points']
        points_norm[i, :N, :T] = sample['points_norm']
        occluded[i, :N, :T] = sample['occluded']
        query_points[i, :N] = sample['query_points']
        valid_frames[i] = T
        valid_points[i] = N

    return {
        'video': videos,
        'points': points,
        'points_norm': points_norm,
        'occluded': occluded,
        'query_points': query_points,
        'valid_frames': valid_frames,
        'valid_points': valid_points,
    }


# =============================================================================
# 3D ConvSSM Point Tracker (Improved Architecture)
# =============================================================================

def fft_depthwise_conv_3d(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """3D FFT-based depthwise convolution."""
    B, T, H, W, C = x.shape
    kt, kh, kw = kernel.shape[1], kernel.shape[2], kernel.shape[3]

    center_t = kt // 2
    center_h = kh // 2
    center_w = kw // 2

    t_idx = jnp.arange(kt)
    h_idx = jnp.arange(kh)
    w_idx = jnp.arange(kw)

    target_t = (t_idx - center_t) % T
    target_h = (h_idx - center_h) % H
    target_w = (w_idx - center_w) % W

    tt, th, tw = jnp.meshgrid(target_t, target_h, target_w, indexing='ij')

    padded_kernel = jnp.zeros((C, T, H, W), dtype=kernel.dtype)
    padded_kernel = padded_kernel.at[:, tt, th, tw].set(kernel)

    x_f = jnp.fft.fftn(x, axes=(1, 2, 3))
    kernel_f = jnp.fft.fftn(padded_kernel, axes=(1, 2, 3))
    kernel_f = kernel_f.transpose(1, 2, 3, 0)[None, ...]
    out_f = x_f * kernel_f
    out = jnp.fft.ifftn(out_f, axes=(1, 2, 3)).real

    return out


def ssm_associative_op(left, right):
    """Associative operation for SSM parallel scan."""
    a_left, b_left = left
    a_right, b_right = right
    return (a_left * a_right, a_right * b_left + b_right)


class ParallelConvSSM3D(nn.Module):
    """3D Parallel ConvSSM using associative scan."""
    dim: int
    iterations: int = 8
    kernel_size: Tuple[int, int, int] = (3, 7, 7)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, H, W, C = x.shape
        kt, kh, kw = self.kernel_size

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

        A_kernel_stable = 0.9 * jnp.tanh(A_kernel)

        center_t, center_h, center_w = kt // 2, kh // 2, kw // 2
        t_idx, h_idx, w_idx = jnp.arange(kt), jnp.arange(kh), jnp.arange(kw)
        target_t = (t_idx - center_t) % T
        target_h = (h_idx - center_h) % H
        target_w = (w_idx - center_w) % W
        tt, th, tw = jnp.meshgrid(target_t, target_h, target_w, indexing='ij')

        A_padded = jnp.zeros((C, T, H, W), dtype=self.dtype)
        A_padded = A_padded.at[:, tt, th, tw].set(A_kernel_stable)
        A_f = jnp.fft.fftn(A_padded, axes=(1, 2, 3))
        A_f = A_f.transpose(1, 2, 3, 0)[None, ...]

        B_padded = jnp.zeros((C, T, H, W), dtype=self.dtype)
        B_padded = B_padded.at[:, tt, th, tw].set(B_kernel)
        B_f = jnp.fft.fftn(B_padded, axes=(1, 2, 3))
        B_f = B_f.transpose(1, 2, 3, 0)[None, ...]

        x_f = jnp.fft.fftn(x, axes=(1, 2, 3))
        Bx_f = B_f * x_f

        a_seq = jnp.broadcast_to(A_f, (self.iterations, B, T, H, W, C))
        b_seq = jnp.broadcast_to(Bx_f, (self.iterations, B, T, H, W, C))

        _, h_all_f = jax.lax.associative_scan(
            ssm_associative_op,
            (a_seq, b_seq),
            axis=0
        )

        h_final_f = h_all_f[-1]
        h_final = jnp.fft.ifftn(h_final_f, axes=(1, 2, 3)).real

        return h_final


class CorrelationOnlyTracker(nn.Module):
    """Correlation-only Point Tracker (baseline without SSM).

    Architecture:
    1. Video encoder (simple CNN)
    2. Query feature extraction
    3. Correlation-based position prediction (NO SSM)
    4. Occlusion prediction
    """
    hidden_dim: int = 256
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        video: jnp.ndarray,        # (B, T, H, W, 3) uint8
        query_points: jnp.ndarray, # (B, N, 3) normalized (t, y, x)
        train: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict point trajectories using correlation only."""
        B, T, H, W, _ = video.shape
        N = query_points.shape[1]

        # Normalize video
        x = video.astype(self.dtype) / 255.0

        # Video encoder
        x = nn.Conv(64, (1, 7, 7), (1, 2, 2), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(128, (1, 3, 3), (1, 2, 2), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_dim, (1, 3, 3), (1, 2, 2), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)

        feat_H, feat_W = x.shape[2], x.shape[3]

        # Sample query features
        query_t = (query_points[:, :, 0] * (T - 1)).astype(jnp.int32)
        query_y_feat = (query_points[:, :, 1] * (feat_H - 1))
        query_x_feat = (query_points[:, :, 2] * (feat_W - 1))

        query_feats = self._sample_features(x, query_t, query_y_feat, query_x_feat)

        # Correlation between query and all positions (NO SSM refinement)
        query_feats_exp = query_feats[:, :, None, None, None, :]
        x_exp = x[:, None, :, :, :, :]
        correlation = jnp.sum(query_feats_exp * x_exp, axis=-1)
        correlation = correlation / jnp.sqrt(self.hidden_dim)

        # Soft argmax to get coordinates
        heatmaps_flat = correlation.reshape(B, N, T, -1)
        probs = jax.nn.softmax(heatmaps_flat * 10.0, axis=-1)

        y_coords = jnp.arange(feat_H, dtype=self.dtype)
        x_coords = jnp.arange(feat_W, dtype=self.dtype)
        yy, xx = jnp.meshgrid(y_coords, x_coords, indexing='ij')
        coords_flat = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)

        pred_coords = jnp.einsum('bntk,kc->bntc', probs, coords_flat)

        # Scale to full resolution
        pred_coords_full = pred_coords.at[..., 0].set(pred_coords[..., 0] * (W / feat_W))
        pred_coords_full = pred_coords_full.at[..., 1].set(pred_coords[..., 1] * (H / feat_H))

        # Occlusion prediction from correlation
        max_corr = jnp.max(correlation, axis=(-2, -1))  # (B, N, T)
        occlusion = jax.nn.sigmoid(-max_corr)  # Low correlation = occluded

        return pred_coords_full, occlusion

    def _sample_features(self, features, t_idx, y_idx, x_idx):
        """Bilinear sample features at given coordinates."""
        B, T, H, W, C = features.shape
        t_idx = jnp.clip(t_idx, 0, T - 1)
        y0 = jnp.clip(jnp.floor(y_idx).astype(jnp.int32), 0, H - 1)
        y1 = jnp.clip(y0 + 1, 0, H - 1)
        x0 = jnp.clip(jnp.floor(x_idx).astype(jnp.int32), 0, W - 1)
        x1 = jnp.clip(x0 + 1, 0, W - 1)

        wy1 = y_idx - y0.astype(self.dtype)
        wy0 = 1.0 - wy1
        wx1 = x_idx - x0.astype(self.dtype)
        wx0 = 1.0 - wx1

        batch_idx = jnp.arange(B)[:, None]
        f00 = features[batch_idx, t_idx, y0, x0]
        f01 = features[batch_idx, t_idx, y0, x1]
        f10 = features[batch_idx, t_idx, y1, x0]
        f11 = features[batch_idx, t_idx, y1, x1]

        f0 = f00 * wx0[:, :, None] + f01 * wx1[:, :, None]
        f1 = f10 * wx0[:, :, None] + f11 * wx1[:, :, None]
        return f0 * wy0[:, :, None] + f1 * wy1[:, :, None]


class ConvSSMPointTracker(nn.Module):
    """3D ConvSSM Point Tracker with correlation-based tracking.

    Architecture:
    1. Video encoder (simple CNN)
    2. Query feature extraction
    3. 3D ConvSSM for spatiotemporal propagation
    4. Correlation-based position prediction
    5. Occlusion prediction
    """
    hidden_dim: int = 256
    ssm_iterations: int = 8
    kernel_size: Tuple[int, int, int] = (3, 7, 7)
    num_refinement: int = 3
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        video: jnp.ndarray,        # (B, T, H, W, 3) uint8
        query_points: jnp.ndarray, # (B, N, 3) normalized (t, y, x)
        train: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict point trajectories.

        Returns:
            trajectories: (B, N, T, 2) predicted positions (x, y) in pixels
            occlusion: (B, N, T) occlusion probability
        """
        B, T, H, W, _ = video.shape
        N = query_points.shape[1]

        # Normalize video
        x = video.astype(self.dtype) / 255.0

        # Video encoder - extract multi-scale features
        x = nn.Conv(64, (1, 7, 7), (1, 2, 2), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(128, (1, 3, 3), (1, 2, 2), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_dim, (1, 3, 3), (1, 2, 2), padding='SAME', dtype=self.dtype)(x)
        x = nn.relu(x)

        # Feature shape: (B, T, H/8, W/8, hidden_dim)
        feat_H, feat_W = x.shape[2], x.shape[3]

        # Sample query features
        # Convert normalized query (t, y, x) to feature grid coords
        query_t = (query_points[:, :, 0] * (T - 1)).astype(jnp.int32)  # (B, N)
        query_y_feat = (query_points[:, :, 1] * (feat_H - 1))  # (B, N)
        query_x_feat = (query_points[:, :, 2] * (feat_W - 1))  # (B, N)

        # Bilinear sample query features
        query_feats = self._sample_features(x, query_t, query_y_feat, query_x_feat)  # (B, N, C)

        # Inject query information into feature volume
        # Create query attention: which spatial locations match query features
        query_feats_exp = query_feats[:, :, None, None, None, :]  # (B, N, 1, 1, 1, C)
        x_exp = x[:, None, :, :, :, :]  # (B, 1, T, H, W, C)

        # Correlation between query and all positions
        correlation = jnp.sum(query_feats_exp * x_exp, axis=-1)  # (B, N, T, H, W)
        correlation = correlation / jnp.sqrt(self.hidden_dim)

        # Softmax to get attention weights
        corr_flat = correlation.reshape(B, N, T, -1)
        attn_weights = jax.nn.softmax(corr_flat, axis=-1)
        attn_weights = attn_weights.reshape(B, N, T, feat_H, feat_W)

        # Apply 3D ConvSSM to propagate correlation maps over time
        # Reshape for SSM: (B*N, T, H, W, 1)
        corr_ssm = attn_weights.reshape(B * N, T, feat_H, feat_W, 1)

        # Add feature channels for SSM processing
        corr_ssm = nn.Conv(64, (1, 1, 1), dtype=self.dtype)(corr_ssm)

        # 3D ConvSSM refinement
        for i in range(self.num_refinement):
            corr_residual = ParallelConvSSM3D(
                dim=64,
                iterations=self.ssm_iterations,
                kernel_size=self.kernel_size,
                dtype=self.dtype,
                name=f'ssm_{i}'
            )(corr_ssm)
            corr_residual = nn.LayerNorm(dtype=self.dtype)(corr_residual)
            corr_ssm = corr_ssm + corr_residual

        # Decode to heatmaps
        heatmaps = nn.Conv(1, (1, 1, 1), dtype=self.dtype)(corr_ssm)
        heatmaps = heatmaps.reshape(B, N, T, feat_H, feat_W)

        # Soft argmax to get coordinates
        heatmaps_flat = heatmaps.reshape(B, N, T, -1)
        probs = jax.nn.softmax(heatmaps_flat * 10.0, axis=-1)  # Temperature

        # Create coordinate grids
        y_coords = jnp.arange(feat_H, dtype=self.dtype)
        x_coords = jnp.arange(feat_W, dtype=self.dtype)
        yy, xx = jnp.meshgrid(y_coords, x_coords, indexing='ij')

        coords_flat = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)  # (H*W, 2)

        # Weighted sum to get expected coordinates
        pred_coords = jnp.einsum('bntk,kc->bntc', probs, coords_flat)  # (B, N, T, 2)

        # Scale to full resolution
        pred_coords_full = pred_coords.at[..., 0].set(pred_coords[..., 0] * (W / feat_W))
        pred_coords_full = pred_coords_full.at[..., 1].set(pred_coords[..., 1] * (H / feat_H))

        # Occlusion prediction
        occ_feats = corr_ssm.reshape(B, N, T, feat_H, feat_W, -1)
        occ_pooled = jnp.mean(occ_feats, axis=(3, 4))  # (B, N, T, C)
        occ_logits = nn.Dense(1, dtype=self.dtype)(occ_pooled)  # (B, N, T, 1)
        occlusion = jax.nn.sigmoid(occ_logits[..., 0])  # (B, N, T)

        return pred_coords_full, occlusion

    def _sample_features(self, features, t_idx, y_idx, x_idx):
        """Bilinear sample features at given coordinates."""
        B, T, H, W, C = features.shape
        N = t_idx.shape[1]

        # Clamp coordinates
        t_idx = jnp.clip(t_idx, 0, T - 1)
        y0 = jnp.clip(jnp.floor(y_idx).astype(jnp.int32), 0, H - 1)
        y1 = jnp.clip(y0 + 1, 0, H - 1)
        x0 = jnp.clip(jnp.floor(x_idx).astype(jnp.int32), 0, W - 1)
        x1 = jnp.clip(x0 + 1, 0, W - 1)

        # Interpolation weights
        wy1 = y_idx - y0.astype(self.dtype)
        wy0 = 1.0 - wy1
        wx1 = x_idx - x0.astype(self.dtype)
        wx0 = 1.0 - wx1

        # Gather and interpolate
        batch_idx = jnp.arange(B)[:, None]  # (B, 1)

        f00 = features[batch_idx, t_idx, y0, x0]  # (B, N, C)
        f01 = features[batch_idx, t_idx, y0, x1]
        f10 = features[batch_idx, t_idx, y1, x0]
        f11 = features[batch_idx, t_idx, y1, x1]

        f0 = f00 * wx0[:, :, None] + f01 * wx1[:, :, None]
        f1 = f10 * wx0[:, :, None] + f11 * wx1[:, :, None]
        out = f0 * wy0[:, :, None] + f1 * wy1[:, :, None]

        return out


# =============================================================================
# Loss Functions and Metrics
# =============================================================================

def huber_loss(pred, target, delta=1.0):
    """Huber loss for robust coordinate regression."""
    diff = pred - target
    abs_diff = jnp.abs(diff)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def point_tracking_loss(
    pred_coords: jnp.ndarray,   # (B, N, T, 2)
    pred_occ: jnp.ndarray,      # (B, N, T)
    gt_coords: jnp.ndarray,     # (B, N, T, 2)
    gt_occ: jnp.ndarray,        # (B, N, T)
    valid_mask: jnp.ndarray,    # (B, N, T) validity mask
):
    """Combined loss for point tracking."""
    # Only compute coordinate loss on visible points
    visible_mask = (~gt_occ) & valid_mask

    # Coordinate loss (Huber)
    coord_diff = huber_loss(pred_coords, gt_coords)
    coord_loss = jnp.sum(coord_diff * visible_mask[..., None]) / (jnp.sum(visible_mask) + 1e-6)

    # Occlusion loss (BCE)
    # Convert probabilities to logits: logit(p) = log(p / (1-p))
    pred_occ_clipped = jnp.clip(pred_occ, 1e-6, 1-1e-6)
    pred_occ_logits = jnp.log(pred_occ_clipped / (1 - pred_occ_clipped))
    occ_loss = optax.sigmoid_binary_cross_entropy(
        pred_occ_logits,
        gt_occ.astype(jnp.float32)
    )
    occ_loss = jnp.sum(occ_loss * valid_mask) / (jnp.sum(valid_mask) + 1e-6)

    total_loss = coord_loss + 0.1 * occ_loss

    return total_loss, {
        'coord_loss': coord_loss,
        'occ_loss': occ_loss,
    }


def compute_metrics(
    pred_coords: np.ndarray,
    pred_occ: np.ndarray,
    gt_coords: np.ndarray,
    gt_occ: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute TAP-Vid evaluation metrics."""
    visible = (~gt_occ) & valid_mask

    # Position errors
    errors = np.linalg.norm(pred_coords - gt_coords, axis=-1)  # (B, N, T)

    metrics = {}

    # Average error on visible points
    if np.sum(visible) > 0:
        metrics['avg_error'] = float(np.mean(errors[visible]))
    else:
        metrics['avg_error'] = 0.0

    # Delta metrics: fraction within threshold
    thresholds = [1, 2, 4, 8, 16]
    for thresh in thresholds:
        within = errors < thresh
        if np.sum(visible) > 0:
            metrics[f'delta_{thresh}'] = float(np.mean(within[visible]))
        else:
            metrics[f'delta_{thresh}'] = 0.0

    # Occlusion accuracy
    pred_occ_binary = pred_occ > 0.5
    if np.sum(valid_mask) > 0:
        metrics['occ_acc'] = float(np.mean((pred_occ_binary == gt_occ)[valid_mask]))
    else:
        metrics['occ_acc'] = 0.0

    # Average Jaccard (simplified)
    jaccard_sum = 0.0
    for thresh in thresholds:
        pos_correct = errors < thresh
        occ_correct = pred_occ_binary == gt_occ
        both_correct = pos_correct & occ_correct & valid_mask

        pred_vis = (~pred_occ_binary) & valid_mask
        gt_vis = (~gt_occ) & valid_mask

        tp = both_correct & pred_vis & gt_vis
        union = valid_mask & (pred_vis | gt_vis | both_correct)

        if np.sum(union) > 0:
            jaccard_sum += np.sum(tp) / np.sum(union)

    metrics['avg_jaccard'] = jaccard_sum / len(thresholds)

    return metrics


# =============================================================================
# Training Functions
# =============================================================================

class TrainState(train_state.TrainState):
    """Extended train state with dropout RNG."""
    key: jax.Array


def create_train_state(
    rng: jax.Array,
    model: nn.Module,
    learning_rate: float,
    video_shape: Tuple,
    query_shape: Tuple,
):
    """Create initial training state."""
    dummy_video = jnp.ones(video_shape, dtype=jnp.uint8)
    dummy_query = jnp.ones(query_shape, dtype=jnp.float32)

    params = model.init(rng, dummy_video, dummy_query, train=False)['params']

    # Count parameters
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Optimizer with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=500,
        decay_steps=10000,
        end_value=learning_rate * 0.01,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=0.01),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        key=rng,
    )


@jax.jit
def train_step(state, video, query_points, gt_coords, gt_occ, valid_mask):
    """Single training step."""
    def loss_fn(params):
        pred_coords, pred_occ = state.apply_fn(
            {'params': params},
            video,
            query_points,
            train=True,
        )
        loss, loss_dict = point_tracking_loss(
            pred_coords, pred_occ, gt_coords, gt_occ, valid_mask
        )
        return loss, (loss_dict, pred_coords, pred_occ)

    (loss, (loss_dict, pred_coords, pred_occ)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)

    state = state.apply_gradients(grads=grads)

    return state, loss, loss_dict, pred_coords, pred_occ


@jax.jit
def eval_step(state, video, query_points):
    """Evaluation step (no gradients)."""
    pred_coords, pred_occ = state.apply_fn(
        {'params': state.params},
        video,
        query_points,
        train=False,
    )
    return pred_coords, pred_occ


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args):
    """Main training function."""
    print("="*70)
    print("3D ConvSSM Point Tracking Training")
    print("="*70)

    # Setup
    rng = jax.random.PRNGKey(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if HAS_WANDB and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"point-track-{args.model_type}",
            config=vars(args),
        )
        print(f"Wandb run: {wandb.run.url}")

    # Load datasets with proper train/val split
    train_dataset = TAPVidDataset(
        args.data_path,
        resize=(args.img_size, args.img_size),
        max_frames=args.max_frames,
        max_points=args.max_points,
        split='train',
        val_ratio=0.2,
    )
    val_dataset = TAPVidDataset(
        args.data_path,
        resize=(args.img_size, args.img_size),
        max_frames=args.max_frames,
        max_points=args.max_points,
        split='val',
        val_ratio=0.2,
    )
    dataset = train_dataset  # For training loop

    # Create model based on model_type
    if args.model_type == 'correlation_only':
        print("Using CorrelationOnlyTracker (no SSM)")
        model = CorrelationOnlyTracker(
            hidden_dim=args.hidden_dim,
        )
    else:
        print(f"Using ConvSSMPointTracker (ssm_iterations={args.ssm_iterations})")
        model = ConvSSMPointTracker(
            hidden_dim=args.hidden_dim,
            ssm_iterations=args.ssm_iterations,
            kernel_size=(args.kernel_t, args.kernel_h, args.kernel_w),
            num_refinement=args.num_refinement,
        )

    # Create train state
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        init_rng,
        model,
        args.learning_rate,
        video_shape=(1, args.max_frames, args.img_size, args.img_size, 3),
        query_shape=(1, args.max_points, 3),
    )

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Dataset: {len(dataset)} videos")

    best_metric = 0.0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Shuffle dataset
        rng, shuffle_rng = jax.random.split(rng)
        indices = jax.random.permutation(shuffle_rng, len(dataset))

        epoch_losses = []

        # Training loop
        for batch_start in range(0, len(dataset), args.batch_size):
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            samples = [dataset[int(i)] for i in batch_indices]
            batch = collate_batch(samples, args.max_frames, args.max_points)

            # Convert to JAX arrays
            video = jnp.array(batch['video'])
            query_points = jnp.array(batch['query_points'])
            gt_coords = jnp.array(batch['points'])
            gt_occ = jnp.array(batch['occluded'])

            # Create valid mask
            valid_frames = batch['valid_frames']
            valid_points = batch['valid_points']
            valid_mask = np.zeros((len(samples), args.max_points, args.max_frames), dtype=bool)
            for i in range(len(samples)):
                valid_mask[i, :valid_points[i], :valid_frames[i]] = True
            valid_mask = jnp.array(valid_mask)

            # Train step
            state, loss, loss_dict, pred_coords, pred_occ = train_step(
                state, video, query_points, gt_coords, gt_occ, valid_mask
            )

            epoch_losses.append(float(loss))

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start

        # Evaluate on VALIDATION dataset (held-out)
        all_pred_coords = []
        all_pred_occ = []
        all_gt_coords = []
        all_gt_occ = []
        all_valid = []

        for i in range(len(val_dataset)):
            sample = val_dataset[i]  # Use VAL dataset for evaluation
            batch = collate_batch([sample], args.max_frames, args.max_points)

            video = jnp.array(batch['video'])
            query_points = jnp.array(batch['query_points'])

            pred_coords, pred_occ = eval_step(state, video, query_points)

            # Extract valid region
            vf = batch['valid_frames'][0]
            vp = batch['valid_points'][0]

            all_pred_coords.append(np.array(pred_coords[0, :vp, :vf]))
            all_pred_occ.append(np.array(pred_occ[0, :vp, :vf]))
            all_gt_coords.append(batch['points'][0, :vp, :vf])
            all_gt_occ.append(batch['occluded'][0, :vp, :vf])
            all_valid.append(np.ones((vp, vf), dtype=bool))

        # Compute metrics
        metrics = compute_metrics(
            np.concatenate([c.flatten() for c in all_pred_coords]).reshape(-1, 2)[None],
            np.concatenate([o.flatten() for o in all_pred_occ]).reshape(-1)[None],
            np.concatenate([c.flatten() for c in all_gt_coords]).reshape(-1, 2)[None],
            np.concatenate([o.flatten() for o in all_gt_occ]).reshape(-1)[None],
            np.ones((1, sum(v.size for v in all_valid)), dtype=bool),
        )

        print(f"Epoch {epoch+1:3d}/{args.epochs} ({epoch_time:.1f}s) | "
              f"Loss: {avg_loss:.4f} | "
              f"Err: {metrics['avg_error']:.2f}px | "
              f"δ<4: {metrics['delta_4']*100:.1f}% | "
              f"AJ: {metrics['avg_jaccard']*100:.1f}%")

        # Log to wandb
        if HAS_WANDB and args.wandb_project:
            wandb.log({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'coord_loss': float(loss_dict['coord_loss']),
                'occ_loss': float(loss_dict['occ_loss']),
                **metrics,
            })

        # Save best model
        if metrics['avg_jaccard'] > best_metric:
            best_metric = metrics['avg_jaccard']
            # Save checkpoint
            import pickle
            ckpt_path = output_dir / 'best_model.pkl'
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'params': jax.tree_util.tree_map(np.array, state.params)}, f)
            print(f"  -> New best AJ: {best_metric*100:.2f}%")

    print("\n" + "="*70)
    print(f"Training complete! Best Average Jaccard: {best_metric*100:.2f}%")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Train 3D ConvSSM Point Tracker')

    # Data
    parser.add_argument('--data_path', type=str,
                        default='./data/tapvid/tapvid_davis/tapvid_davis/tapvid_davis.pkl',
                        help='Path to TAP-Vid pickle file')
    parser.add_argument('--output_dir', type=str,
                        default='./checkpoints_point_tracking',
                        help='Output directory')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--max_frames', type=int, default=32,
                        help='Maximum number of frames per video')
    parser.add_argument('--max_points', type=int, default=20,
                        help='Maximum number of points per video')

    # Model
    parser.add_argument('--model_type', type=str, default='ssm_3d',
                        choices=['ssm_3d', 'correlation_only'],
                        help='Model type (ssm_3d or correlation_only baseline)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--ssm_iterations', type=int, default=8,
                        help='SSM iterations')
    parser.add_argument('--kernel_t', type=int, default=3,
                        help='Temporal kernel size')
    parser.add_argument('--kernel_h', type=int, default=7,
                        help='Height kernel size')
    parser.add_argument('--kernel_w', type=int, default=7,
                        help='Width kernel size')
    parser.add_argument('--num_refinement', type=int, default=3,
                        help='Number of SSM refinement blocks')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Logging
    parser.add_argument('--wandb_project', type=str, default='convnext-point-tracking',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
