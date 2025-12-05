"""L-JEPA (Lean Joint-Embedding Predictive Architecture) Training in JAX.

Implementation of L-JEPA from https://github.com/rbalestr-lab/lejepa

Key features:
- SIGReg loss (Sketched Isotropic Gaussian Regularization)
- Single hyperparameter (λ)
- No stop-gradient, no teacher-student networks, no LR schedulers needed
- Multi-crop augmentation: 2 global views (224×224) + N local views (98×98)

Loss = λ * SIGReg + (1 - λ) * Invariance

Where:
- SIGReg = ||Σ - I||²_F (covariance regularization)
- Invariance = -mean(z1 · z2 / (||z1|| * ||z2||)) (cosine similarity)

Reference: https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md
"""

import os
import pickle
import argparse
from pathlib import Path
from functools import partial
from typing import Tuple, Dict, Any, Optional, Sequence
import time

import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np

# Try to import wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not installed. Logging disabled.")


# =============================================================================
# Data Augmentation (Multi-crop)
# =============================================================================

def random_crop(key: jax.Array, image: jnp.ndarray, crop_size: int) -> jnp.ndarray:
    """Random crop from image.

    Args:
        key: PRNG key
        image: (H, W, C) image
        crop_size: Size of the square crop

    Returns:
        (crop_size, crop_size, C) cropped image
    """
    H, W, C = image.shape
    key1, key2 = random.split(key)

    max_h = H - crop_size
    max_w = W - crop_size

    h_start = random.randint(key1, (), 0, max_h + 1)
    w_start = random.randint(key2, (), 0, max_w + 1)

    crop = jax.lax.dynamic_slice(
        image,
        (h_start, w_start, 0),
        (crop_size, crop_size, C)
    )
    return crop


def center_crop(image: jnp.ndarray, crop_size: int) -> jnp.ndarray:
    """Center crop from image.

    Args:
        image: (H, W, C) image
        crop_size: Size of the square crop

    Returns:
        (crop_size, crop_size, C) cropped image
    """
    H, W, C = image.shape
    h_start = (H - crop_size) // 2
    w_start = (W - crop_size) // 2
    return image[h_start:h_start + crop_size, w_start:w_start + crop_size, :]


def resize_image(image: jnp.ndarray, target_size: int) -> jnp.ndarray:
    """Resize image using bilinear interpolation.

    Args:
        image: (H, W, C) image
        target_size: Target height/width

    Returns:
        (target_size, target_size, C) resized image
    """
    return jax.image.resize(
        image,
        (target_size, target_size, image.shape[-1]),
        method='bilinear'
    )


def random_horizontal_flip(key: jax.Array, image: jnp.ndarray) -> jnp.ndarray:
    """Random horizontal flip.

    Args:
        key: PRNG key
        image: (H, W, C) image

    Returns:
        Possibly flipped image
    """
    flip = random.bernoulli(key, 0.5)
    return jax.lax.cond(flip, lambda x: jnp.flip(x, axis=1), lambda x: x, image)


def color_jitter(
    key: jax.Array,
    image: jnp.ndarray,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.2,
    hue: float = 0.1
) -> jnp.ndarray:
    """Apply color jitter augmentation.

    Args:
        key: PRNG key
        image: (H, W, C) image in [0, 1] range
        brightness, contrast, saturation, hue: jitter parameters

    Returns:
        Augmented image
    """
    keys = random.split(key, 4)

    # Brightness
    b_factor = 1 + random.uniform(keys[0], (), minval=-brightness, maxval=brightness)
    image = image * b_factor

    # Contrast
    c_factor = 1 + random.uniform(keys[1], (), minval=-contrast, maxval=contrast)
    mean = jnp.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * c_factor + mean

    # Saturation (convert to grayscale and blend)
    s_factor = 1 + random.uniform(keys[2], (), minval=-saturation, maxval=saturation)
    gray = jnp.mean(image, axis=-1, keepdims=True)
    image = gray + s_factor * (image - gray)

    # Clamp to valid range
    image = jnp.clip(image, 0, 1)

    return image


def gaussian_blur(key: jax.Array, image: jnp.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> jnp.ndarray:
    """Apply Gaussian blur.

    Args:
        key: PRNG key
        image: (H, W, C) image
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation

    Returns:
        Blurred image
    """
    # Create Gaussian kernel
    x = jnp.arange(kernel_size) - kernel_size // 2
    kernel_1d = jnp.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = jnp.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d[:, :, None, None]

    # Apply per channel
    H, W, C = image.shape
    image = image.transpose(2, 0, 1)[None, ...]  # (1, C, H, W)

    # Pad and convolve
    pad = kernel_size // 2
    image = jnp.pad(image, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')

    # Simple convolution using lax.conv_general_dilated
    output = jax.lax.conv_general_dilated(
        image,
        jnp.tile(kernel_2d, (1, 1, C, 1)).transpose(2, 3, 0, 1),  # (C, 1, k, k)
        window_strides=(1, 1),
        padding='VALID',
        feature_group_count=C
    )

    return output[0].transpose(1, 2, 0)  # (H, W, C)


def multi_crop_augment(
    key: jax.Array,
    image: jnp.ndarray,
    global_size: int = 224,
    local_size: int = 98,
    num_global: int = 2,
    num_local: int = 6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Multi-crop augmentation for L-JEPA.

    Args:
        key: PRNG key
        image: (H, W, C) input image
        global_size: Size of global crops
        local_size: Size of local crops
        num_global: Number of global crops
        num_local: Number of local crops

    Returns:
        global_crops: (num_global, global_size, global_size, C)
        local_crops: (num_local, local_size, local_size, C)
    """
    keys = random.split(key, num_global + num_local + 1)

    # Global crops (larger coverage)
    global_crops = []
    for i in range(num_global):
        k1, k2, k3 = random.split(keys[i], 3)
        crop = random_crop(k1, image, global_size)
        crop = random_horizontal_flip(k2, crop)
        crop = color_jitter(k3, crop)
        global_crops.append(crop)

    # Local crops (smaller, focused regions)
    local_crops = []
    for i in range(num_local):
        k1, k2, k3 = random.split(keys[num_global + i], 3)
        crop = random_crop(k1, image, local_size)
        crop = random_horizontal_flip(k2, crop)
        crop = color_jitter(k3, crop)
        local_crops.append(crop)

    # Handle case when num_local=0
    if num_local > 0:
        return jnp.stack(global_crops), jnp.stack(local_crops)
    else:
        return jnp.stack(global_crops), None


# =============================================================================
# SIGReg Loss (L-JEPA Core)
# =============================================================================

def sigreg_loss(z: jnp.ndarray, lambda_: float = 0.5) -> jnp.ndarray:
    """SIGReg loss: Sketched Isotropic Gaussian Regularization.

    Loss = λ * ||Σ - I||²_F + (1 - λ) * Invariance

    Where:
    - Σ is the covariance matrix of embeddings
    - Invariance is negative mean cosine similarity between views

    Args:
        z: (B, num_views, D) normalized embeddings from different views
        lambda_: Balance between SIGReg and invariance

    Returns:
        Scalar loss
    """
    B, V, D = z.shape

    # Flatten all views for covariance computation
    z_flat = z.reshape(B * V, D)  # (B*V, D)

    # Center the embeddings
    z_centered = z_flat - jnp.mean(z_flat, axis=0, keepdims=True)

    # Compute covariance matrix
    cov = (z_centered.T @ z_centered) / (B * V - 1)  # (D, D)

    # SIGReg: ||Σ - I||²_F (Frobenius norm of difference from identity)
    identity = jnp.eye(D)
    sigreg = jnp.sum((cov - identity) ** 2)

    # Invariance: negative mean cosine similarity between all pairs of views
    # For each sample, compute cosine similarity between view pairs
    # z: (B, V, D)
    # Normalize (should already be normalized, but ensure)
    z_norm = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    # Compute pairwise cosine similarities
    # (B, V, D) @ (B, D, V) -> (B, V, V)
    sim_matrix = jnp.einsum('bvd,bwd->bvw', z_norm, z_norm)

    # Mean of off-diagonal elements (exclude self-similarity)
    mask = 1 - jnp.eye(V)
    invariance = -jnp.sum(sim_matrix * mask[None, :, :]) / (B * V * (V - 1))

    # Combined loss
    loss = lambda_ * sigreg + (1 - lambda_) * invariance

    return loss, sigreg, invariance


# =============================================================================
# Encoder Wrapper for L-JEPA
# =============================================================================

class LEJEPAEncoder(nn.Module):
    """Encoder wrapper for L-JEPA.

    Wraps a backbone (e.g., ConvSSM) and adds a projection head.
    """
    backbone: nn.Module
    projection_dim: int = 256
    hidden_dim: int = 2048
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, H, W, C) input image

        Returns:
            (B, projection_dim) L2-normalized embedding
        """
        # Get backbone features
        # Backbone should output (B, feature_dim)
        features = self.backbone(x, train=train)

        # If backbone outputs logits, we need to get the representation before the head
        # For now, assume backbone can be modified to return features

        # Projection head (MLP)
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(features)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(self.projection_dim, dtype=self.dtype)(x)

        # L2 normalize
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

        return x


class ConvSSMBackbone(nn.Module):
    """ConvSSM backbone that outputs features instead of logits.

    This is a wrapper that extracts features before the classification head.
    """
    base_model_fn: Any  # Function to create base model
    num_classes: int = 0  # Dummy, will be ignored

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Import here to avoid circular imports
        from flashfftconv.convnext_3d_pure_ssm import PureConvNeXt3DSSM, LayerNorm3D

        B, H, W, C_in = x.shape
        T = 8  # Default temporal depth

        # Repeat input T times
        x = jnp.tile(x[:, None, ...], (1, T, 1, 1, 1))

        # Stem
        x = x.reshape(B * T, H, W, C_in)
        x = nn.Conv(96, kernel_size=(4, 4), strides=(4, 4), padding='VALID', name='stem')(x)
        _, H_stem, W_stem, _ = x.shape
        x = x.reshape(B, T, H_stem, W_stem, 96)
        x = LayerNorm3D(name='stem_norm')(x)

        # Stages (simplified - reuse PureConvNeXt3DSSM internals)
        dims = (96, 192, 384, 768)
        depths = (3, 3, 9, 3)

        # For simplicity, just do global pooling after stem for now
        # A full implementation would include all stages

        # Global pooling
        x = jnp.mean(x, axis=(1, 2, 3))  # (B, C)

        # Final norm
        x = LayerNorm3D(name='head_norm')(x[..., None, None, None, :])
        x = x.squeeze(axis=(1, 2, 3))

        return x


# =============================================================================
# Simple ConvNeXt Backbone (2D, no SSM) for faster prototyping
# =============================================================================

class SimpleConvNeXtBlock(nn.Module):
    """Simple ConvNeXt block without SSM."""
    dim: int
    expansion_ratio: int = 4
    layer_scale_init: float = 1e-6
    drop_path_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        B, H, W, C = x.shape

        # Depthwise conv
        x = nn.Conv(C, kernel_size=(7, 7), feature_group_count=C, padding='SAME', dtype=self.dtype)(x)

        # LayerNorm
        x = nn.LayerNorm(dtype=self.dtype)(x)

        # MLP
        hidden_dim = int(C * self.expansion_ratio)
        x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(C, dtype=self.dtype)(x)

        # Layer scale
        gamma = self.param('layer_scale', nn.initializers.constant(self.layer_scale_init), (C,))
        x = x * gamma

        # Drop path
        if train and self.drop_path_rate > 0:
            keep_prob = 1.0 - self.drop_path_rate
            rng = self.make_rng('dropout')
            mask = random.bernoulli(rng, keep_prob, (B, 1, 1, 1))
            x = x / keep_prob * mask

        return x + residual


class SimpleConvNeXtBackbone(nn.Module):
    """Simple ConvNeXt backbone for L-JEPA.

    Outputs feature embeddings instead of classification logits.
    """
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    drop_path_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        B, H, W, C_in = x.shape

        # Stem
        x = nn.Conv(self.dims[0], kernel_size=(4, 4), strides=(4, 4), padding='VALID',
                    dtype=self.dtype, name='stem')(x)
        x = nn.LayerNorm(dtype=self.dtype, name='stem_norm')(x)

        # Stochastic depth schedule
        total_blocks = sum(self.depths)
        dpr = [self.drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0

        # Stages
        for stage_idx, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            # Downsample
            if stage_idx > 0:
                x = nn.LayerNorm(dtype=self.dtype, name=f'downsample_norm_{stage_idx}')(x)
                x = nn.Conv(dim, kernel_size=(2, 2), strides=(2, 2), padding='VALID',
                            dtype=self.dtype, name=f'downsample_{stage_idx}')(x)

            # Blocks
            for block_i in range(depth):
                x = SimpleConvNeXtBlock(
                    dim=dim,
                    drop_path_rate=dpr[block_idx],
                    dtype=self.dtype,
                    name=f'stage_{stage_idx}_block_{block_i}'
                )(x, train=train)
                block_idx += 1

        # Global pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, C)

        # Final norm
        x = nn.LayerNorm(dtype=self.dtype, name='head_norm')(x)

        return x


class LEJEPAModel(nn.Module):
    """Full L-JEPA model with backbone and projection head."""
    backbone_type: str = 'convnext'  # 'convnext', 'convssm', 'gated_convssm', 'gated_convssm_v2'
    projection_dim: int = 256
    hidden_dim: int = 2048
    T: int = 8  # For ConvSSM
    kernel_size: int = 7
    kernel_size_t: int = 5
    num_basis: int = 4  # For gated_convssm_v2
    gating_mode: str = 'both'  # For gated_convssm_v2: 'coefficient', 'kernel_attention', 'both'
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.backbone_type == 'convnext':
            self.backbone = SimpleConvNeXtBackbone(dtype=self.dtype)
        elif self.backbone_type == 'convssm':
            from flashfftconv.convnext_3d_pure_ssm import pure_convnext_3d_ssm_tiny
            # Create without classification head
            self.backbone = PureConvSSMBackbone(
                T=self.T,
                kernel_size=self.kernel_size,
                kernel_size_t=self.kernel_size_t,
                dtype=self.dtype
            )
        elif self.backbone_type == 'gated_convssm':
            self.backbone = GatedConvSSMBackbone(
                T=self.T,
                kernel_size=self.kernel_size,
                kernel_size_t=self.kernel_size_t,
                dtype=self.dtype
            )
        elif self.backbone_type == 'gated_convssm_v2':
            self.backbone = GatedConvSSMV2Backbone(
                T=self.T,
                kernel_size=self.kernel_size,
                kernel_size_t=self.kernel_size_t,
                num_basis=self.num_basis,
                gating_mode=self.gating_mode,
                dtype=self.dtype
            )

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, H, W, C) input image

        Returns:
            (B, projection_dim) L2-normalized embedding
        """
        # Get backbone features
        features = self.backbone(x, train=train)

        # Projection head
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(features)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(self.projection_dim, dtype=self.dtype)(x)

        # L2 normalize
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

        return x


# =============================================================================
# Pure ConvSSM Backbone for L-JEPA
# =============================================================================

class PureConvSSMBackbone(nn.Module):
    """Pure ConvSSM backbone that outputs features.

    Based on PureConvNeXt3DSSM but without classification head.
    """
    T: int = 8
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    kernel_size: int = 7
    kernel_size_t: int = 5
    drop_path_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        from flashfftconv.convnext_3d_pure_ssm import (
            PureConvNeXt3DSSMBlock, Downsample3D, LayerNorm3D
        )

        B, H, W, C_in = x.shape

        # Repeat input T times
        x = jnp.tile(x[:, None, ...], (1, self.T, 1, 1, 1))

        # Stem
        x = x.reshape(B * self.T, H, W, C_in)
        x = nn.Conv(self.dims[0], kernel_size=(4, 4), strides=(4, 4),
                    padding='VALID', dtype=self.dtype, name='stem')(x)
        _, H_stem, W_stem, _ = x.shape
        x = x.reshape(B, self.T, H_stem, W_stem, self.dims[0])
        x = LayerNorm3D(dtype=self.dtype, name='stem_norm')(x)

        # Stochastic depth schedule
        total_blocks = sum(self.depths)
        dpr = [self.drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0

        # Stages
        for stage_idx, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            if stage_idx > 0:
                x = Downsample3D(dim, dtype=self.dtype, name=f'downsample_{stage_idx}')(x)

            for block_i in range(depth):
                x = PureConvNeXt3DSSMBlock(
                    dim=dim,
                    kernel_size=self.kernel_size,
                    kernel_size_t=self.kernel_size_t,
                    drop_path_rate=dpr[block_idx],
                    dtype=self.dtype,
                    name=f'stage_{stage_idx}_block_{block_i}'
                )(x, train=train)
                block_idx += 1

        # Global pooling
        x = jnp.mean(x, axis=(1, 2, 3))  # (B, C)

        # Final norm
        x = LayerNorm3D(dtype=self.dtype, name='head_norm')(x[..., None, None, None, :])
        x = x.squeeze(axis=(1, 2, 3))

        return x


# =============================================================================
# Gated ConvSSM Backbone for L-JEPA
# =============================================================================

class GatedConvSSMBackbone(nn.Module):
    """Gated ConvSSM backbone (minGRU-style) that outputs features."""
    T: int = 8
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    kernel_size: int = 7
    kernel_size_t: int = 5
    drop_path_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        from flashfftconv.convnext_gated_convssm import (
            GatedConvNeXt3DBlock, Downsample3D, LayerNorm3D
        )

        B, H, W, C_in = x.shape

        # Repeat input T times
        x = jnp.tile(x[:, None, ...], (1, self.T, 1, 1, 1))

        # Stem
        x = x.reshape(B * self.T, H, W, C_in)
        x = nn.Conv(self.dims[0], kernel_size=(4, 4), strides=(4, 4),
                    padding='VALID', dtype=self.dtype, name='stem')(x)
        _, H_stem, W_stem, _ = x.shape
        x = x.reshape(B, self.T, H_stem, W_stem, self.dims[0])
        x = LayerNorm3D(dtype=self.dtype, name='stem_norm')(x)

        # Stochastic depth schedule
        total_blocks = sum(self.depths)
        dpr = [self.drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0

        # Stages
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

        # Global pooling
        x = jnp.mean(x, axis=(1, 2, 3))  # (B, C)

        # Final norm
        x = LayerNorm3D(dtype=self.dtype, name='head_norm')(x[..., None, None, None, :])
        x = x.squeeze(axis=(1, 2, 3))

        return x


# =============================================================================
# Gated ConvSSM V2 Backbone for L-JEPA (Input-dependent kernel gating)
# =============================================================================

class GatedConvSSMV2Backbone(nn.Module):
    """Gated ConvSSM V2 backbone (input-dependent kernel gating, 2D) that outputs features.

    This uses the 2D implementation from convnext_gated_convssm_v2.py which applies
    SSM iterations within each block (not along a temporal dimension).
    """
    T: int = 8  # num_iterations for the SSM within each block
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    kernel_size: int = 7
    kernel_size_t: int = 5  # Unused in 2D version, kept for API compatibility
    num_basis: int = 4
    gating_mode: str = 'both'  # 'coefficient', 'kernel_attention', 'both'
    drop_path_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        from flashfftconv.convnext_gated_convssm_v2 import (
            GatedConvNeXtBlock_V2, LayerNorm2D
        )

        B, H, W, C_in = x.shape

        # Stem (4x4 patch embedding)
        x = nn.Conv(self.dims[0], kernel_size=(4, 4), strides=(4, 4),
                    padding='VALID', dtype=self.dtype, name='stem')(x)
        x = LayerNorm2D(dtype=self.dtype, name='stem_norm')(x)

        # Stochastic depth schedule
        total_blocks = sum(self.depths)
        dpr = [self.drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0

        # Stages
        for stage_idx, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            # Downsample between stages
            if stage_idx > 0:
                x = LayerNorm2D(dtype=self.dtype, name=f'downsample_norm_{stage_idx}')(x)
                x = nn.Conv(dim, kernel_size=(2, 2), strides=(2, 2),
                           padding='VALID', dtype=self.dtype, name=f'downsample_{stage_idx}')(x)

            # Blocks in this stage
            for block_i in range(depth):
                x = GatedConvNeXtBlock_V2(
                    dim=dim,
                    kernel_size=self.kernel_size,
                    num_iterations=self.T,  # T controls SSM iterations per block
                    num_basis=self.num_basis,
                    gating_mode=self.gating_mode,
                    drop_path_rate=dpr[block_idx],
                    dtype=self.dtype,
                    name=f'stage_{stage_idx}_block_{block_i}'
                )(x, train=train)
                block_idx += 1

        # Global pooling (B, H, W, C) -> (B, C)
        x = jnp.mean(x, axis=(1, 2))

        # Final norm (already 2D: B, C)
        x = LayerNorm2D(dtype=self.dtype, name='head_norm')(x)

        return x


# =============================================================================
# Training State and Functions
# =============================================================================

class TrainState(train_state.TrainState):
    """Training state with dropout key."""
    key: jax.Array


def create_train_state(
    key: jax.Array,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.05,
    input_shape: Tuple[int, ...] = (1, 224, 224, 3),
) -> TrainState:
    """Create training state."""
    key, init_key, dropout_key = random.split(key, 3)

    dummy_input = jnp.ones(input_shape)
    variables = model.init({'params': init_key, 'dropout': dropout_key}, dummy_input, train=False)
    params = variables['params']

    # AdamW optimizer
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        key=dropout_key,
    )


@partial(jax.jit, static_argnums=(3,))
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    key: jax.Array,
    lambda_: float = 0.5,
) -> Tuple[TrainState, Dict[str, float]]:
    """Single training step.

    Args:
        state: Training state
        batch: Dict with 'global_crops' (B, 2, H, W, C) and 'local_crops' (B, N, h, w, C)
        key: PRNG key
        lambda_: SIGReg loss weight

    Returns:
        Updated state and metrics dict
    """
    global_crops = batch['global_crops']  # (B, num_global, H, W, C)
    local_crops = batch.get('local_crops')  # (B, num_local, h, w, C) or None

    B, num_global, H, W, C = global_crops.shape

    def loss_fn(params):
        # Get embeddings for all views
        embeddings = []

        # Process global crops
        for i in range(num_global):
            z = state.apply_fn(
                {'params': params},
                global_crops[:, i],
                train=True,
                rngs={'dropout': random.fold_in(key, i)}
            )
            embeddings.append(z)

        # Process local crops if provided
        if local_crops is not None:
            _, num_local, h, w, _ = local_crops.shape
            for i in range(num_local):
                # Resize local crops to global size for the backbone
                local_crop_i = local_crops[:, i]
                local_crop_i = jax.vmap(lambda x: resize_image(x, H))(local_crop_i)

                z = state.apply_fn(
                    {'params': params},
                    local_crop_i,
                    train=True,
                    rngs={'dropout': random.fold_in(key, num_global + i)}
                )
                embeddings.append(z)

        # Stack all embeddings: (B, num_views, D)
        z = jnp.stack(embeddings, axis=1)

        # Compute SIGReg loss
        loss, sigreg, invariance = sigreg_loss(z, lambda_)

        return loss, {'sigreg': sigreg, 'invariance': invariance}

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Update parameters
    state = state.apply_gradients(grads=grads)

    metrics = {
        'loss': loss,
        'sigreg': aux['sigreg'],
        'invariance': aux['invariance'],
    }

    return state, metrics


# =============================================================================
# Dataset
# =============================================================================

class ImageNetteDataset:
    """Simple Imagenette dataset loader."""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        global_size: int = 224,
        local_size: int = 98,
        num_global: int = 2,
        num_local: int = 6,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.global_size = global_size
        self.local_size = local_size
        self.num_global = num_global
        self.num_local = num_local

        # Load image paths
        split_dir = self.data_dir / split
        self.image_paths = list(split_dir.rglob('*.JPEG'))
        print(f"Found {len(self.image_paths)} images in {split_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img) / 255.0  # Normalize to [0, 1]

        return {'image': img.astype(np.float32)}


def collate_fn(batch, key, global_size=224, local_size=98, num_global=2, num_local=6):
    """Collate function with multi-crop augmentation."""
    images = [item['image'] for item in batch]
    B = len(images)

    # Split key for each sample
    keys = random.split(key, B)

    global_crops = []
    local_crops = []

    for i, (img, k) in enumerate(zip(images, keys)):
        # Resize image if too small
        H, W = img.shape[:2]
        min_size = max(global_size, local_size) + 32  # Some padding
        if min(H, W) < min_size:
            scale = min_size / min(H, W)
            new_H, new_W = int(H * scale), int(W * scale)
            img = jax.image.resize(img, (new_H, new_W, 3), method='bilinear')

        # Multi-crop augment
        g, l = multi_crop_augment(
            k, img,
            global_size=global_size,
            local_size=local_size,
            num_global=num_global,
            num_local=num_local
        )
        global_crops.append(g)
        if l is not None:
            local_crops.append(l)

    result = {
        'global_crops': jnp.stack(global_crops),  # (B, num_global, H, W, C)
    }
    # Only include local_crops if they exist
    if local_crops:
        result['local_crops'] = jnp.stack(local_crops)  # (B, num_local, h, w, C)
    else:
        result['local_crops'] = None
    return result


# =============================================================================
# Main Training Loop
# =============================================================================

def train_lejepa(
    data_dir: str,
    output_dir: str,
    backbone_type: str = 'convnext',
    projection_dim: int = 256,
    hidden_dim: int = 2048,
    lambda_: float = 0.5,
    T: int = 8,
    kernel_size: int = 7,
    kernel_size_t: int = 5,
    gating_mode: str = 'both',
    num_basis: int = 4,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.05,
    global_size: int = 224,
    local_size: int = 98,
    num_global: int = 2,
    num_local: int = 0,  # Start without local crops for simplicity
    seed: int = 42,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
):
    """Train L-JEPA model."""
    print("=" * 70)
    print("L-JEPA Training (JAX)")
    print("=" * 70)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if HAS_WANDB and wandb_project:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                'backbone_type': backbone_type,
                'projection_dim': projection_dim,
                'hidden_dim': hidden_dim,
                'lambda': lambda_,
                'T': T,
                'kernel_size': kernel_size,
                'kernel_size_t': kernel_size_t,
                'gating_mode': gating_mode,
                'num_basis': num_basis,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'global_size': global_size,
                'local_size': local_size,
                'num_global': num_global,
                'num_local': num_local,
            }
        )

    # Initialize RNG
    key = random.PRNGKey(seed)
    key, init_key, data_key = random.split(key, 3)

    # Create model
    print(f"\nCreating model: backbone={backbone_type}")
    if backbone_type == 'gated_convssm_v2':
        print(f"  gating_mode={gating_mode}, num_basis={num_basis}")
    model = LEJEPAModel(
        backbone_type=backbone_type,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
        T=T,
        kernel_size=kernel_size,
        kernel_size_t=kernel_size_t,
        num_basis=num_basis,
        gating_mode=gating_mode,
    )

    # Create training state
    state = create_train_state(
        init_key,
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        input_shape=(1, global_size, global_size, 3),
    )

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Load dataset
    print(f"\nLoading dataset from {data_dir}")
    dataset = ImageNetteDataset(
        data_dir,
        split='train',
        global_size=global_size,
        local_size=local_size,
        num_global=num_global,
        num_local=num_local,
    )

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    steps_per_epoch = len(dataset) // batch_size

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_sigreg = 0.0
        epoch_invariance = 0.0
        num_steps = 0

        # Shuffle indices
        key, shuffle_key = random.split(key)
        indices = random.permutation(shuffle_key, len(dataset))

        for step in range(steps_per_epoch):
            # Get batch
            batch_indices = indices[step * batch_size:(step + 1) * batch_size]
            batch_items = [dataset[int(i)] for i in batch_indices]

            # Collate with augmentation
            key, collate_key, step_key = random.split(key, 3)
            batch = collate_fn(
                batch_items, collate_key,
                global_size=global_size,
                local_size=local_size,
                num_global=num_global,
                num_local=num_local,
            )

            # Training step
            state, metrics = train_step(state, batch, step_key, lambda_)

            epoch_loss += float(metrics['loss'])
            epoch_sigreg += float(metrics['sigreg'])
            epoch_invariance += float(metrics['invariance'])
            num_steps += 1

        # Epoch stats
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_steps
        avg_sigreg = epoch_sigreg / num_steps
        avg_invariance = epoch_invariance / num_steps

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"SIGReg: {avg_sigreg:.4f} | "
              f"Invariance: {avg_invariance:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Log to wandb
        if HAS_WANDB and wandb_project:
            wandb.log({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'sigreg': avg_sigreg,
                'invariance': avg_invariance,
                'epoch_time': epoch_time,
            })

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f'checkpoint_epoch_{epoch+1}.pkl'
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'params': state.params, 'epoch': epoch + 1}, f)
            print(f"  Saved checkpoint to {ckpt_path}")

    # Save final model
    final_path = output_dir / 'final_model.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump({'params': state.params, 'epoch': epochs}, f)
    print(f"\nSaved final model to {final_path}")

    if HAS_WANDB and wandb_project:
        wandb.finish()

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='L-JEPA Training (JAX)')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data/imagenette2-320',
                        help='Path to ImageNet/Imagenette dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_lejepa',
                        help='Output directory for checkpoints')

    # Model
    parser.add_argument('--backbone_type', type=str, default='convnext',
                        choices=['convnext', 'convssm', 'gated_convssm', 'gated_convssm_v2'],
                        help='Backbone architecture')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='Projection head output dimension')
    parser.add_argument('--hidden_dim', type=int, default=2048,
                        help='Projection head hidden dimension')
    parser.add_argument('--T', type=int, default=8,
                        help='Temporal depth for ConvSSM')
    parser.add_argument('--kernel_size', type=int, default=7,
                        help='Spatial kernel size for ConvSSM')
    parser.add_argument('--kernel_size_t', type=int, default=5,
                        help='Temporal kernel size for ConvSSM')
    parser.add_argument('--gating_mode', type=str, default='both',
                        choices=['coefficient', 'kernel_attention', 'both'],
                        help='Gating mode for gated_convssm_v2')
    parser.add_argument('--num_basis', type=int, default=4,
                        help='Number of basis kernels for kernel_attention gating mode')

    # L-JEPA
    parser.add_argument('--lambda_', type=float, default=0.5,
                        help='SIGReg loss weight (lambda)')
    parser.add_argument('--global_size', type=int, default=224,
                        help='Global crop size')
    parser.add_argument('--local_size', type=int, default=98,
                        help='Local crop size')
    parser.add_argument('--num_global', type=int, default=2,
                        help='Number of global crops')
    parser.add_argument('--num_local', type=int, default=0,
                        help='Number of local crops')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)

    # Logging
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()

    train_lejepa(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backbone_type=args.backbone_type,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        lambda_=args.lambda_,
        T=args.T,
        kernel_size=args.kernel_size,
        kernel_size_t=args.kernel_size_t,
        gating_mode=args.gating_mode,
        num_basis=args.num_basis,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        global_size=args.global_size,
        local_size=args.local_size,
        num_global=args.num_global,
        num_local=args.num_local,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == '__main__':
    main()
