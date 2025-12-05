#!/usr/bin/env python3
"""
Train ConvNeXt-SSM on Imagenette with multi-GPU support.

Usage:
    python -m flashfftconv.train_imagenette --data_dir ./data/imagenette2-320

Uses JAX pmap for data-parallel training across all available GPUs.
"""

import os
import time
import argparse
from typing import Dict, Any, Tuple, List
from functools import partial
from pathlib import Path

import wandb
import jax
import jax.numpy as jnp
from jax import random, pmap, local_device_count
import numpy as np
import optax
import flax
from flax.training import train_state, checkpoints
from flax import jax_utils

from flashfftconv.convnext_ssm import (
    ConvNeXtSSM,
    convnext_ssm_tiny,
    convnext_ssm_small,
    convnext_ssm_base,
    count_params,
)

# Pre-FFT Fourier model (no FFT in forward pass!)
from flashfftconv.convnext_fourier import convnext_fourier_prefft_tiny

# V2 Fourier model (all-FFT, stable real/imag representation)
from flashfftconv.convnext_fourier_v2 import (
    convnext_fourier_v2_tiny,
    convnext_fourier_v2_debug,
    convnext_fourier_v2_tiny_bf16,
    convnext_fourier_v2_debug_bf16,
)

# V3 Fourier model (parameter-efficient: same params as ConvNeXt, all-FFT)
from flashfftconv.convnext_fourier_v2 import (
    convnext_fourier_v3_tiny,
    convnext_fourier_v3_debug,
    convnext_fourier_v3_tiny_bf16,
    convnext_fourier_v3_debug_bf16,
)

# Pure Fourier model (no SSM - exact equivalent to ConvNeXt, just in FFT domain)
from flashfftconv.convnext_fourier_v2 import (
    convnext_fourier_pure_tiny,
    convnext_fourier_pure_debug,
)

# Standard ConvNeXt baseline (no SSM, no FFT)
from flashfftconv.convnext_baseline import (
    convnext_tiny,
    convnext_small,
    convnext_base,
)

# Simple FFT ConvNeXt (same as baseline, but depthwise conv uses FFT)
from flashfftconv.convnext_fft_simple import convnext_fft_tiny

# FFT ConvNeXt + ConvSSM (FFT-Simple + SSM blocks)
from flashfftconv.convnext_fft_ssm import convnext_fft_ssm_tiny

# Spatial ConvNeXt + ConvSSM (standard nn.Conv + SSM blocks)
from flashfftconv.convnext_spatial_ssm import convnext_spatial_ssm_tiny

# Parallel ConvSSM using associative_scan (O(log T) parallel, much faster!)
from flashfftconv.convnext_fft_ssm_parallel import convnext_parallel_ssm_tiny

# Gated ConvSSM (Mamba2-style input-dependent gates with parallel scan)
from flashfftconv.convnext_gated_ssm import convnext_gated_ssm_tiny

# Gated ConvSSM V2 (input-dependent kernel gating with parallel scan)
from flashfftconv.convnext_gated_convssm_v2 import gated_convnext_ssm_v2_tiny

# 3D ConvSSM (repeats input T times, uses 3D FFT convolutions)
from flashfftconv.convnext_3d_ssm import convnext_3d_ssm_tiny
from flashfftconv.convnext_3d_pure_ssm import pure_convnext_3d_ssm_tiny

# 3D Mamba-Gated ConvSSM (3D SSM with input-dependent gates)
from flashfftconv.convnext_3d_mamba import convnext_3d_mamba_tiny

# Imagenette class names (10 classes)
IMAGENETTE_CLASSES = [
    'n01440764',  # tench
    'n02102040',  # English springer
    'n02979186',  # cassette player
    'n03000684',  # chain saw
    'n03028079',  # church
    'n03394916',  # French horn
    'n03417042',  # garbage truck
    'n03425413',  # gas pump
    'n03445777',  # golf ball
    'n03888257',  # parachute
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# =============================================================================
# RandAugment Implementation (following timm/torchvision)
# =============================================================================

def apply_autocontrast(img: np.ndarray) -> np.ndarray:
    """Apply autocontrast to image."""
    for c in range(3):
        channel = img[:, :, c]
        lo, hi = channel.min(), channel.max()
        if hi - lo > 1e-5:
            img[:, :, c] = (channel - lo) / (hi - lo)
    return img


def apply_equalize(img: np.ndarray) -> np.ndarray:
    """Apply histogram equalization (approximate with rescaling)."""
    return apply_autocontrast(img)


def apply_invert(img: np.ndarray) -> np.ndarray:
    """Invert image."""
    return 1.0 - img


def apply_rotate(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Rotate image by angle."""
    from PIL import Image
    # Convert to PIL, rotate, convert back
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    angle = magnitude * 30  # Max 30 degrees
    pil_img = pil_img.rotate(angle, Image.BILINEAR, fillcolor=(128, 128, 128))
    return np.array(pil_img).astype(np.float32) / 255.0


def apply_posterize(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Reduce color bits."""
    bits = int(8 - magnitude * 4)  # 8 -> 4 bits
    bits = max(1, bits)
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    img_uint8 = (img_uint8 >> (8 - bits)) << (8 - bits)
    return img_uint8.astype(np.float32) / 255.0


def apply_solarize(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Solarize image."""
    threshold = 1.0 - magnitude
    img = np.where(img > threshold, 1.0 - img, img)
    return img


def apply_sharpness(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Adjust sharpness."""
    from PIL import Image, ImageEnhance
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    factor = 1.0 + magnitude * 0.9  # 1.0 -> 1.9
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(factor)
    return np.array(pil_img).astype(np.float32) / 255.0


def apply_contrast(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Adjust contrast."""
    factor = 1.0 + magnitude * 0.9
    mean = img.mean()
    return (img - mean) * factor + mean


def apply_brightness(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Adjust brightness."""
    factor = 1.0 + magnitude * 0.9
    return img * factor


def apply_color(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Adjust color saturation."""
    from PIL import Image, ImageEnhance
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    factor = 1.0 + magnitude * 0.9
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(factor)
    return np.array(pil_img).astype(np.float32) / 255.0


def apply_shear_x(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Shear horizontally."""
    from PIL import Image
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    shear = magnitude * 0.3  # Max 0.3 shear
    pil_img = pil_img.transform(pil_img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0),
                                 Image.BILINEAR, fillcolor=(128, 128, 128))
    return np.array(pil_img).astype(np.float32) / 255.0


def apply_shear_y(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Shear vertically."""
    from PIL import Image
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    shear = magnitude * 0.3
    pil_img = pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0),
                                 Image.BILINEAR, fillcolor=(128, 128, 128))
    return np.array(pil_img).astype(np.float32) / 255.0


def apply_translate_x(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Translate horizontally."""
    from PIL import Image
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    pixels = int(magnitude * img.shape[1] * 0.45)
    pil_img = pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
                                 Image.BILINEAR, fillcolor=(128, 128, 128))
    return np.array(pil_img).astype(np.float32) / 255.0


def apply_translate_y(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Translate vertically."""
    from PIL import Image
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    pixels = int(magnitude * img.shape[0] * 0.45)
    pil_img = pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
                                 Image.BILINEAR, fillcolor=(128, 128, 128))
    return np.array(pil_img).astype(np.float32) / 255.0


RANDAUGMENT_OPS = [
    ('AutoContrast', apply_autocontrast, False),
    ('Equalize', apply_equalize, False),
    ('Invert', apply_invert, False),
    ('Rotate', apply_rotate, True),
    ('Posterize', apply_posterize, True),
    ('Solarize', apply_solarize, True),
    ('Sharpness', apply_sharpness, True),
    ('Contrast', apply_contrast, True),
    ('Brightness', apply_brightness, True),
    ('Color', apply_color, True),
    ('ShearX', apply_shear_x, True),
    ('ShearY', apply_shear_y, True),
    ('TranslateX', apply_translate_x, True),
    ('TranslateY', apply_translate_y, True),
]


def randaugment(img: np.ndarray, n: int = 2, m: int = 9) -> np.ndarray:
    """
    Apply RandAugment to image.

    Args:
        img: Image array (H, W, 3), values in [0, 1]
        n: Number of augmentations to apply
        m: Magnitude (0-10 scale)

    Returns:
        Augmented image
    """
    magnitude = m / 10.0

    # Randomly select n operations
    ops = np.random.choice(len(RANDAUGMENT_OPS), n, replace=False)

    for op_idx in ops:
        name, fn, use_magnitude = RANDAUGMENT_OPS[op_idx]
        if np.random.random() > 0.5:  # 50% chance to apply with negation
            if use_magnitude:
                img = fn(img, magnitude)
            else:
                img = fn(img)

    return np.clip(img, 0, 1)


# =============================================================================
# Mixup and CutMix (applied at batch level)
# =============================================================================

def mixup_batch(images: np.ndarray, labels: np.ndarray, alpha: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Mixup to a batch.

    Args:
        images: (B, H, W, C) batch of images
        labels: (B,) integer labels
        alpha: Mixup alpha parameter

    Returns:
        Mixed images and soft labels (B, num_classes)
    """
    batch_size = images.shape[0]

    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    # Random permutation for mixing
    perm = np.random.permutation(batch_size)

    # Mix images
    mixed_images = lam * images + (1 - lam) * images[perm]

    # Return mixed images with (original_label, mixed_label, lambda)
    return mixed_images, (labels, labels[perm], lam)


def cutmix_batch(images: np.ndarray, labels: np.ndarray, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply CutMix to a batch.

    Args:
        images: (B, H, W, C) batch of images
        labels: (B,) integer labels
        alpha: CutMix alpha parameter

    Returns:
        Mixed images and soft labels info
    """
    batch_size, H, W, C = images.shape

    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    # Random permutation for mixing
    perm = np.random.permutation(batch_size)

    # Compute cut region
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply cut
    mixed_images = images.copy()
    mixed_images[:, y1:y2, x1:x2, :] = images[perm, y1:y2, x1:x2, :]

    # Adjust lambda based on actual cut area
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)

    return mixed_images, (labels, labels[perm], lam)


def apply_mixup_cutmix(
    images: np.ndarray,
    labels: np.ndarray,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 0.5,
    switch_prob: float = 0.5,
) -> Tuple[np.ndarray, Tuple]:
    """
    Apply Mixup or CutMix to batch with probability.

    Args:
        images: Batch of images
        labels: Batch of labels
        mixup_alpha: Mixup alpha
        cutmix_alpha: CutMix alpha
        mix_prob: Probability of applying any mixing
        switch_prob: Probability of CutMix vs Mixup when mixing

    Returns:
        Mixed images and label info tuple
    """
    if np.random.random() > mix_prob:
        # No mixing - return one-hot labels with lambda=1
        return images, (labels, labels, 1.0)

    if np.random.random() < switch_prob and cutmix_alpha > 0:
        return cutmix_batch(images, labels, cutmix_alpha)
    elif mixup_alpha > 0:
        return mixup_batch(images, labels, mixup_alpha)
    else:
        return images, (labels, labels, 1.0)


# =============================================================================
# Data Loading (Pure NumPy/PIL - no TensorFlow dependency issues)
# =============================================================================

def load_image(path: str, size: int = 224) -> np.ndarray:
    """Load and preprocess a single image."""
    from PIL import Image

    img = Image.open(path).convert('RGB')

    # Resize maintaining aspect ratio, then center crop
    w, h = img.size
    scale = size / min(w, h) * 1.15  # Slight over-scale for crop
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    # Convert to array and normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    return arr


def load_image_train(
    path: str,
    size: int = 224,
    use_randaugment: bool = False,
    randaug_n: int = 2,
    randaug_m: int = 9,
) -> np.ndarray:
    """Load image with training augmentation.

    Args:
        path: Path to image
        size: Target size
        use_randaugment: Whether to apply RandAugment
        randaug_n: Number of RandAugment operations
        randaug_m: RandAugment magnitude (0-10)
    """
    from PIL import Image

    img = Image.open(path).convert('RGB')
    w, h = img.size

    # Random resized crop
    scale = np.random.uniform(0.08, 1.0)
    ratio = np.random.uniform(0.75, 1.333)

    crop_w = int(min(w, h) * np.sqrt(scale * ratio))
    crop_h = int(min(w, h) * np.sqrt(scale / ratio))
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    left = np.random.randint(0, w - crop_w + 1)
    top = np.random.randint(0, h - crop_h + 1)

    img = img.crop((left, top, left + crop_w, top + crop_h))
    img = img.resize((size, size), Image.BILINEAR)

    # Random horizontal flip
    if np.random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Convert to array (0-1 range for augmentation)
    arr = np.array(img, dtype=np.float32) / 255.0

    # Apply RandAugment before normalization
    if use_randaugment:
        arr = randaugment(arr, n=randaug_n, m=randaug_m)

    # Normalize
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    return arr


def create_dataset(
    data_dir: str,
    split: str = 'train',
    image_size: int = 224,
    use_randaugment: bool = False,
    randaug_n: int = 2,
    randaug_m: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load entire dataset into memory (Imagenette is small enough).

    Args:
        data_dir: Path to dataset
        split: 'train' or 'val'
        image_size: Target image size
        use_randaugment: Whether to apply RandAugment (train only)
        randaug_n: Number of RandAugment operations
        randaug_m: RandAugment magnitude (0-10)

    Returns:
        images: (N, H, W, 3)
        labels: (N,)
    """
    split_dir = Path(data_dir) / split

    images = []
    labels = []

    class_to_idx = {c: i for i, c in enumerate(IMAGENETTE_CLASSES)}

    for class_name in IMAGENETTE_CLASSES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        for img_path in class_dir.glob('*.JPEG'):
            try:
                if split == 'train':
                    img = load_image_train(
                        str(img_path), image_size,
                        use_randaugment=use_randaugment,
                        randaug_n=randaug_n,
                        randaug_m=randaug_m,
                    )
                else:
                    img = load_image(str(img_path), image_size)
                images.append(img)
                labels.append(class_to_idx[class_name])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

    return np.stack(images), np.array(labels)


class DataLoader:
    """Simple data loader with shuffling and batching."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(images)

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            yield self.images[batch_indices], self.labels[batch_indices]


def precompute_fft(images: np.ndarray) -> np.ndarray:
    """Pre-compute FFT for all images ONCE at load time.

    Args:
        images: (N, H, W, C) NHWC format, float32

    Returns:
        Complex64 array (N, C, H, W) in Fourier domain
    """
    print("  Pre-computing FFT for all images (one-time cost)...")
    # Convert to channels-first
    images_chfirst = np.transpose(images, (0, 3, 1, 2))  # (N, C, H, W)
    # FFT2D - this is done ONCE, not per batch
    # Use complex64 (not complex128) for faster GPU computation and JIT compilation
    images_fft = np.fft.fft2(images_chfirst.astype(np.float32), axes=(-2, -1)).astype(np.complex64)
    print(f"  Done! Shape: {images_fft.shape}, dtype: {images_fft.dtype}")
    return images_fft


class FourierDataLoader:
    """Data loader for pre-FFT'd images.

    Key insight: FFT is computed ONCE at dataset load time using numpy.
    This means:
    1. No FFT in JAX graph (fast JIT compilation)
    2. No FFT during training (fast epochs)
    """

    def __init__(
        self,
        images_fft: np.ndarray,  # Already FFT'd! (N, C, H, W) complex
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.images_fft = images_fft  # (N, C, H, W) complex, already FFT'd
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(images_fft)

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            # Just index into pre-computed FFT - no computation!
            yield self.images_fft[batch_indices], self.labels[batch_indices]


# =============================================================================
# Multi-GPU Training State
# =============================================================================

class TrainState(train_state.TrainState):
    """Extended training state with dropout RNG."""
    dropout_rng: jax.Array


def create_train_state(
    rng: jax.Array,
    model,
    learning_rate: float,
    weight_decay: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    image_size: int = 224,
    fourier_mode: bool = False,
) -> TrainState:
    """Create training state with optimizer."""
    params_rng, dropout_rng = random.split(rng)

    # Initialize model
    if fourier_mode:
        # Fourier model expects complex64 (B, C, H, W) input
        dummy_spatial = np.ones((batch_size, 3, image_size, image_size), dtype=np.float32)
        dummy_fft = np.fft.fft2(dummy_spatial, axes=(-2, -1)).astype(np.complex64)
        dummy_input = jnp.array(dummy_fft)
    else:
        # SSM model expects (B, H, W, C) spatial input
        dummy_input = jnp.ones((batch_size, image_size, image_size, 3))

    variables = model.init({'params': params_rng, 'dropout': dropout_rng}, dummy_input, train=False)
    params = variables['params']

    # Learning rate schedule: warmup + cosine decay
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps - warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps],
    )

    # Optimizer: AdamW with weight decay
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule_fn, weight_decay=weight_decay),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )


# =============================================================================
# Training and Evaluation Steps (pmap-compatible)
# =============================================================================

def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int = 10,
    smoothing: float = 0.1,
) -> jnp.ndarray:
    """Cross-entropy loss with label smoothing."""
    one_hot = jax.nn.one_hot(labels, num_classes)

    # Label smoothing
    one_hot = one_hot * (1 - smoothing) + smoothing / num_classes

    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(one_hot * log_probs, axis=-1).mean()


def cross_entropy_loss_mixup(
    logits: jnp.ndarray,
    labels1: jnp.ndarray,
    labels2: jnp.ndarray,
    lam: float,
    num_classes: int = 10,
    smoothing: float = 0.1,
) -> jnp.ndarray:
    """Cross-entropy loss for mixup/cutmix with soft labels.

    Args:
        logits: Model predictions
        labels1: Original labels
        labels2: Mixed labels
        lam: Mixing coefficient (weight for labels1)
        num_classes: Number of classes
        smoothing: Label smoothing factor
    """
    # Create soft target from both label sets
    one_hot1 = jax.nn.one_hot(labels1, num_classes)
    one_hot2 = jax.nn.one_hot(labels2, num_classes)

    # Apply label smoothing
    one_hot1 = one_hot1 * (1 - smoothing) + smoothing / num_classes
    one_hot2 = one_hot2 * (1 - smoothing) + smoothing / num_classes

    # Mix the labels
    soft_labels = lam * one_hot1 + (1 - lam) * one_hot2

    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(soft_labels * log_probs, axis=-1).mean()


def train_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int = 10,
    smoothing: float = 0.1,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Single training step (will be pmapped)."""
    dropout_rng, new_dropout_rng = random.split(state.dropout_rng)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            images,
            train=True,
            rngs={'dropout': dropout_rng},
        )
        loss = cross_entropy_loss(logits, labels, num_classes, smoothing)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')

    # Accuracy
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    accuracy = jax.lax.pmean(accuracy, axis_name='batch')

    # Update state
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }

    return state, metrics


def train_step_mixup(
    state: TrainState,
    images: jnp.ndarray,
    labels1: jnp.ndarray,
    labels2: jnp.ndarray,
    lam: float,
    num_classes: int = 10,
    smoothing: float = 0.1,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Training step for mixup/cutmix with soft labels."""
    dropout_rng, new_dropout_rng = random.split(state.dropout_rng)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            images,
            train=True,
            rngs={'dropout': dropout_rng},
        )
        loss = cross_entropy_loss_mixup(logits, labels1, labels2, lam, num_classes, smoothing)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')

    # Accuracy (use original labels for accuracy computation)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels1)
    accuracy = jax.lax.pmean(accuracy, axis_name='batch')

    # Update state
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }

    return state, metrics


def eval_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int = 10,
) -> Dict[str, jnp.ndarray]:
    """Single evaluation step (will be pmapped)."""
    logits = state.apply_fn(
        {'params': state.params},
        images,
        train=False,
    )
    loss = cross_entropy_loss(logits, labels, num_classes)

    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)

    # Average across devices
    loss = jax.lax.pmean(loss, axis_name='batch')
    accuracy = jax.lax.pmean(accuracy, axis_name='batch')

    return {
        'loss': loss,
        'accuracy': accuracy,
    }


# =============================================================================
# Training Loop
# =============================================================================

def shard_batch(batch: Tuple[np.ndarray, np.ndarray], num_devices: int):
    """Shard a batch across devices."""
    images, labels = batch
    batch_size = images.shape[0]

    # Reshape to (num_devices, per_device_batch, ...)
    per_device = batch_size // num_devices
    images = images[:num_devices * per_device].reshape(num_devices, per_device, *images.shape[1:])
    labels = labels[:num_devices * per_device].reshape(num_devices, per_device)

    return jnp.array(images), jnp.array(labels)


def train_epoch(
    state: TrainState,
    train_loader: DataLoader,
    p_train_step,
    num_devices: int,
    p_train_step_mixup=None,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 0.5,
) -> Tuple[TrainState, Dict[str, float], Dict[str, float]]:
    """Train for one epoch.

    Args:
        state: Training state
        train_loader: Data loader
        p_train_step: pmap'd training step (standard)
        num_devices: Number of devices
        p_train_step_mixup: pmap'd training step (mixup version)
        mixup_alpha: Mixup alpha (0 = disabled)
        cutmix_alpha: CutMix alpha (0 = disabled)
        mix_prob: Probability of applying mixup/cutmix

    Returns:
        state: Updated training state
        avg_metrics: Average loss/accuracy over epoch
        timing_stats: Step timing statistics (mean, min, max, std in ms)
    """
    train_metrics = []
    step_times = []
    use_mixup = (mixup_alpha > 0 or cutmix_alpha > 0) and p_train_step_mixup is not None

    for images, labels in train_loader:
        step_start = time.time()

        # Apply mixup/cutmix at batch level (NumPy, before JAX)
        if use_mixup:
            images, (labels1, labels2, lam) = apply_mixup_cutmix(
                images, labels,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                mix_prob=mix_prob,
            )
            images, labels1 = shard_batch((images, labels1), num_devices)
            _, labels2 = shard_batch((images, labels2), num_devices)
            state, metrics = p_train_step_mixup(state, images, labels1, labels2, lam)
        else:
            images, labels = shard_batch((images, labels), num_devices)
            state, metrics = p_train_step(state, images, labels)

        # Block until computation is done for accurate timing
        jax.block_until_ready(state.params)
        step_time = (time.time() - step_start) * 1000  # ms
        step_times.append(step_time)
        train_metrics.append(metrics)

    # Average metrics (take from first device since they're synced)
    avg_metrics = {
        k: float(np.mean([m[k][0] for m in train_metrics]))
        for k in train_metrics[0].keys()
    }

    # Step timing statistics
    timing_stats = {
        'step_time_mean_ms': float(np.mean(step_times)),
        'step_time_min_ms': float(np.min(step_times)),
        'step_time_max_ms': float(np.max(step_times)),
        'step_time_std_ms': float(np.std(step_times)),
        'throughput_samples_per_sec': len(train_loader) * train_loader.batch_size / (sum(step_times) / 1000),
    }

    return state, avg_metrics, timing_stats


def evaluate(
    state: TrainState,
    eval_loader: DataLoader,
    p_eval_step,
    num_devices: int,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    eval_metrics = []

    for images, labels in eval_loader:
        # Pad batch if needed
        batch_size = images.shape[0]
        target_size = num_devices * (batch_size // num_devices)
        if target_size == 0:
            target_size = num_devices

        if batch_size < target_size:
            pad_size = target_size - batch_size
            images = np.concatenate([images, np.zeros((pad_size, *images.shape[1:]), dtype=images.dtype)])
            labels = np.concatenate([labels, np.zeros(pad_size, dtype=labels.dtype)])

        images, labels = shard_batch((images[:target_size], labels[:target_size]), num_devices)
        metrics = p_eval_step(state, images, labels)
        eval_metrics.append(metrics)

    # Average metrics
    avg_metrics = {
        k: float(np.mean([m[k][0] for m in eval_metrics]))
        for k in eval_metrics[0].keys()
    }

    return avg_metrics


def get_gpu_memory_usage():
    """Get GPU memory usage in GB for all devices."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            mem_mb = [int(x.strip()) for x in result.stdout.strip().split('\n')]
            return sum(mem_mb) / 1024  # Total GB
    except:
        pass
    return 0.0


def train(args):
    """Main training function."""
    num_devices = local_device_count()
    print(f"Training on {num_devices} devices: {jax.devices()}")

    # Determine model mode
    convnext_mode = args.model_type == 'convnext'  # Standard baseline
    fft_simple_mode = args.model_type == 'fft_simple'  # Simple FFT conv (same as baseline but FFT)
    fft_ssm_mode = args.model_type == 'fft_ssm'  # FFT-Simple + ConvSSM blocks
    spatial_ssm_mode = args.model_type == 'spatial_ssm'  # Spatial Conv + ConvSSM blocks
    parallel_ssm_mode = args.model_type == 'parallel_ssm'  # Parallel ConvSSM (associative_scan)
    gated_ssm_mode = args.model_type == 'gated_ssm'  # Mamba2-style gated parallel ConvSSM
    gated_ssm_v2_mode = args.model_type == 'gated_ssm_v2'  # Input-dependent kernel gating with parallel scan
    ssm_3d_mode = args.model_type == 'ssm_3d'  # 3D ConvSSM (repeats input T times)
    pure_ssm_3d_mode = args.model_type == 'pure_ssm_3d'  # Pure 3D ConvSSM (SSM replaces dwconv)
    mamba_3d_mode = args.model_type == 'mamba_3d'  # 3D Mamba-gated ConvSSM
    fourier_mode = args.model_type == 'fourier'  # Pre-FFT mode
    fourier_v2_mode = args.model_type == 'fourier_v2'  # All-FFT stable mode
    fourier_v2_bf16_mode = args.model_type == 'fourier_v2_bf16'  # All-FFT with bfloat16 SSM
    fourier_v3_mode = args.model_type == 'fourier_v3'  # All-FFT param-efficient (~28M params)
    fourier_v3_bf16_mode = args.model_type == 'fourier_v3_bf16'  # All-FFT param-efficient bf16
    fourier_pure_mode = args.model_type == 'fourier_pure'  # Exact ConvNeXt equiv in FFT domain (no SSM)

    # Create model
    if spatial_ssm_mode:
        print(f"Using ConvNeXt-Spatial-SSM (spatial depthwise conv + ConvSSM blocks, T={args.T})")
        model = convnext_spatial_ssm_tiny(
            num_classes=10,
            T=args.T,
            kernel_size=args.kernel_size,
            drop_path_rate=args.drop_path,
        )
        model_name = f"ConvNeXt-Spatial-SSM-Tiny-T{args.T}"
    elif parallel_ssm_mode:
        print(f"Using ConvNeXt-Parallel-SSM (FFT conv + parallel ConvSSM via associative_scan, T={args.T})")
        model = convnext_parallel_ssm_tiny(
            num_classes=10,
            T=args.T,
            kernel_size=args.kernel_size,
            drop_path_rate=args.drop_path,
        )
        model_name = f"ConvNeXt-Parallel-SSM-Tiny-T{args.T}"
    elif gated_ssm_mode:
        print(f"Using ConvNeXt-Gated-SSM (Mamba2-style gated parallel ConvSSM, T={args.T})")
        model = convnext_gated_ssm_tiny(
            num_classes=10,
            T=args.T,
            kernel_size=args.kernel_size,
            drop_path_rate=args.drop_path,
        )
        model_name = f"ConvNeXt-Gated-SSM-Tiny-T{args.T}"
    elif gated_ssm_v2_mode:
        print(f"Using ConvNeXt-Gated-SSM-V2 (input-dependent kernel gating, mode={args.gating_mode}, T={args.T}, num_basis={args.num_basis})")
        model = gated_convnext_ssm_v2_tiny(
            num_classes=10,
            num_iterations=args.T,
            kernel_size=args.kernel_size,
            num_basis=args.num_basis,
            gating_mode=args.gating_mode,
            drop_path_rate=args.drop_path,
        )
        model_name = f"ConvNeXt-Gated-SSM-V2-Tiny-T{args.T}-{args.gating_mode}"
    elif ssm_3d_mode:
        print(f"Using ConvNeXt-3D-SSM (3D FFT conv 7x7x1 + 3D ConvSSM {args.ssm_kernel_size}x{args.ssm_kernel_size}x{args.ssm_kernel_size_t}, T={args.T})")
        model = convnext_3d_ssm_tiny(
            num_classes=10,
            T=args.T,
            kernel_size=args.kernel_size,  # 7 for depthwise conv (spatial only)
            ssm_kernel_size=args.ssm_kernel_size,
            ssm_kernel_size_t=args.ssm_kernel_size_t,
            drop_path_rate=args.drop_path,
        )
        model_name = f"ConvNeXt-3D-SSM-Tiny-T{args.T}-SSM{args.ssm_kernel_size}x{args.ssm_kernel_size}x{args.ssm_kernel_size_t}"
    elif pure_ssm_3d_mode:
        print(f"Using Pure ConvNeXt-3D-SSM (ConvSSM replaces dwconv, {args.kernel_size}x{args.kernel_size}x{args.ssm_kernel_size_t}, T={args.T})")
        model = pure_convnext_3d_ssm_tiny(
            num_classes=10,
            T=args.T,
            kernel_size=args.kernel_size,  # 7 spatial (replaces dwconv)
            kernel_size_t=args.ssm_kernel_size_t,  # 5 temporal
            drop_path_rate=args.drop_path,
        )
        model_name = f"Pure-ConvNeXt-3D-SSM-Tiny-T{args.T}-K{args.kernel_size}x{args.kernel_size}x{args.ssm_kernel_size_t}"
    elif mamba_3d_mode:
        print(f"Using ConvNeXt-3D-Mamba (3D FFT conv + Mamba-gated 3D ConvSSM {args.ssm_kernel_size}x{args.ssm_kernel_size}x{args.ssm_kernel_size_t}, T={args.T})")
        model = convnext_3d_mamba_tiny(
            num_classes=10,
            T=args.T,
            kernel_size=args.kernel_size,  # 7 for depthwise conv (spatial only)
            ssm_kernel_size=args.ssm_kernel_size,
            ssm_kernel_size_t=args.ssm_kernel_size_t,
            use_delta=True,
            drop_path_rate=args.drop_path,
        )
        model_name = f"ConvNeXt-3D-Mamba-Tiny-T{args.T}-SSM{args.ssm_kernel_size}x{args.ssm_kernel_size}x{args.ssm_kernel_size_t}"
    elif fft_ssm_mode:
        print(f"Using ConvNeXt-FFT-SSM (FFT depthwise conv + ConvSSM blocks, T={args.T})")
        model = convnext_fft_ssm_tiny(
            num_classes=10,
            T=args.T,
            kernel_size=args.kernel_size,
            drop_path_rate=args.drop_path,
        )
        model_name = f"ConvNeXt-FFT-SSM-Tiny-T{args.T}"
    elif fft_simple_mode:
        print(f"Using ConvNeXt-FFT-Simple (same as baseline, depthwise conv uses IFFT(FFT(x)*FFT(w)))")
        model = convnext_fft_tiny(
            num_classes=10,
            drop_path_rate=args.drop_path,
        )
        model_name = "ConvNeXt-FFT-Simple-Tiny"
    elif fourier_pure_mode:
        print(f"Using ConvNeXt-Fourier-Pure (exact ConvNeXt equivalent in FFT domain, no SSM)")
        model = convnext_fourier_pure_tiny(
            num_classes=10,
            kernel_size=args.kernel_size,
            drop_path_rate=args.drop_path,
        )
        model_name = "ConvNeXt-Fourier-Pure-Tiny"
    elif fourier_v3_bf16_mode:
        print(f"Using ConvNeXt-Fourier-V3 (parameter-efficient, ~28M params, bfloat16 SSM)")
        model = convnext_fourier_v3_tiny_bf16(
            num_classes=10,
            T=args.T,
            kernel_size=args.kernel_size,
        )
        model_name = "ConvNeXt-Fourier-V3-Tiny-BF16"
    elif fourier_v3_mode:
        print(f"Using ConvNeXt-Fourier-V3 (parameter-efficient, ~28M params, all-FFT)")
        model = convnext_fourier_v3_tiny(
            num_classes=10,
            T=args.T,
            kernel_size=args.kernel_size,
            drop_path_rate=args.drop_path,
        )
        model_name = "ConvNeXt-Fourier-V3-Tiny"
    elif convnext_mode:
        print(f"Using standard ConvNeXt baseline (7x7 depthwise conv, no SSM)")
        model_map = {
            'tiny': convnext_tiny,
            'small': convnext_small,
            'base': convnext_base,
        }
        model = model_map[args.model](
            num_classes=10,
            drop_path_rate=args.drop_path,
        )
        model_name = f"ConvNeXt-{args.model.capitalize()}"
    elif fourier_v2_bf16_mode:
        print(f"Using ConvNeXt-Fourier-V2 model with bfloat16 SSM operations")
        model = convnext_fourier_v2_tiny_bf16(
            num_classes=10,
            T=args.T,
        )
        model_name = "ConvNeXt-Fourier-V2-Tiny-BF16"
    elif fourier_v2_mode:
        print(f"Using ConvNeXt-Fourier-V2 model (all-FFT with stable real/imag representation)")
        model = convnext_fourier_v2_tiny(
            num_classes=10,
            T=args.T,
            drop_path_rate=args.drop_path,
        )
    elif fourier_mode:
        use_checkpoint = not args.no_checkpointing
        print(f"Using ConvNeXt-Fourier-PreFFT model (pre-FFT in data loader, checkpointing={use_checkpoint})")
        model = convnext_fourier_prefft_tiny(
            num_classes=10,
            T=args.T,
            drop_path_rate=args.drop_path,
            use_checkpoint=use_checkpoint,
        )
    else:
        model_map = {
            'tiny': convnext_ssm_tiny,
            'small': convnext_ssm_small,
            'base': convnext_ssm_base,
        }
        model = model_map[args.model](
            num_classes=10,  # Imagenette has 10 classes
            T=args.T,
            kernel_size=args.kernel_size,
            use_fourier=not args.use_spatial,
            drop_path_rate=args.drop_path,
        )

    # Load datasets with augmentation settings
    print("Loading training data...")
    if args.use_randaugment:
        print(f"  RandAugment enabled: n={args.randaug_n}, m={args.randaug_m}")
    train_images, train_labels = create_dataset(
        args.data_dir, 'train', args.image_size,
        use_randaugment=args.use_randaugment,
        randaug_n=args.randaug_n,
        randaug_m=args.randaug_m,
    )
    print(f"  Train: {len(train_images)} images")

    print("Loading validation data...")
    val_images, val_labels = create_dataset(args.data_dir, 'val', args.image_size)
    print(f"  Val: {len(val_images)} images")

    # Mixup/CutMix settings
    use_mixup = args.mixup_alpha > 0 or args.cutmix_alpha > 0
    if use_mixup:
        print(f"  Mixup alpha={args.mixup_alpha}, CutMix alpha={args.cutmix_alpha}, prob={args.mix_prob}")

    # Effective batch size = per_device_batch * num_devices
    per_device_batch = args.batch_size // num_devices
    effective_batch = per_device_batch * num_devices

    # Use FourierDataLoader for pre-FFT mode
    if fourier_mode:
        print("Pre-computing FFT for all images (one-time cost)...")
        train_images_fft = precompute_fft(train_images)
        val_images_fft = precompute_fft(val_images)
        print("Using FourierDataLoader with pre-computed FFT")
        train_loader = FourierDataLoader(train_images_fft, train_labels, effective_batch, shuffle=True)
        val_loader = FourierDataLoader(val_images_fft, val_labels, effective_batch, shuffle=False, drop_last=False)
        # Free original spatial images to save memory
        del train_images, val_images
    else:
        train_loader = DataLoader(train_images, train_labels, effective_batch, shuffle=True)
        val_loader = DataLoader(val_images, val_labels, effective_batch, shuffle=False, drop_last=False)

    steps_per_epoch = len(train_loader)

    # Initialize training state
    rng = random.PRNGKey(args.seed)
    state = create_train_state(
        rng,
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=per_device_batch,
        image_size=args.image_size,
        fourier_mode=fourier_mode,
    )

    # Replicate state across devices
    state = jax_utils.replicate(state)

    n_params = count_params(jax_utils.unreplicate(state).params)
    if fourier_v2_mode:
        model_name = "ConvNeXt-Fourier-V2-Tiny"
    elif fourier_mode:
        model_name = "ConvNeXt-Fourier-PreFFT-Tiny"
    else:
        model_name = f"ConvNeXt-SSM-{args.model.capitalize()}"
    print(f"\nModel: {model_name}")
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"SSM: T={args.T}")
    if not (fourier_mode or fourier_v2_mode):
        print(f"  kernel_size={args.kernel_size}")
    print(f"Batch size: {effective_batch} ({per_device_batch} per device x {num_devices} devices)")
    print(f"Steps per epoch: {steps_per_epoch}")

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run_name or f"{model_name}_T{args.T}_bs{effective_batch}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                'model': model_name,
                'model_type': args.model_type,
                'model_size': args.model if not fourier_mode else 'tiny',
                'T': args.T,
                'kernel_size': args.kernel_size if not fourier_mode else None,
                'use_spatial': args.use_spatial if not fourier_mode else False,
                'fourier_mode': fourier_mode,
                'use_checkpoint': not args.no_checkpointing if fourier_mode else None,
                'num_params': n_params,
                'num_params_M': n_params / 1e6,
                'batch_size': effective_batch,
                'per_device_batch': per_device_batch,
                'num_devices': num_devices,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'warmup_epochs': args.warmup_epochs,
                'drop_path': args.drop_path,
                'image_size': args.image_size,
                'seed': args.seed,
                'steps_per_epoch': steps_per_epoch,
                # Augmentation settings
                'use_randaugment': args.use_randaugment,
                'randaug_n': args.randaug_n,
                'randaug_m': args.randaug_m,
                'mixup_alpha': args.mixup_alpha,
                'cutmix_alpha': args.cutmix_alpha,
                'mix_prob': args.mix_prob,
                'label_smoothing': args.label_smoothing,
            }
        )
        print(f"Wandb run: {wandb.run.url}")

    # Create pmap'd functions
    p_train_step = pmap(
        partial(train_step, num_classes=10, smoothing=args.label_smoothing),
        axis_name='batch'
    )
    p_eval_step = pmap(partial(eval_step, num_classes=10), axis_name='batch')

    # Mixup training step (only pmap if mixup/cutmix enabled)
    p_train_step_mixup = None
    if use_mixup:
        p_train_step_mixup = pmap(
            partial(train_step_mixup, num_classes=10, smoothing=args.label_smoothing),
            axis_name='batch',
            static_broadcasted_argnums=(4,),  # lam is a scalar, broadcast to all devices
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    best_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...")
    print("(First epoch includes JIT compilation, may take 10-15 minutes...)")
    import sys
    sys.stdout.flush()

    for epoch in range(args.epochs):
        start_time = time.time()

        # Reload training data with fresh augmentation each epoch
        # (Skip for fourier mode - use pre-computed FFT throughout)
        if epoch > 0 and not fourier_mode:
            train_images, train_labels = create_dataset(
                args.data_dir, 'train', args.image_size,
                use_randaugment=args.use_randaugment,
                randaug_n=args.randaug_n,
                randaug_m=args.randaug_m,
            )
            train_loader = DataLoader(train_images, train_labels, effective_batch, shuffle=True)

        # Train
        state, train_metrics, timing_stats = train_epoch(
            state, train_loader, p_train_step, num_devices,
            p_train_step_mixup=p_train_step_mixup,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
        )

        # Evaluate
        val_metrics = evaluate(state, val_loader, p_eval_step, num_devices)

        epoch_time = time.time() - start_time

        # Get current learning rate (from scheduler)
        current_step = (epoch + 1) * steps_per_epoch
        # Get GPU memory
        gpu_mem_gb = get_gpu_memory_usage()

        # Logging
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} ({epoch_time:5.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']*100:5.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:5.2f}% | "
            f"Step: {timing_stats['step_time_mean_ms']:.0f}ms"
        )

        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'] * 100,
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'] * 100,
                'timing/epoch_time_s': epoch_time,
                'timing/step_time_mean_ms': timing_stats['step_time_mean_ms'],
                'timing/step_time_min_ms': timing_stats['step_time_min_ms'],
                'timing/step_time_max_ms': timing_stats['step_time_max_ms'],
                'timing/step_time_std_ms': timing_stats['step_time_std_ms'],
                'timing/throughput_samples_per_sec': timing_stats['throughput_samples_per_sec'],
                'system/gpu_memory_gb': gpu_mem_gb,
                'system/step': current_step,
            })

        # Save checkpoint
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            # Unreplicate state for saving
            checkpoints.save_checkpoint(
                args.output_dir,
                jax_utils.unreplicate(state),
                epoch,
                prefix='best_',
                overwrite=True,
            )
            print(f"  -> New best accuracy: {best_acc*100:.2f}%")

        # Regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoints.save_checkpoint(
                args.output_dir,
                jax_utils.unreplicate(state),
                epoch,
                prefix='checkpoint_',
            )

    print(f"\nTraining complete! Best accuracy: {best_acc*100:.2f}%")

    # Finish wandb
    if use_wandb:
        wandb.log({'best_val_accuracy': best_acc * 100})
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train ConvNeXt-SSM on Imagenette')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data/imagenette2-320',
                        help='Path to Imagenette data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')

    # Model
    parser.add_argument('--model_type', type=str, default='ssm',
                        choices=['convnext', 'ssm', 'fourier', 'fourier_v2', 'fourier_v2_bf16', 'fourier_v3', 'fourier_v3_bf16', 'fourier_pure', 'fft_simple', 'fft_ssm', 'spatial_ssm', 'parallel_ssm', 'gated_ssm', 'gated_ssm_v2', 'ssm_3d', 'pure_ssm_3d'],
                        help='Model type: convnext (standard baseline), ssm (ConvNeXt-SSM), fourier (ConvNeXt-Fourier-PreFFT), fourier_v2 (all-FFT stable), fourier_v2_bf16 (all-FFT bf16), fourier_v3 (all-FFT param-efficient ~28M), fourier_v3_bf16 (all-FFT param-efficient bf16), fourier_pure (exact ConvNeXt equiv in FFT domain, no SSM), fft_simple (simple FFT conv drop-in), fft_ssm (FFT-Simple + ConvSSM blocks), spatial_ssm (spatial Conv + ConvSSM blocks), parallel_ssm (parallel ConvSSM via associative_scan), gated_ssm (Mamba2-style gated parallel ConvSSM), gated_ssm_v2 (input-dependent kernel gating)')
    parser.add_argument('--model', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'],
                        help='Model size (for ssm model type)')
    parser.add_argument('--T', type=int, default=8,
                        help='Number of SSM iterations')
    parser.add_argument('--kernel_size', type=int, default=7,
                        help='SSM kernel size')
    parser.add_argument('--ssm_kernel_size', type=int, default=3,
                        help='3D SSM spatial kernel size (for ssm_3d mode)')
    parser.add_argument('--ssm_kernel_size_t', type=int, default=3,
                        help='3D SSM temporal kernel size (for ssm_3d mode)')
    parser.add_argument('--gating_mode', type=str, default='both',
                        choices=['coefficient', 'kernel_attention', 'both'],
                        help='Gating mode for gated_ssm_v2: coefficient (scalar modulation), kernel_attention (attention-weighted basis kernels), both (combined)')
    parser.add_argument('--num_basis', type=int, default=4,
                        help='Number of basis kernels for kernel_attention gating mode (gated_ssm_v2)')
    parser.add_argument('--use_spatial', action='store_true',
                        help='Use spatial-domain SSM instead of Fourier')
    parser.add_argument('--drop_path', type=float, default=0.1,
                        help='Drop path rate')
    parser.add_argument('--no_checkpointing', action='store_true',
                        help='Disable gradient checkpointing (uses more memory but slightly faster)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Total batch size (across all devices)')
    parser.add_argument('--learning_rate', type=float, default=4e-3,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')

    # Augmentation (established ConvNeXt recipe)
    parser.add_argument('--use_randaugment', action='store_true',
                        help='Use RandAugment (recommended for established recipe)')
    parser.add_argument('--randaug_n', type=int, default=2,
                        help='Number of RandAugment operations')
    parser.add_argument('--randaug_m', type=int, default=9,
                        help='RandAugment magnitude (0-10)')
    parser.add_argument('--mixup_alpha', type=float, default=0.0,
                        help='Mixup alpha (0 = disabled, 0.8 = typical)')
    parser.add_argument('--cutmix_alpha', type=float, default=0.0,
                        help='CutMix alpha (0 = disabled, 1.0 = typical)')
    parser.add_argument('--mix_prob', type=float, default=0.5,
                        help='Probability of applying mixup/cutmix')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Wandb
    parser.add_argument('--wandb_project', type=str, default='convnext-ssm',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (auto-generated if not specified)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
