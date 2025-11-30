# ImageNet Training Script for ConvNeXt-SSM
# Uses JAX/Flax with Optax and tf.data pipeline
#
# Usage:
#   python -m flashfftconv.train_imagenet \
#       --data_dir /path/to/imagenet \
#       --output_dir ./checkpoints \
#       --batch_size 256 \
#       --epochs 300

import os
import time
import argparse
from typing import Dict, Any, Tuple, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
import optax
import flax
from flax.training import train_state, checkpoints
from flax import jax_utils

import tensorflow as tf
import tensorflow_datasets as tfds

from flashfftconv.convnext_ssm import (
    ConvNeXtSSM,
    convnext_ssm_tiny,
    convnext_ssm_small,
    convnext_ssm_base,
    count_params,
)


# =============================================================================
# Data Pipeline
# =============================================================================

IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406])
IMAGENET_STD = jnp.array([0.229, 0.224, 0.225])


def preprocess_for_train(example: Dict, image_size: int = 224) -> Dict:
    """Training preprocessing with data augmentation."""
    image = example['image']

    # Decode if needed
    if image.dtype == tf.string:
        image = tf.io.decode_jpeg(image, channels=3)

    # Random resized crop
    shape = tf.shape(image)
    min_dim = tf.minimum(shape[0], shape[1])

    # Random crop between 8% and 100% of image area
    scale = tf.random.uniform([], 0.08, 1.0)
    ratio = tf.random.uniform([], 0.75, 1.333)

    target_h = tf.cast(tf.sqrt(tf.cast(min_dim * min_dim, tf.float32) * scale / ratio), tf.int32)
    target_w = tf.cast(tf.sqrt(tf.cast(min_dim * min_dim, tf.float32) * scale * ratio), tf.int32)
    target_h = tf.minimum(target_h, shape[0])
    target_w = tf.minimum(target_w, shape[1])

    offset_h = tf.random.uniform([], 0, shape[0] - target_h + 1, dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, shape[1] - target_w + 1, dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, target_h, target_w)
    image = tf.image.resize(image, [image_size, image_size])

    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Color jitter (simplified)
    image = tf.image.random_brightness(image, 0.4)
    image = tf.image.random_contrast(image, 0.6, 1.4)
    image = tf.image.random_saturation(image, 0.6, 1.4)

    # Normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # ImageNet normalization
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return {'image': image, 'label': example['label']}


def preprocess_for_eval(example: Dict, image_size: int = 224) -> Dict:
    """Evaluation preprocessing (center crop)."""
    image = example['image']

    # Decode if needed
    if image.dtype == tf.string:
        image = tf.io.decode_jpeg(image, channels=3)

    # Resize and center crop
    shape = tf.shape(image)
    resize_size = int(image_size / 0.875)  # 256 for 224
    image = tf.image.resize(image, [resize_size, resize_size])

    # Center crop
    offset = (resize_size - image_size) // 2
    image = tf.image.crop_to_bounding_box(image, offset, offset, image_size, image_size)

    # Normalize
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return {'image': image, 'label': example['label']}


def create_imagenet_dataset(
    data_dir: str,
    batch_size: int,
    split: str = 'train',
    image_size: int = 224,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    shuffle_buffer: int = 10000,
    cache: bool = False,
) -> tf.data.Dataset:
    """
    Create ImageNet tf.data pipeline.

    Args:
        data_dir: Path to ImageNet data (tfrecords or raw)
        batch_size: Batch size per device
        split: 'train' or 'validation'
        image_size: Target image size
        num_parallel_calls: Parallelism for map operations
        shuffle_buffer: Shuffle buffer size
        cache: Whether to cache dataset in memory

    Returns:
        tf.data.Dataset yielding (images, labels)
    """
    # Try TFDS first, fall back to manual loading
    try:
        ds = tfds.load(
            'imagenet2012',
            split=split,
            data_dir=data_dir,
            shuffle_files=(split == 'train'),
        )
    except Exception:
        # Manual loading from directory structure
        if split == 'train':
            ds = tf.keras.utils.image_dataset_from_directory(
                os.path.join(data_dir, 'train'),
                batch_size=None,
                image_size=None,
                shuffle=True,
            )
            ds = ds.map(lambda x, y: {'image': x, 'label': y})
        else:
            ds = tf.keras.utils.image_dataset_from_directory(
                os.path.join(data_dir, 'val'),
                batch_size=None,
                image_size=None,
                shuffle=False,
            )
            ds = ds.map(lambda x, y: {'image': x, 'label': y})

    # Preprocessing
    if split == 'train':
        preprocess_fn = partial(preprocess_for_train, image_size=image_size)
        ds = ds.shuffle(shuffle_buffer)
    else:
        preprocess_fn = partial(preprocess_for_eval, image_size=image_size)

    if cache:
        ds = ds.cache()

    ds = ds.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def prepare_batch_for_jax(batch: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert tf.data batch to JAX arrays."""
    images = jnp.array(batch['image'].numpy())
    labels = jnp.array(batch['label'].numpy())
    return images, labels


# =============================================================================
# Training State
# =============================================================================

class TrainState(train_state.TrainState):
    """Extended training state with batch stats and dropout RNG."""
    dropout_rng: jax.Array


def create_train_state(
    rng: jax.Array,
    model: ConvNeXtSSM,
    learning_rate: float,
    weight_decay: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    input_shape: Tuple[int, ...] = (1, 224, 224, 3),
) -> TrainState:
    """Create training state with optimizer."""
    params_rng, dropout_rng = random.split(rng)

    # Initialize model
    dummy_input = jnp.ones(input_shape)
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
# Training and Evaluation Steps
# =============================================================================

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy loss with label smoothing."""
    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, num_classes)

    # Label smoothing
    smoothing = 0.1
    one_hot = one_hot * (1 - smoothing) + smoothing / num_classes

    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(one_hot * log_probs, axis=-1).mean()


@partial(jax.jit, donate_argnums=(0,))
def train_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Single training step."""
    dropout_rng, new_dropout_rng = random.split(state.dropout_rng)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            images,
            train=True,
            rngs={'dropout': dropout_rng},
        )
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    # Accuracy
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)

    # Update state
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }

    return state, metrics


@jax.jit
def eval_step(
    state: TrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Single evaluation step."""
    logits = state.apply_fn(
        {'params': state.params},
        images,
        train=False,
    )
    loss = cross_entropy_loss(logits, labels)

    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)

    # Top-5 accuracy
    top5_preds = jnp.argsort(logits, axis=-1)[:, -5:]
    top5_correct = jnp.any(top5_preds == labels[:, None], axis=-1)
    top5_accuracy = jnp.mean(top5_correct)

    return {
        'loss': loss,
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
    }


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    state: TrainState,
    train_ds: tf.data.Dataset,
    steps_per_epoch: int,
) -> Tuple[TrainState, Dict[str, float]]:
    """Train for one epoch."""
    train_metrics = []

    for step, batch in enumerate(train_ds.take(steps_per_epoch)):
        images, labels = prepare_batch_for_jax(batch)
        state, metrics = train_step(state, images, labels)
        train_metrics.append(metrics)

    # Average metrics
    avg_metrics = {
        k: float(jnp.mean(jnp.array([m[k] for m in train_metrics])))
        for k in train_metrics[0].keys()
    }

    return state, avg_metrics


def evaluate(
    state: TrainState,
    eval_ds: tf.data.Dataset,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    eval_metrics = []

    for batch in eval_ds:
        images, labels = prepare_batch_for_jax(batch)
        metrics = eval_step(state, images, labels)
        eval_metrics.append(metrics)

    # Average metrics
    avg_metrics = {
        k: float(jnp.mean(jnp.array([m[k] for m in eval_metrics])))
        for k in eval_metrics[0].keys()
    }

    return avg_metrics


def train(args):
    """Main training function."""
    print(f"JAX devices: {jax.devices()}")

    # Create model
    model_map = {
        'tiny': convnext_ssm_tiny,
        'small': convnext_ssm_small,
        'base': convnext_ssm_base,
    }
    model = model_map[args.model](
        num_classes=1000,
        T=args.T,
        kernel_size=args.kernel_size,
        use_fourier=not args.use_spatial,
    )

    # Create datasets
    train_ds = create_imagenet_dataset(
        args.data_dir,
        batch_size=args.batch_size,
        split='train',
        image_size=args.image_size,
    )
    val_ds = create_imagenet_dataset(
        args.data_dir,
        batch_size=args.batch_size,
        split='validation',
        image_size=args.image_size,
    )

    # Calculate steps
    num_train_samples = 1281167  # ImageNet train size
    steps_per_epoch = num_train_samples // args.batch_size

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
        input_shape=(args.batch_size, args.image_size, args.image_size, 3),
    )

    n_params = count_params(state.params)
    print(f"Model: ConvNeXt-SSM-{args.model.capitalize()}")
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"SSM: T={args.T}, kernel_size={args.kernel_size}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        state, train_metrics = train_epoch(state, train_ds, steps_per_epoch)

        # Evaluate
        val_metrics = evaluate(state, val_ds)

        epoch_time = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']*100:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%, "
            f"Top-5: {val_metrics['top5_accuracy']*100:.2f}%"
        )

        # Save checkpoint
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            checkpoints.save_checkpoint(
                args.output_dir,
                state,
                epoch,
                prefix='best_',
                overwrite=True,
            )
            print(f"  -> New best accuracy: {best_acc*100:.2f}%")

        # Regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoints.save_checkpoint(
                args.output_dir,
                state,
                epoch,
                prefix='checkpoint_',
            )

    print(f"\nTraining complete! Best accuracy: {best_acc*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train ConvNeXt-SSM on ImageNet')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to ImageNet data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')

    # Model
    parser.add_argument('--model', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'],
                        help='Model size')
    parser.add_argument('--T', type=int, default=8,
                        help='Number of SSM iterations')
    parser.add_argument('--kernel_size', type=int, default=7,
                        help='SSM kernel size')
    parser.add_argument('--use_spatial', action='store_true',
                        help='Use spatial-domain SSM instead of Fourier')

    # Training
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=4e-3,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='Number of warmup epochs')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
