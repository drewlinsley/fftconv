"""Simple single-GPU training script for debugging.

This is a minimal training script that:
1. Uses a single GPU (no pmap complexity)
2. Prints progress immediately (no buffering issues)
3. Uses the new Mamba-style SSM

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m flashfftconv.train_simple
"""

import os
import sys
import time
import argparse
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

# Force print to flush immediately
print = partial(print, flush=True)

# Import our models
from .mamba_ssm import SimpleMambaConvNeXt, create_simple_mamba_model
from .convnext_fourier_v2 import convnext_fourier_v2_debug, convnext_fourier_v2_tiny


def load_imagenette(data_dir: str, batch_size: int = 32, image_size: int = 224):
    """Load Imagenette dataset using PIL (simple, no dependencies)."""
    from PIL import Image

    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'

    # Class names
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Found {len(classes)} classes: {classes}")

    def load_split(split_dir):
        images, labels = [], []
        for class_name in classes:
            class_dir = split_dir / class_name
            for img_path in class_dir.glob('*.JPEG'):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((image_size, image_size), Image.BILINEAR)
                    img = np.array(img, dtype=np.float32) / 255.0
                    images.append(img)
                    labels.append(class_to_idx[class_name])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        return np.stack(images), np.array(labels)

    print("Loading training set...")
    train_images, train_labels = load_split(train_dir)
    print(f"  Loaded {len(train_images)} training images")

    print("Loading validation set...")
    val_images, val_labels = load_split(val_dir)
    print(f"  Loaded {len(val_images)} validation images")

    return (train_images, train_labels), (val_images, val_labels), len(classes)


def create_train_state(model, rng, learning_rate, input_shape=(1, 224, 224, 3)):
    """Initialize model and optimizer."""
    print("Initializing model...")
    params = model.init(rng, jnp.ones(input_shape), train=False)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params['params']))
    print(f"  Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Optimizer with warmup
    tx = optax.adamw(learning_rate, weight_decay=0.05)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def cross_entropy_loss(logits, labels):
    """Standard cross-entropy loss."""
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))


@jax.jit
def train_step(state, batch):
    """Single training step."""
    images, labels = batch

    def loss_fn(params):
        logits = state.apply_fn(params, images, train=True)
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    # Accuracy
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == labels)

    state = state.apply_gradients(grads=grads)
    return state, loss, acc


@jax.jit
def eval_step(state, batch):
    """Single evaluation step."""
    images, labels = batch
    logits = state.apply_fn(state.params, images, train=False)
    loss = cross_entropy_loss(logits, labels)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == labels)
    return loss, acc


def train_epoch(state, train_data, batch_size, rng):
    """Train for one epoch."""
    images, labels = train_data
    n_samples = len(images)
    n_batches = n_samples // batch_size

    # Shuffle
    perm = jax.random.permutation(rng, n_samples)
    images = images[perm]
    labels = labels[perm]

    epoch_loss = 0.0
    epoch_acc = 0.0

    for i in range(n_batches):
        batch_images = images[i*batch_size:(i+1)*batch_size]
        batch_labels = labels[i*batch_size:(i+1)*batch_size]

        state, loss, acc = train_step(state, (batch_images, batch_labels))
        epoch_loss += loss
        epoch_acc += acc

        # Print progress every 10 batches
        if (i + 1) % 10 == 0:
            print(f"    Batch {i+1}/{n_batches}: loss={loss:.4f}, acc={acc:.4f}")

    return state, epoch_loss / n_batches, epoch_acc / n_batches


def evaluate(state, val_data, batch_size):
    """Evaluate on validation set."""
    images, labels = val_data
    n_samples = len(images)
    n_batches = n_samples // batch_size

    total_loss = 0.0
    total_acc = 0.0

    for i in range(n_batches):
        batch_images = images[i*batch_size:(i+1)*batch_size]
        batch_labels = labels[i*batch_size:(i+1)*batch_size]

        loss, acc = eval_step(state, (batch_images, batch_labels))
        total_loss += loss
        total_acc += acc

    return total_loss / n_batches, total_acc / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/imagenette2-320')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--T', type=int, default=4, help='SSM iterations')
    parser.add_argument('--model', type=str, default='mamba', choices=['mamba', 'fourier_v2'],
                        help='Model type: mamba (real domain) or fourier_v2 (all FFT)')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='mamba-fourier-debug')
    args = parser.parse_args()

    print("=" * 60)
    print("Simple Mamba-Fourier Training Script")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Model: {args.model}")
    print(f"Data dir: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"T (SSM steps): {args.T}")

    # Initialize WandB if requested
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"{args.model}-T{args.T}-debug"
            )
            print(f"WandB initialized: {wandb_run.url}")
        except ImportError:
            print("WandB not installed, skipping logging")

    # Load data
    print("\n--- Loading Dataset ---")
    train_data, val_data, num_classes = load_imagenette(
        args.data_dir, args.batch_size
    )

    # Create model
    print("\n--- Creating Model ---")
    if args.model == 'mamba':
        model = create_simple_mamba_model(num_classes=num_classes, T=args.T)
    elif args.model == 'fourier_v2':
        model = convnext_fourier_v2_debug(num_classes=num_classes, T=args.T)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    rng = jax.random.PRNGKey(0)
    state = create_train_state(model, rng, args.lr)

    # Compile the train step (do a warmup)
    print("\n--- Compiling (first step) ---")
    compile_start = time.time()
    dummy_batch = (train_data[0][:args.batch_size], train_data[1][:args.batch_size])
    state, _, _ = train_step(state, dummy_batch)
    compile_time = time.time() - compile_start
    print(f"  Compilation time: {compile_time:.1f}s")

    # Training loop
    print("\n--- Training ---")
    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        rng, epoch_rng = jax.random.split(rng)
        state, train_loss, train_acc = train_epoch(
            state, train_data, args.batch_size, epoch_rng
        )

        # Evaluate
        val_loss, val_acc = evaluate(state, val_data, args.batch_size)

        epoch_time = time.time() - epoch_start

        # Print results
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
              f"time={epoch_time:.1f}s")

        # Log to WandB
        if wandb_run:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': float(train_loss),
                'train/accuracy': float(train_acc),
                'val/loss': float(val_loss),
                'val/accuracy': float(val_acc),
                'time/epoch': epoch_time,
            })

    print("\n--- Done! ---")
    if wandb_run:
        wandb.finish()


if __name__ == '__main__':
    main()
