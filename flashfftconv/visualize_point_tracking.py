"""Visualize Point Tracking Predictions.

This script loads a trained point tracking model and generates visualizations
showing predicted vs ground truth trajectories overlaid on video frames.

Usage:
    python -m flashfftconv.visualize_point_tracking \
        --checkpoint ./checkpoints_point_tracking_ssm8_split/best_model.pkl \
        --model_type ssm_3d \
        --output_dir ./visualizations \
        --num_videos 3
"""

import os
import pickle
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import cv2
import jax
import jax.numpy as jnp
import flax.linen as nn

# Import from training script
from flashfftconv.train_point_tracking import (
    TAPVidDataset,
    collate_batch,
    ConvSSMPointTracker,
    CorrelationOnlyTracker,
)


def create_model(
    model_type: str,
    hidden_dim: int = 256,
    ssm_iterations: int = 8,
    kernel_size: Tuple[int, int, int] = (3, 7, 7),
    num_refinement: int = 3,
):
    """Create model based on model type."""
    if model_type == 'correlation_only':
        return CorrelationOnlyTracker(hidden_dim=hidden_dim)
    else:
        return ConvSSMPointTracker(
            hidden_dim=hidden_dim,
            ssm_iterations=ssm_iterations,
            kernel_size=kernel_size,
            num_refinement=num_refinement,
        )


def load_model(
    checkpoint_path: str,
    model_type: str,
    hidden_dim: int = 256,
    ssm_iterations: int = 8,
    kernel_size: Tuple[int, int, int] = (3, 7, 7),
    num_refinement: int = 3,
    img_size: int = 256,
    max_frames: int = 32,
    max_points: int = 20,
):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)

    params = ckpt['params']
    model = create_model(
        model_type, hidden_dim, ssm_iterations, kernel_size, num_refinement
    )

    return model, params


def run_inference(model, params, video, query_points):
    """Run inference on a single sample."""
    pred_coords, pred_occ = model.apply(
        {'params': params},
        video,
        query_points,
        train=False,
    )
    return np.array(pred_coords), np.array(pred_occ)


def get_color_for_point(idx: int, total: int) -> Tuple[int, int, int]:
    """Get a distinct color for each point using HSV colorspace."""
    import colorsys
    hue = idx / max(total, 1)
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return tuple(int(c * 255) for c in rgb)


def draw_tracks_on_frame(
    frame: np.ndarray,
    pred_coords: np.ndarray,  # (N, T, 2)
    gt_coords: np.ndarray,    # (N, T, 2)
    pred_occ: np.ndarray,     # (N, T)
    gt_occ: np.ndarray,       # (N, T)
    frame_idx: int,
    trail_length: int = 10,
) -> np.ndarray:
    """Draw predicted and GT tracks on a frame.

    Args:
        frame: (H, W, 3) uint8 frame
        pred_coords: (N, T, 2) predicted coordinates (x, y)
        gt_coords: (N, T, 2) ground truth coordinates (x, y)
        pred_occ: (N, T) predicted occlusion
        gt_occ: (N, T) ground truth occlusion
        frame_idx: current frame index
        trail_length: how many past frames to show in trail

    Returns:
        Frame with tracks drawn
    """
    frame = frame.copy()
    N, T = pred_coords.shape[:2]

    for point_idx in range(N):
        color = get_color_for_point(point_idx, N)

        # Draw trail (past positions)
        start_t = max(0, frame_idx - trail_length)

        # GT trail (solid line)
        for t in range(start_t, frame_idx):
            if not gt_occ[point_idx, t] and not gt_occ[point_idx, t + 1]:
                pt1 = tuple(int(c) for c in gt_coords[point_idx, t])
                pt2 = tuple(int(c) for c in gt_coords[point_idx, t + 1])
                cv2.line(frame, pt1, pt2, color, 2)

        # Predicted trail (dashed/lighter)
        pred_color = tuple(min(255, int(c * 1.3)) for c in color)
        for t in range(start_t, frame_idx):
            if pred_occ[point_idx, t] < 0.5 and pred_occ[point_idx, t + 1] < 0.5:
                pt1 = tuple(int(c) for c in pred_coords[point_idx, t])
                pt2 = tuple(int(c) for c in pred_coords[point_idx, t + 1])
                # Dashed line effect
                if (t - start_t) % 2 == 0:
                    cv2.line(frame, pt1, pt2, pred_color, 1, cv2.LINE_AA)

        # Draw current positions
        # GT position (circle)
        if not gt_occ[point_idx, frame_idx]:
            gt_pt = tuple(int(c) for c in gt_coords[point_idx, frame_idx])
            cv2.circle(frame, gt_pt, 6, color, -1)
            cv2.circle(frame, gt_pt, 6, (255, 255, 255), 1)

        # Predicted position (X marker)
        if pred_occ[point_idx, frame_idx] < 0.5:
            pred_pt = tuple(int(c) for c in pred_coords[point_idx, frame_idx])
            size = 5
            cv2.line(frame,
                     (pred_pt[0] - size, pred_pt[1] - size),
                     (pred_pt[0] + size, pred_pt[1] + size),
                     pred_color, 2)
            cv2.line(frame,
                     (pred_pt[0] - size, pred_pt[1] + size),
                     (pred_pt[0] + size, pred_pt[1] - size),
                     pred_color, 2)

    return frame


def create_comparison_video(
    video: np.ndarray,           # (T, H, W, 3)
    pred_coords: np.ndarray,     # (N, T, 2)
    gt_coords: np.ndarray,       # (N, T, 2)
    pred_occ: np.ndarray,        # (N, T)
    gt_occ: np.ndarray,          # (N, T)
    output_path: str,
    fps: int = 10,
) -> None:
    """Create a video showing predicted vs GT tracks."""
    T, H, W, _ = video.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for t in range(T):
        frame = draw_tracks_on_frame(
            video[t], pred_coords, gt_coords, pred_occ, gt_occ, t
        )
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"  Saved video to {output_path}")


def create_frame_grid(
    video: np.ndarray,           # (T, H, W, 3)
    pred_coords: np.ndarray,     # (N, T, 2)
    gt_coords: np.ndarray,       # (N, T, 2)
    pred_occ: np.ndarray,        # (N, T)
    gt_occ: np.ndarray,          # (N, T)
    output_path: str,
    num_frames: int = 8,
) -> None:
    """Create an image grid showing predictions at key frames."""
    T, H, W, _ = video.shape

    # Select evenly spaced frames
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

    # Create frames with tracks
    frames = []
    for t in frame_indices:
        frame = draw_tracks_on_frame(
            video[t], pred_coords, gt_coords, pred_occ, gt_occ, t
        )
        # Add frame number
        cv2.putText(frame, f"t={t}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        frames.append(frame)

    # Arrange in grid (2 rows x 4 cols)
    rows = []
    cols_per_row = 4
    for i in range(0, len(frames), cols_per_row):
        row_frames = frames[i:i + cols_per_row]
        # Pad if necessary
        while len(row_frames) < cols_per_row:
            row_frames.append(np.zeros_like(frames[0]))
        row = np.concatenate(row_frames, axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)

    # Add legend
    legend_h = 60
    legend = np.zeros((legend_h, grid.shape[1], 3), dtype=np.uint8)

    # GT marker
    cv2.circle(legend, (50, 30), 6, (0, 255, 0), -1)
    cv2.putText(legend, "Ground Truth", (70, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    # Pred marker
    cv2.line(legend, (195, 25), (205, 35), (100, 255, 100), 2)
    cv2.line(legend, (195, 35), (205, 25), (100, 255, 100), 2)
    cv2.putText(legend, "Prediction", (220, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    grid_with_legend = np.concatenate([legend, grid], axis=0)

    # Save
    cv2.imwrite(output_path, cv2.cvtColor(grid_with_legend, cv2.COLOR_RGB2BGR))
    print(f"  Saved grid to {output_path}")


def visualize_dataset(
    checkpoint_path: str,
    model_type: str,
    data_path: str,
    output_dir: str,
    num_videos: int = 5,
    hidden_dim: int = 256,
    ssm_iterations: int = 8,
    kernel_t: int = 3,
    kernel_h: int = 7,
    kernel_w: int = 7,
    num_refinement: int = 3,
    img_size: int = 256,
    max_frames: int = 32,
    max_points: int = 20,
):
    """Visualize predictions on validation set."""
    print("="*70)
    print("Point Tracking Visualization")
    print("="*70)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, params = load_model(
        checkpoint_path,
        model_type,
        hidden_dim,
        ssm_iterations,
        (kernel_t, kernel_h, kernel_w),
        num_refinement,
        img_size,
        max_frames,
        max_points,
    )

    # Load validation dataset
    val_dataset = TAPVidDataset(
        data_path,
        resize=(img_size, img_size),
        max_frames=max_frames,
        max_points=max_points,
        split='val',
        val_ratio=0.2,
    )

    print(f"\nGenerating visualizations for {min(num_videos, len(val_dataset))} videos...")

    # JIT compile the inference
    @jax.jit
    def jit_inference(params, video, query_points):
        return model.apply(
            {'params': params},
            video,
            query_points,
            train=False,
        )

    # Process videos
    for i in range(min(num_videos, len(val_dataset))):
        sample = val_dataset[i]
        video_name = sample['name']
        print(f"\nProcessing video {i+1}: {video_name}")

        # Prepare batch
        batch = collate_batch([sample], max_frames, max_points)
        video = jnp.array(batch['video'])
        query_points = jnp.array(batch['query_points'])

        # Run inference
        pred_coords, pred_occ = jit_inference(params, video, query_points)
        pred_coords = np.array(pred_coords[0])
        pred_occ = np.array(pred_occ[0])

        # Get ground truth
        vf = batch['valid_frames'][0]
        vp = batch['valid_points'][0]
        gt_coords = batch['points'][0, :vp, :vf]
        gt_occ = batch['occluded'][0, :vp, :vf]
        video_frames = batch['video'][0, :vf]

        # Trim predictions to valid region
        pred_coords = pred_coords[:vp, :vf]
        pred_occ = pred_occ[:vp, :vf]

        # Create video
        video_path = output_dir / f"{video_name}_tracking.mp4"
        create_comparison_video(
            video_frames, pred_coords, gt_coords, pred_occ, gt_occ,
            str(video_path)
        )

        # Create image grid
        grid_path = output_dir / f"{video_name}_grid.png"
        create_frame_grid(
            video_frames, pred_coords, gt_coords, pred_occ, gt_occ,
            str(grid_path)
        )

        # Compute per-video metrics
        visible = (~gt_occ)
        if np.sum(visible) > 0:
            errors = np.linalg.norm(pred_coords - gt_coords, axis=-1)
            avg_error = np.mean(errors[visible])
            delta_4 = np.mean(errors[visible] < 4)
            print(f"  Avg error: {avg_error:.2f}px, δ<4: {delta_4*100:.1f}%")

    print(f"\n{'='*70}")
    print(f"Visualizations saved to {output_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualize Point Tracking')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='ssm_3d',
                        choices=['ssm_3d', 'correlation_only'],
                        help='Model type')
    parser.add_argument('--data_path', type=str,
                        default='./data/tapvid/tapvid_davis/tapvid_davis/tapvid_davis.pkl',
                        help='Path to TAP-Vid pickle file')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Output directory')
    parser.add_argument('--num_videos', type=int, default=5,
                        help='Number of videos to visualize')

    # Model params
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--ssm_iterations', type=int, default=8)
    parser.add_argument('--kernel_t', type=int, default=3)
    parser.add_argument('--kernel_h', type=int, default=7)
    parser.add_argument('--kernel_w', type=int, default=7)
    parser.add_argument('--num_refinement', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--max_points', type=int, default=20)

    args = parser.parse_args()

    visualize_dataset(
        args.checkpoint,
        args.model_type,
        args.data_path,
        args.output_dir,
        args.num_videos,
        args.hidden_dim,
        args.ssm_iterations,
        args.kernel_t,
        args.kernel_h,
        args.kernel_w,
        args.num_refinement,
        args.img_size,
        args.max_frames,
        args.max_points,
    )


if __name__ == '__main__':
    main()
