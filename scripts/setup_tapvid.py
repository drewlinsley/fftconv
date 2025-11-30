#!/usr/bin/env python3
"""Download and setup TAP-Vid datasets for point tracking evaluation.

Datasets:
- TAP-Vid-DAVIS: 30 real videos with dense annotations
- TAP-Vid-Kinetics: 1,189 YouTube videos with ~25 points each
- TAP-Vid-Kubric: Synthetic with perfect ground truth
- TAPVid-3D: 4,000+ videos with 2.1M 3D trajectories

Usage:
    python scripts/setup_tapvid.py --data_dir ./data/tapvid --datasets davis kubric

Reference:
    - TAP-Vid: https://tapvid.github.io/
    - TAPVid-3D: https://tapvid3d.github.io/
    - GitHub: https://github.com/google-deepmind/tapnet
"""

import os
import argparse
import subprocess
from pathlib import Path


def download_tapvid_davis(data_dir: Path):
    """Download TAP-Vid-DAVIS dataset.

    30 real videos from DAVIS with dense point annotations.
    Good for quick evaluation.
    """
    print("\n" + "="*70)
    print("Downloading TAP-Vid-DAVIS...")
    print("="*70)

    davis_dir = data_dir / "tapvid_davis"
    davis_dir.mkdir(parents=True, exist_ok=True)

    # TAP-Vid DAVIS is hosted on Google Cloud Storage
    url = "gs://dm-tapnet/tapvid_davis.zip"

    # Try gsutil first, fall back to wget with direct URL
    try:
        subprocess.run([
            "gsutil", "-m", "cp", url, str(davis_dir / "tapvid_davis.zip")
        ], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("gsutil not available, trying direct download...")
        # Alternative: direct HTTP URL
        http_url = "https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip"
        subprocess.run([
            "wget", "-P", str(davis_dir), http_url
        ], check=True)

    # Extract
    subprocess.run([
        "unzip", "-o", str(davis_dir / "tapvid_davis.zip"), "-d", str(davis_dir)
    ], check=True)

    print(f"TAP-Vid-DAVIS downloaded to: {davis_dir}")


def download_tapvid_kubric(data_dir: Path):
    """Download TAP-Vid-Kubric dataset.

    Synthetic dataset with perfect ground truth.
    Good for debugging and detailed analysis.
    """
    print("\n" + "="*70)
    print("Downloading TAP-Vid-Kubric...")
    print("="*70)

    kubric_dir = data_dir / "tapvid_kubric"
    kubric_dir.mkdir(parents=True, exist_ok=True)

    url = "gs://dm-tapnet/tapvid_kubric.zip"

    try:
        subprocess.run([
            "gsutil", "-m", "cp", url, str(kubric_dir / "tapvid_kubric.zip")
        ], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        http_url = "https://storage.googleapis.com/dm-tapnet/tapvid_kubric.zip"
        subprocess.run([
            "wget", "-P", str(kubric_dir), http_url
        ], check=True)

    subprocess.run([
        "unzip", "-o", str(kubric_dir / "tapvid_kubric.zip"), "-d", str(kubric_dir)
    ], check=True)

    print(f"TAP-Vid-Kubric downloaded to: {kubric_dir}")


def download_tapvid_kinetics(data_dir: Path):
    """Download TAP-Vid-Kinetics dataset.

    1,189 real YouTube videos with ~25 points each.
    Warning: Large dataset, requires downloading videos from YouTube.
    """
    print("\n" + "="*70)
    print("Downloading TAP-Vid-Kinetics annotations...")
    print("="*70)

    kinetics_dir = data_dir / "tapvid_kinetics"
    kinetics_dir.mkdir(parents=True, exist_ok=True)

    # Download annotations (videos need separate YouTube download)
    url = "gs://dm-tapnet/tapvid_kinetics.zip"

    try:
        subprocess.run([
            "gsutil", "-m", "cp", url, str(kinetics_dir / "tapvid_kinetics.zip")
        ], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        http_url = "https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip"
        subprocess.run([
            "wget", "-P", str(kinetics_dir), http_url
        ], check=True)

    subprocess.run([
        "unzip", "-o", str(kinetics_dir / "tapvid_kinetics.zip"), "-d", str(kinetics_dir)
    ], check=True)

    print(f"TAP-Vid-Kinetics annotations downloaded to: {kinetics_dir}")
    print("\nNOTE: You need to download the actual videos from YouTube separately.")
    print("See: https://github.com/google-deepmind/tapnet#downloading-videos")


def download_tapvid3d(data_dir: Path):
    """Download TAPVid-3D dataset.

    4,000+ videos with 2.1M metric 3D point trajectories.
    Perfect for 3D ConvSSM evaluation.
    """
    print("\n" + "="*70)
    print("Downloading TAPVid-3D...")
    print("="*70)

    tapvid3d_dir = data_dir / "tapvid3d"
    tapvid3d_dir.mkdir(parents=True, exist_ok=True)

    # TAPVid-3D components
    components = [
        "tapvid3d_aria.zip",      # Aria glasses recordings
        "tapvid3d_drivetrack.zip", # Driving scenes
        "tapvid3d_pstudio.zip",   # Panoptic Studio
    ]

    for component in components:
        url = f"gs://dm-tapnet/{component}"
        try:
            subprocess.run([
                "gsutil", "-m", "cp", url, str(tapvid3d_dir / component)
            ], check=True)
            subprocess.run([
                "unzip", "-o", str(tapvid3d_dir / component), "-d", str(tapvid3d_dir)
            ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Warning: Could not download {component}")
            print("TAPVid-3D may require manual download from: https://tapvid3d.github.io/")

    print(f"TAPVid-3D downloaded to: {tapvid3d_dir}")


def setup_tapnet_repo(data_dir: Path):
    """Clone and setup the official TAPNet repository for evaluation."""
    print("\n" + "="*70)
    print("Setting up TAPNet evaluation code...")
    print("="*70)

    tapnet_dir = data_dir / "tapnet"

    if tapnet_dir.exists():
        print(f"TAPNet already exists at {tapnet_dir}")
        subprocess.run(["git", "-C", str(tapnet_dir), "pull"], check=True)
    else:
        subprocess.run([
            "git", "clone",
            "https://github.com/google-deepmind/tapnet.git",
            str(tapnet_dir)
        ], check=True)

    print(f"TAPNet repo cloned to: {tapnet_dir}")
    print("\nTo install dependencies:")
    print(f"  cd {tapnet_dir}")
    print("  pip install -r requirements.txt")


def create_dataloader_template(data_dir: Path):
    """Create a template dataloader for TAP-Vid."""

    template = '''"""TAP-Vid dataloader for 3D ConvSSM point tracking.

This provides a PyTorch/JAX compatible dataloader for TAP-Vid datasets.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle


class TAPVidDataset:
    """TAP-Vid dataset loader.

    Each sample contains:
        - video: (T, H, W, 3) uint8 RGB frames
        - points: (N, T, 2) float32 point coordinates (x, y in pixels)
        - occluded: (N, T) bool, True if point is occluded
        - query_points: (N, 3) float32, (t, x, y) query frame and position
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "davis",  # davis, kubric, kinetics
        resize: Optional[Tuple[int, int]] = (256, 256),
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.resize = resize

        # Load dataset index
        self.samples = self._load_index()

    def _load_index(self):
        """Load dataset index based on split."""
        if self.split == "davis":
            pkl_path = self.data_dir / "tapvid_davis" / "tapvid_davis.pkl"
        elif self.split == "kubric":
            pkl_path = self.data_dir / "tapvid_kubric" / "tapvid_kubric.pkl"
        elif self.split == "kinetics":
            pkl_path = self.data_dir / "tapvid_kinetics" / "tapvid_kinetics.pkl"
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        else:
            print(f"Warning: {pkl_path} not found. Run download first.")
            return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample.

        Returns dict with:
            video: (T, H, W, 3) frames
            points: (N, T, 2) trajectories
            occluded: (N, T) occlusion mask
            query_points: (N, 3) query (t, x, y)
        """
        sample = self.samples[idx]

        video = sample["video"]  # (T, H, W, 3)
        points = sample["points"]  # (N, T, 2)
        occluded = sample["occluded"]  # (N, T)
        query_points = sample["query_points"]  # (N, 3)

        # Resize if needed
        if self.resize is not None:
            video, points = self._resize(video, points)

        return {
            "video": video,
            "points": points,
            "occluded": occluded,
            "query_points": query_points,
        }

    def _resize(self, video, points):
        """Resize video and scale point coordinates."""
        import cv2

        T, H, W, C = video.shape
        new_H, new_W = self.resize

        # Resize frames
        resized = np.zeros((T, new_H, new_W, C), dtype=video.dtype)
        for t in range(T):
            resized[t] = cv2.resize(video[t], (new_W, new_H))

        # Scale points
        scale_x = new_W / W
        scale_y = new_H / H
        points_scaled = points.copy()
        points_scaled[..., 0] *= scale_x
        points_scaled[..., 1] *= scale_y

        return resized, points_scaled


def compute_tapvid_metrics(
    pred_tracks: np.ndarray,  # (N, T, 2) predicted
    pred_occlusion: np.ndarray,  # (N, T) predicted occlusion prob
    gt_tracks: np.ndarray,  # (N, T, 2) ground truth
    gt_occlusion: np.ndarray,  # (N, T) ground truth occlusion
    query_points: np.ndarray,  # (N, 3) query (t, x, y)
) -> Dict[str, float]:
    """Compute TAP-Vid evaluation metrics.

    Metrics:
        - Average Jaccard (AJ): Primary metric
        - <δ^x_avg: Fraction within threshold
        - Occlusion Accuracy (OA)

    See: https://arxiv.org/abs/2211.03726
    """
    # Position error
    errors = np.linalg.norm(pred_tracks - gt_tracks, axis=-1)  # (N, T)

    # Thresholds for <δ metric
    thresholds = [1, 2, 4, 8, 16]

    # Only evaluate visible points
    visible = ~gt_occlusion

    metrics = {}

    # Average position error on visible points
    metrics["avg_error"] = float(np.mean(errors[visible]))

    # <δ metrics: fraction of points within threshold
    for thresh in thresholds:
        within = errors < thresh
        metrics[f"delta_{thresh}"] = float(np.mean(within[visible]))

    # Occlusion accuracy
    pred_occ_binary = pred_occlusion > 0.5
    occ_correct = (pred_occ_binary == gt_occlusion)
    metrics["occlusion_accuracy"] = float(np.mean(occ_correct))

    # Jaccard: requires both position and occlusion correct
    # A point is "correct" if within threshold AND occlusion matches
    for thresh in thresholds:
        position_correct = errors < thresh
        both_correct = position_correct & occ_correct

        # Jaccard = intersection / union
        # True positives: predicted visible AND correct
        # All positives: predicted visible OR actually visible
        pred_visible = ~pred_occ_binary
        true_visible = visible

        tp = both_correct & pred_visible & true_visible
        fp = pred_visible & ~true_visible
        fn = ~pred_visible & true_visible

        intersection = np.sum(tp)
        union = np.sum(tp | fp | fn)

        jaccard = intersection / max(union, 1)
        metrics[f"jaccard_{thresh}"] = float(jaccard)

    # Average Jaccard (primary metric)
    metrics["average_jaccard"] = float(np.mean([
        metrics[f"jaccard_{t}"] for t in thresholds
    ]))

    return metrics


if __name__ == "__main__":
    # Test loading
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/tapvid"

    print("Testing TAP-Vid dataloader...")
    dataset = TAPVidDataset(data_dir, split="davis")
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Video shape: {sample['video'].shape}")
        print(f"Points shape: {sample['points'].shape}")
        print(f"Occluded shape: {sample['occluded'].shape}")
        print(f"Query points shape: {sample['query_points'].shape}")
'''

    loader_path = data_dir / "tapvid_dataloader.py"
    with open(loader_path, "w") as f:
        f.write(template)

    print(f"\nDataloader template created at: {loader_path}")


def main():
    parser = argparse.ArgumentParser(description="Download TAP-Vid datasets")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/tapvid",
        help="Directory to store datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["davis", "kubric", "kinetics", "tapvid3d", "all"],
        default=["davis"],
        help="Which datasets to download"
    )
    parser.add_argument(
        "--setup_tapnet",
        action="store_true",
        help="Also clone TAPNet repo for evaluation"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("TAP-Vid Dataset Setup")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Datasets: {args.datasets}")

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["davis", "kubric", "kinetics", "tapvid3d"]

    # Download requested datasets
    if "davis" in datasets:
        download_tapvid_davis(data_dir)

    if "kubric" in datasets:
        download_tapvid_kubric(data_dir)

    if "kinetics" in datasets:
        download_tapvid_kinetics(data_dir)

    if "tapvid3d" in datasets:
        download_tapvid3d(data_dir)

    # Setup evaluation code
    if args.setup_tapnet:
        setup_tapnet_repo(data_dir)

    # Create dataloader template
    create_dataloader_template(data_dir)

    print("\n" + "="*70)
    print("Setup complete!")
    print("="*70)
    print(f"\nData directory: {data_dir}")
    print("\nNext steps:")
    print("1. Run evaluation with your 3D ConvSSM model")
    print("2. Compare against TAP-Vid baselines (PIPs, TAPIR, CoTracker)")
    print("\nSee: https://github.com/google-deepmind/tapnet")


if __name__ == "__main__":
    main()
