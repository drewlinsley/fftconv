"""Plot benchmark results using seaborn."""
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_benchmarks(path: str = "benchmark_results.json") -> pd.DataFrame:
    """Load benchmarks from JSON file into DataFrame."""
    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["benchmarks"])
    return df


def plot_benchmarks(df: pd.DataFrame, output_prefix: str = "benchmark"):
    """Create benchmark visualization plots."""
    # Set style
    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11

    # Filter to only include runs with valid step times
    df = df[df["step_time_ms"] > 0].copy()

    if df.empty:
        print("No benchmarks with valid step times found!")
        return

    # Sort by accuracy for consistent ordering
    df = df.sort_values("best_val_acc", ascending=False)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Plot 1: Accuracy vs Step Time (scatter) ---
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        df["step_time_ms"],
        df["best_val_acc"],
        c=df["num_params_M"],
        s=100 + df["num_params_M"] * 3,  # Size proportional to params
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5
    )
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Parameters (M)", fontsize=10)

    # Add labels for each point
    for _, row in df.iterrows():
        ax1.annotate(
            row["display_name"],
            (row["step_time_ms"], row["best_val_acc"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8
        )

    ax1.set_xlabel("Step Time (ms)", fontsize=12)
    ax1.set_ylabel("Best Validation Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy vs Training Speed", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Params vs Step Time (scatter) ---
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(
        df["step_time_ms"],
        df["num_params_M"],
        c=df["best_val_acc"],
        s=100,
        cmap="RdYlGn",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5
    )
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label("Accuracy (%)", fontsize=10)

    # Add labels
    for _, row in df.iterrows():
        ax2.annotate(
            row["display_name"],
            (row["step_time_ms"], row["num_params_M"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8
        )

    ax2.set_xlabel("Step Time (ms)", fontsize=12)
    ax2.set_ylabel("Parameters (M)", fontsize=12)
    ax2.set_title("Model Size vs Training Speed", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Step Time Bar Chart ---
    ax3 = axes[1, 0]

    # Use unique display names, take first occurrence if duplicates
    df_unique = df.drop_duplicates(subset=["display_name"], keep="first")

    colors = sns.color_palette("husl", len(df_unique))
    bars = ax3.barh(
        df_unique["display_name"],
        df_unique["step_time_ms"],
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )

    # Add value labels on bars
    for bar, val in zip(bars, df_unique["step_time_ms"]):
        ax3.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}ms",
            va="center",
            fontsize=9
        )

    ax3.set_xlabel("Step Time (ms)", fontsize=12)
    ax3.set_ylabel("")
    ax3.set_title("Training Step Time by Model", fontsize=14, fontweight="bold")
    ax3.invert_yaxis()  # Best at top

    # --- Plot 4: Accuracy Bar Chart ---
    ax4 = axes[1, 1]

    bars2 = ax4.barh(
        df_unique["display_name"],
        df_unique["best_val_acc"],
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )

    # Add value labels on bars
    for bar, val in zip(bars2, df_unique["best_val_acc"]):
        ax4.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9
        )

    ax4.set_xlabel("Best Validation Accuracy (%)", fontsize=12)
    ax4.set_ylabel("")
    ax4.set_title("Accuracy by Model", fontsize=14, fontweight="bold")
    ax4.invert_yaxis()  # Best at top
    ax4.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_overview.png", dpi=150, bbox_inches="tight")
    print(f"Saved {output_prefix}_overview.png")
    plt.close()

    # --- Create efficiency plot (accuracy per ms) ---
    fig2, ax5 = plt.subplots(figsize=(12, 8))

    df_unique["efficiency"] = df_unique["best_val_acc"] / df_unique["step_time_ms"]
    df_sorted = df_unique.sort_values("efficiency", ascending=True)

    colors2 = sns.color_palette("viridis", len(df_sorted))
    bars3 = ax5.barh(
        df_sorted["display_name"],
        df_sorted["efficiency"],
        color=colors2,
        edgecolor="black",
        linewidth=0.5
    )

    for bar, val in zip(bars3, df_sorted["efficiency"]):
        ax5.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9
        )

    ax5.set_xlabel("Efficiency (Accuracy % / Step Time ms)", fontsize=12)
    ax5.set_ylabel("")
    ax5.set_title("Training Efficiency by Model\n(Higher = Better Accuracy per Compute)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_efficiency.png", dpi=150, bbox_inches="tight")
    print(f"Saved {output_prefix}_efficiency.png")
    plt.close()

    # Print summary table
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Model':<35s} | {'Accuracy':>10s} | {'Params (M)':>10s} | {'Step (ms)':>10s} | {'Batch':>6s}")
    print("-" * 100)
    for _, row in df_unique.iterrows():
        print(f"{row['display_name']:<35s} | {row['best_val_acc']:>10.2f} | {row['num_params_M']:>10.1f} | {row['step_time_ms']:>10.1f} | {row['batch_size']:>6d}")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmark_results.json", help="Input JSON file")
    parser.add_argument("--output", default="benchmark", help="Output file prefix")
    args = parser.parse_args()

    df = load_benchmarks(args.input)
    plot_benchmarks(df, args.output)
