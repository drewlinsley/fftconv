"""Extract benchmark data from wandb runs and save to JSON file."""
import json
import glob
import os
import yaml
from datetime import datetime


def extract_benchmark_data(wandb_dir: str = "wandb") -> list:
    """Extract benchmark data from all wandb runs."""
    benchmarks = []

    run_dirs = glob.glob(os.path.join(wandb_dir, "run-*"))

    for run_dir in sorted(run_dirs):
        summary_path = os.path.join(run_dir, "files", "wandb-summary.json")
        config_path = os.path.join(run_dir, "files", "config.yaml")

        if not os.path.exists(summary_path) or not os.path.exists(config_path):
            continue

        try:
            with open(summary_path) as f:
                summary = json.load(f)

            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Extract run info from directory name
            run_name = os.path.basename(run_dir)
            timestamp = run_name.split("-")[1] + "_" + run_name.split("-")[2]

            # Only include completed runs (epoch >= 100)
            epoch = summary.get("epoch", 0)
            if epoch < 100:
                continue

            # Extract config values (wandb stores them as nested dicts)
            def get_config_val(key, default=None):
                if key in config and isinstance(config[key], dict):
                    return config[key].get("value", default)
                return config.get(key, default)

            model_type = get_config_val("model_type", "unknown")
            num_params_M = get_config_val("num_params_M", 0)
            batch_size = get_config_val("batch_size", 32)
            T = get_config_val("T", 1)
            kernel_size = get_config_val("kernel_size", 7)

            # Extract summary values
            best_val_acc = summary.get("best_val_accuracy", summary.get("val/accuracy", 0))
            step_time_ms = summary.get("timing/step_time_mean_ms", 0)

            # Create display name
            display_name = model_type
            if T > 1:
                display_name += f" T={T}"
            if kernel_size != 7:
                display_name += f" K={kernel_size}"

            benchmark = {
                "run_id": run_name,
                "timestamp": timestamp,
                "model_type": model_type,
                "display_name": display_name,
                "num_params_M": num_params_M,
                "batch_size": batch_size,
                "T": T,
                "kernel_size": kernel_size,
                "best_val_acc": best_val_acc,
                "step_time_ms": step_time_ms,
                "epochs": epoch,
            }

            benchmarks.append(benchmark)

        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            continue

    return benchmarks


def save_benchmarks(benchmarks: list, output_path: str = "benchmark_results.json"):
    """Save benchmarks to JSON file."""
    # Sort by accuracy descending
    benchmarks = sorted(benchmarks, key=lambda x: x["best_val_acc"], reverse=True)

    output = {
        "last_updated": datetime.now().isoformat(),
        "num_runs": len(benchmarks),
        "benchmarks": benchmarks
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(benchmarks)} benchmarks to {output_path}")
    return output


def add_manual_benchmark(output_path: str, **kwargs):
    """Add a manual benchmark entry to the results file."""
    try:
        with open(output_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"last_updated": None, "num_runs": 0, "benchmarks": []}

    required = ["model_type", "display_name", "num_params_M", "best_val_acc", "step_time_ms", "batch_size"]
    for key in required:
        if key not in kwargs:
            raise ValueError(f"Missing required field: {key}")

    # Set defaults
    kwargs.setdefault("run_id", f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    kwargs.setdefault("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    kwargs.setdefault("T", 1)
    kwargs.setdefault("kernel_size", 7)
    kwargs.setdefault("epochs", 100)

    data["benchmarks"].append(kwargs)
    data["benchmarks"] = sorted(data["benchmarks"], key=lambda x: x["best_val_acc"], reverse=True)
    data["num_runs"] = len(data["benchmarks"])
    data["last_updated"] = datetime.now().isoformat()

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Added benchmark: {kwargs['display_name']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_dir", default="wandb", help="Path to wandb directory")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file")
    parser.add_argument("--add_manual", action="store_true", help="Add manual benchmark entry")

    # Manual entry fields
    parser.add_argument("--model_type", help="Model type for manual entry")
    parser.add_argument("--display_name", help="Display name for manual entry")
    parser.add_argument("--params", type=float, help="Parameters (M) for manual entry")
    parser.add_argument("--acc", type=float, help="Best val accuracy for manual entry")
    parser.add_argument("--step_time", type=float, help="Step time (ms) for manual entry")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for manual entry")
    parser.add_argument("--T", type=int, default=1, help="T value for manual entry")
    parser.add_argument("--kernel", type=int, default=7, help="Kernel size for manual entry")

    args = parser.parse_args()

    if args.add_manual:
        add_manual_benchmark(
            args.output,
            model_type=args.model_type,
            display_name=args.display_name,
            num_params_M=args.params,
            best_val_acc=args.acc,
            step_time_ms=args.step_time,
            batch_size=args.batch,
            T=args.T,
            kernel_size=args.kernel
        )
    else:
        benchmarks = extract_benchmark_data(args.wandb_dir)
        save_benchmarks(benchmarks, args.output)

        # Print summary
        print("\nBenchmark Summary:")
        print("-" * 80)
        for b in benchmarks:
            print(f"{b['display_name']:30s} | Acc: {b['best_val_acc']:.2f}% | "
                  f"Params: {b['num_params_M']:.1f}M | Step: {b['step_time_ms']:.1f}ms | "
                  f"Batch: {b['batch_size']}")
