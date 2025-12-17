"""Analyze gesture benchmark results and plot metrics per resolution.

Usage:
    python analyze_results.py --csv results_summary_no_smoothing.csv --json results_no_smoothing.json --outdir plots_no_smoothing

You can run again with the smoothing files:
    python analyze_results.py --csv results_summary_smoothing.csv --json results_smoothing.json --outdir plots_smoothing

Outputs:
    - Saves one PNG per metric per resolution (bars comparing methods)
    - Prints aggregated tables to the terminal

Dependencies: pandas, matplotlib. Install with:
    pip install pandas matplotlib
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import pandas as pd

METRICS = {
    "accuracy(%)": "Accuracy (%)",
    "unknown_rate(%)": "Unknown Rate (%)",
    "avg_latency_ms": "Average Latency (ms)",
    "fps": "Frames Per Second",
}

METHOD_ORDER = ["MediaPipe", "YOLOv8_pose", "OpenCV"]
COLORS = {
    "MediaPipe": "#1f77b4",
    "YOLOv8_pose": "#ff7f0e",
    "OpenCV": "#2ca02c",
}


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]
    # expected: method, resolution, gesture, accuracy(%), unknown_rate(%), avg_latency_ms, fps
    rename_map = {
        "accuracy(%)": "accuracy(%)",
        "unknown_rate(%)": "unknown_rate(%)",
        "avg_latency_ms": "avg_latency_ms",
    }
    return df


def aggregate_by_resolution(df: pd.DataFrame) -> pd.DataFrame:
    # Compute mean per method+resolution across gestures
    grouped = (
        df.groupby(["method", "resolution"])
        .agg({
            "accuracy(%)": "mean",
            "unknown_rate(%)": "mean",
            "avg_latency_ms": "mean",
            "fps": "mean",
        })
        .reset_index()
    )
    return grouped


def plot_metric_per_resolution(df: pd.DataFrame, metric: str, title: str, outdir: Path):
    # For each resolution, make a bar chart comparing methods
    resolutions = sorted(df["resolution"].unique())
    for res in resolutions:
        subset = df[df["resolution"] == res]
        # Ensure method order
        subset = subset.set_index("method").reindex(METHOD_ORDER).reset_index()

        plt.figure(figsize=(6, 4))
        bars = plt.bar(subset["method"], subset[metric], color=[COLORS.get(m, "gray") for m in subset["method"]])
        plt.title(f"{title} @ {res}")
        plt.ylabel(title)
        plt.xticks(rotation=15)
        plt.tight_layout()

        # Annotate values on bars
        for bar, val in zip(bars, subset[metric]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        out_path = outdir / f"{metric.replace('%','pct').replace(' ','_')}_res_{res.replace('x','_')}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")


def print_tables(df: pd.DataFrame):
    print("\n=== Aggregated Metrics (mean across gestures) ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.sort_values(["resolution", "method"]))


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results by resolution")
    parser.add_argument("--csv", required=True, type=Path, help="Path to results_summary CSV")
    parser.add_argument("--json", type=Path, help="Path to results JSON (optional; not required for plots)")
    parser.add_argument("--outdir", default="plots", type=Path, help="Directory to write plots")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df_raw = load_csv(args.csv)
    # Clean column names back to expected
    df_raw.rename(columns={c: c.lower() for c in df_raw.columns}, inplace=True)

    # Basic validation
    required_cols = {"method", "resolution", "gesture", "accuracy(%)", "unknown_rate(%)", "avg_latency_ms", "fps"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df_agg = aggregate_by_resolution(df_raw)

    # Print tables
    print_tables(df_agg)

    # Plot per metric per resolution
    for metric, title in METRICS.items():
        plot_metric_per_resolution(df_agg, metric, title, args.outdir)

    # Optionally, dump the aggregated data for quick reuse
    agg_out = args.outdir / "aggregated_by_resolution.csv"
    df_agg.to_csv(agg_out, index=False)
    print(f"\nSaved aggregated table: {agg_out}")

    # JSON is optional; we don't plot from it here but keep hook for future extensions
    if args.json and args.json.exists():
        with open(args.json, "r") as f:
            data_json = json.load(f)
        # Example: print available resolutions per method from JSON
        print("\n(JSON) Resolutions present per method:")
        for method, payload in data_json.items():
            if isinstance(payload, dict) and "all_resolutions" in payload:
                print(f"  {method}: {list(payload['all_resolutions'].keys())}")


if __name__ == "__main__":
    main()
