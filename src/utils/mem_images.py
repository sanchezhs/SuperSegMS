# Create a reusable Python script that generates charts from the user's JSON stats and per-image metrics.
# It will not run any plots here (no data), just write the script to /mnt/data and print the path.

from pathlib import Path


"""
Generate publication-ready charts for the thesis/memoria from:
- Dataset statistics per slice (per_slice.json)
- Per-image evaluation metrics (per_image_metrics.json) from multiple experiments

Usage examples:
  # Only dataset stats (per-slice)
  python gen_memoria_charts.py \
      --per-slice /path/to/dataset/stats/per_slice.json \
      --out charts_out

  # Compare multiple experiments (UNet, YOLO)
  python gen_memoria_charts.py \
      --exp "UNet Block (no SR):/path/to/unet_block/predictions/per_image_metrics.json" \
      --exp "YOLO Block (no SR):/path/to/yolo_block/predictions/per_image_metrics.json" \
      --out charts_out

  # Both dataset stats and experiments
  python gen_memoria_charts.py \
      --per-slice /path/to/dataset/stats/per_slice.json \
      --exp "UNet:/exp1/per_image_metrics.json" \
      --exp "YOLO:/exp2/per_image_metrics.json" \
      --out charts_out
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# --------------------------- IO helpers ---------------------------
def load_json_list(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list of objects")
    return data


def ensure_outdir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)


# --------------------------- Dataset stats charts ---------------------------
def plot_hist_lesion_area(per_slice: List[dict], out_dir: Path, log_x: bool = False) -> None:
    """Histogram of lesion_area_px across slices."""
    areas = [int(r.get("lesion_area_px", 0)) for r in per_slice if r.get("lesion_area_px") is not None]
    if not areas:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(areas, bins=40)
    if log_x:
        plt.xscale("log")
    plt.xlabel("Lesion area (pixels)")
    plt.ylabel("Count")
    plt.title("Distribution of lesion area per slice")
    plt.grid(True, alpha=0.3)
    ensure_outdir(out_dir)
    plt.savefig(out_dir / ("hist_lesion_area_logx.png" if log_x else "hist_lesion_area.png"), bbox_inches="tight")
    plt.close()


def plot_box_lesion_ratio(per_slice: List[dict], out_dir: Path) -> None:
    """Boxplot of lesion_ratio across slices."""
    ratios = [float(r.get("lesion_ratio")) for r in per_slice if r.get("lesion_ratio") is not None]
    if not ratios:
        return
    plt.figure(figsize=(6, 5))
    plt.boxplot(ratios, vert=True, showfliers=True)
    plt.ylabel("Lesion ratio (fraction of image)")
    plt.title("Lesion ratio per slice")
    plt.grid(True, axis="y", alpha=0.3)
    ensure_outdir(out_dir)
    plt.savefig(out_dir / "box_lesion_ratio.png", bbox_inches="tight")
    plt.close()


def plot_hist_components(per_slice: List[dict], out_dir: Path) -> None:
    """Histogram of number of connected components in the mask."""
    comps = [int(r.get("num_components", 0)) for r in per_slice if r.get("num_components") is not None]
    if not comps:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(comps, bins=min(30, max(5, len(set(comps)))))
    plt.xlabel("Number of components")
    plt.ylabel("Count")
    plt.title("Connected components per slice")
    plt.grid(True, alpha=0.3)
    ensure_outdir(out_dir)
    plt.savefig(out_dir / "hist_components.png", bbox_inches="tight")
    plt.close()


# --------------------------- Experiment metrics charts ---------------------------
MetricKeys = ("iou", "dice_score", "precision", "recall", "specificity")

def load_experiment_metrics(exp_items: List[Tuple[str, Path]]) -> Dict[str, List[dict]]:
    """
    exp_items: list of (label, path_to_per_image_metrics.json)
    returns: dict label -> list[dict]
    """
    out: Dict[str, List[dict]] = {}
    for label, p in exp_items:
        rows = load_json_list(p)
        out[label] = rows
    return out


def nanmean_std(values: List[float]) -> Tuple[float, float, int]:
    arr = np.array(values, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return (float("nan"), float("nan"), 0)
    m = float(np.nanmean(arr[valid]))
    s = float(np.nanstd(arr[valid]))
    n = int(valid.sum())
    return (m, s, n)


def bar_metrics_by_experiment(experiments: Dict[str, List[dict]], out_dir: Path) -> None:
    """Bar plot with mean±std per metric for each experiment (one chart per metric)."""
    ensure_outdir(out_dir)
    for key in MetricKeys:
        labels = []
        means = []
        stds = []
        ns = []
        for label, rows in experiments.items():
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            m, s, n = nanmean_std(vals)
            labels.append(label)
            means.append(m)
            stds.append(s)
            ns.append(n)
        if not any(math.isfinite(x) for x in means):
            continue
        plt.figure(figsize=(10, 5))
        x = np.arange(len(labels))
        plt.bar(x, means, yerr=stds, capsize=4)
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylabel(key.replace("_", " ").title())
        plt.title(f"{key.replace('_', ' ').title()} by experiment (mean ± std)")
        for xi, ni in zip(x, ns):
            plt.text(xi, 0, f"n={ni}", ha="center", va="bottom", rotation=90)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"bar_{key}.png", bbox_inches="tight")
        plt.close()


def scatter_area_vs_metric(experiments: Dict[str, List[dict]], metric: str, out_dir: Path, log_x: bool = True) -> None:
    """Scatter lesion_area_px vs metric for each experiment (one figure, multiple series)."""
    ensure_outdir(out_dir)
    plt.figure(figsize=(9, 6))
    any_points = False
    for label, rows in experiments.items():
        xs = []
        ys = []
        for r in rows:
            a = r.get("lesion_area_px")
            y = r.get(metric)
            if a is None or y is None:
                continue
            if not (isinstance(a, (int, float)) and isinstance(y, (int, float))):
                continue
            if math.isfinite(float(a)) and math.isfinite(float(y)):
                xs.append(float(a))
                ys.append(float(y))
        if xs:
            any_points = True
            plt.scatter(xs, ys, alpha=0.5, label=label, s=12)
    if not any_points:
        plt.close()
        return
    if log_x:
        plt.xscale("log")
    plt.xlabel("Lesion area (pixels)")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} vs lesion area")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_area_vs_{metric}.png", bbox_inches="tight")
    plt.close()


def scatter_bboxratio_vs_metric(experiments: Dict[str, List[dict]], metric: str, out_dir: Path) -> None:
    """If per-image metrics include bbox_ratio (from stats merge), plot it against the metric."""
    ensure_outdir(out_dir)
    plt.figure(figsize=(9, 6))
    any_points = False
    for label, rows in experiments.items():
        xs = []
        ys = []
        for r in rows:
            x = r.get("bbox_ratio")
            y = r.get(metric)
            if x is None or y is None:
                continue
            if math.isfinite(float(x)) and math.isfinite(float(y)):
                xs.append(float(x))
                ys.append(float(y))
        if xs:
            any_points = True
            plt.scatter(xs, ys, alpha=0.5, label=label, s=12)
    if not any_points:
        plt.close()
        return
    plt.xlabel("BBox ratio")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} vs bbox ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_bboxratio_vs_{metric}.png", bbox_inches="tight")
    plt.close()


# --------------------------- CLI ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate charts for dataset stats and per-image evaluation metrics.")
    p.add_argument("--per-slice", type=Path, default=None, help="Path to per_slice.json (dataset stats).")
    p.add_argument("--exp", action="append", default=[],
                   help="Experiment spec as 'Label:/path/to/per_image_metrics.json'. Can be repeated.")
    p.add_argument("--out", type=Path, default=Path("charts_out"), help="Output directory for charts.")
    p.add_argument("--no-logx", action="store_true", help="Disable log scale on X for area-based plots.")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out
    ensure_outdir(out)

    # Dataset stats charts
    if args.per_slice:
        try:
            per_slice = load_json_list(args.per_slice)
        except Exception as e:
            print(f"[WARN] Could not load per-slice stats: {e}")
            per_slice = []
        if per_slice:
            plot_hist_lesion_area(per_slice, out / "dataset_stats", log_x=not args.no_logx)
            plot_box_lesion_ratio(per_slice, out / "dataset_stats")
            plot_hist_components(per_slice, out / "dataset_stats")

    # Experiments
    exp_items: List[Tuple[str, Path]] = []
    for spec in args.exp:
        if ":" not in spec:
            print(f"[WARN] Invalid --exp spec: {spec}. Expected 'Label:/path/to/json'")
            continue
        label, path = spec.split(":", 1)
        exp_items.append((label, Path(path)))

    if exp_items:
        exps = load_experiment_metrics(exp_items)
        out_exp = out / "experiments"
        ensure_outdir(out_exp)
        # bar charts per metric
        bar_metrics_by_experiment(exps, out_exp / "bars")
        # scatter vs lesion area
        for key in ("dice_score", "iou", "precision", "recall"):
            scatter_area_vs_metric(exps, key, out_exp / "scatter_area", log_x=not args.no_logx)
        # optional: bbox ratio if attached
        for key in ("dice_score", "iou"):
            scatter_bboxratio_vs_metric(exps, key, out_exp / "scatter_bbox")

    print(f"Charts saved under: {out.resolve()}")


if __name__ == "__main__":
    main()
