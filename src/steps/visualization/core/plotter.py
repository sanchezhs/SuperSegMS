# steps/evaluation/viz/metrics.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from schemas.pipeline_schemas import SegmentationMetrics


class MetricsPlotter:
    """
    Visualization helper for segmentation metrics.

    Required:
        metrics: aggregated metrics (e.g., from evaluate()).

    Optional (enable richer plots if provided):
        per_image: list of dicts, each with keys like {"iou":..., "dice_score":..., "precision":..., "recall":..., "specificity":..., "inference_time":...}
        kfold_summary: dict metric -> {"mean": float, "std": float}
        threshold_sweep: dict with optional arrays/lists:
            {
                "thresholds": [...],
                "dice": [...],
                "iou": [...],
                "precision": [...],   # optional
                "recall": [...],      # optional
            }
        latency_list: list[float] of per-image inference times (seconds)
        lesion_areas: list[float] of lesion areas (pixels) aligned with per_image
        heatmap_matrix: 2D np.ndarray of shape (n_models, n_metrics)
        heatmap_row_labels: list[str] model names
        heatmap_col_labels: list[str] metric names
    """

    def __init__(
        self,
        metrics: SegmentationMetrics,
        per_image: Optional[List[Dict[str, float]]] = None,
        kfold_summary: Optional[Dict[str, Dict[str, float]]] = None,
        threshold_sweep: Optional[Dict[str, Sequence[float]]] = None,
        latency_list: Optional[Sequence[float]] = None,
        lesion_areas: Optional[Sequence[float]] = None,
        heatmap_matrix: Optional[np.ndarray] = None,
        heatmap_row_labels: Optional[List[str]] = None,
        heatmap_col_labels: Optional[List[str]] = None,
    ):
        self.iou = float(metrics.iou)
        self.dice_score = float(metrics.dice_score)
        self.precision = float(metrics.precision)
        self.recall = float(metrics.recall)
        self.specificity = float(metrics.specificity)
        self.inference_time = None if metrics.inference_time is None else float(metrics.inference_time)

        self.per_image = per_image or []
        self.kfold_summary = kfold_summary or {}
        self.threshold_sweep = threshold_sweep or {}
        self.latency_list = list(latency_list) if latency_list is not None else None
        self.lesion_areas = list(lesion_areas) if lesion_areas is not None else None

        self.heatmap_matrix = heatmap_matrix
        self.heatmap_row_labels = heatmap_row_labels or []
        self.heatmap_col_labels = heatmap_col_labels or []

    # ---------------- Base summary plots ---------------- #

    def plot_metrics_bar(
        self,
        save_path: Optional[Path] = None,
        title: str = "Segmentation Metrics",
        figsize: tuple = (9, 5)
    ) -> None:
        """Simple bar summary for one model (no error bars)."""
        metrics_data = {
            "IoU": self.iou,
            "Dice": self.dice_score,
            "Precision": self.precision,
            "Recall": self.recall,
            "Specificity": self.specificity,
        }

        plt.figure(figsize=figsize)
        bars = plt.bar(list(metrics_data.keys()), list(metrics_data.values()))
        for b, v in zip(bars, metrics_data.values()):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
        plt.title(title)
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.grid(axis="y", alpha=0.3)
        self._save_or_show(save_path)
        logger.info(f"Metrics bar plot saved to: {save_path}") if save_path else None

    def plot_kfold_errorbars(
        self,
        save_path: Optional[Path] = None,
        title: str = "Cross-Validation (mean ± std)",
        figsize: tuple = (9, 5)
    ) -> None:
        """Bar plot with error bars from k-fold summary."""
        if not self.kfold_summary:
            logger.warning("kfold_summary not provided; skipping plot_kfold_errorbars.")
            return

        names, means, stds = [], [], []
        for k in ["iou", "dice_score", "precision", "recall", "specificity"]:
            if k in self.kfold_summary:
                names.append(self._pretty(k))
                means.append(self.kfold_summary[k]["mean"])
                stds.append(self.kfold_summary[k]["std"])

        if not names:
            logger.warning("kfold_summary does not include expected keys; skipping.")
            return

        x = np.arange(len(names))
        plt.figure(figsize=figsize)
        plt.bar(x, means, yerr=stds, capsize=5)
        plt.xticks(x, names)
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.title(title)
        plt.grid(axis="y", alpha=0.3)
        self._save_or_show(save_path)
        logger.info(f"CV errorbars saved to: {save_path}") if save_path else None

    # ---------------- Distributions & relationships ---------------- #

    def plot_boxplots_per_image(
        self,
        save_path: Optional[Path] = None,
        title: str = "Per-image metric distributions",
        figsize: tuple = (10, 6)
    ) -> None:
        """Boxplots over per-image metrics (IoU, Dice, Precision, Recall, Specificity)."""
        if not self.per_image:
            logger.warning("per_image not provided; skipping plot_boxplots_per_image.")
            return

        keys = ["iou", "dice_score", "precision", "recall", "specificity"]
        data = []
        labels = []
        for k in keys:
            vals = [float(d[k]) for d in self.per_image if k in d]
            if vals:
                data.append(vals)
                labels.append(self._pretty(k))

        if not data:
            logger.warning("No per-image values found; skipping.")
            return

        plt.figure(figsize=figsize)
        plt.boxplot(data, labels=labels, showfliers=True)
        plt.ylim(0, 1.05)
        plt.grid(axis="y", alpha=0.3)
        plt.title(title)
        self._save_or_show(save_path)
        logger.info(f"Per-image boxplots saved to: {save_path}") if save_path else None

    def plot_latency_distribution(
        self,
        save_path: Optional[Path] = None,
        title: str = "Inference time distribution",
        figsize: tuple = (9, 5)
    ) -> None:
        """Histogram + CDF of per-image latency."""
        times = self.latency_list or [d["inference_time"] for d in self.per_image if "inference_time" in d]
        times = [float(t) for t in times if t is not None]
        if not times:
            logger.warning("No latency data; skipping plot_latency_distribution.")
            return

        times_np = np.array(times, dtype=float)
        plt.figure(figsize=figsize)
        # histogram
        plt.hist(times_np, bins=30, alpha=0.6, density=True)
        # CDF
        xs = np.sort(times_np)
        cdf = np.arange(1, len(xs) + 1) / len(xs)
        plt.twinx()
        plt.plot(xs, cdf, drawstyle="steps-post")
        plt.title(title)
        plt.xlabel("Seconds")
        self._save_or_show(save_path)
        logger.info(f"Latency distribution saved to: {save_path}") if save_path else None

    def plot_metric_vs_size(
        self,
        metric: str = "dice_score",
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
        figsize: tuple = (8, 6)
    ) -> None:
        """Scatter of metric vs lesion area (pixels). Requires per_image and lesion_areas aligned."""
        if not self.per_image or self.lesion_areas is None:
            logger.warning("per_image or lesion_areas not provided; skipping plot_metric_vs_size.")
            return

        vals = [float(d.get(metric, np.nan)) for d in self.per_image]
        areas = [float(a) for a in self.lesion_areas]
        if len(vals) != len(areas) or len(vals) == 0:
            logger.warning("metric and lesion_areas length mismatch or empty; skipping.")
            return

        vals_np = np.array(vals, dtype=float)
        areas_np = np.array(areas, dtype=float)
        mask = np.isfinite(vals_np) & np.isfinite(areas_np)
        vals_np, areas_np = vals_np[mask], areas_np[mask]
        if vals_np.size == 0:
            logger.warning("No finite data for metric_vs_size; skipping.")
            return

        plt.figure(figsize=figsize)
        plt.scatter(areas_np, vals_np, alpha=0.5)
        # trend line
        if areas_np.size >= 2:
            z = np.polyfit(areas_np, vals_np, 1)
            xfit = np.linspace(areas_np.min(), areas_np.max(), 100)
            yfit = np.polyval(z, xfit)
            plt.plot(xfit, yfit)

        plt.xlabel("Lesion area (px)")
        plt.ylabel(self._pretty(metric))
        plt.title(title or f"{self._pretty(metric)} vs lesion size")
        plt.grid(True, alpha=0.3)
        self._save_or_show(save_path)
        logger.info(f"Metric vs size plot saved to: {save_path}") if save_path else None

    # ---------------- Threshold sweeps & PR ---------------- #

    def plot_threshold_sweep(
        self,
        save_path: Optional[Path] = None,
        title: str = "Threshold sweep",
        figsize: tuple = (9, 5)
    ) -> None:
        """
        Plot Dice/IoU (and optionally Precision/Recall) as a function of threshold.
        Expects threshold_sweep = {"thresholds": [...], "dice": [...], "iou": [...], "precision": [...], "recall": [...]}
        """
        ts = self.threshold_sweep.get("thresholds")
        if ts is None:
            logger.warning("threshold_sweep.thresholds not provided; skipping plot_threshold_sweep.")
            return

        plt.figure(figsize=figsize)
        for key in ["dice", "iou", "precision", "recall"]:
            vals = self.threshold_sweep.get(key)
            if vals is not None:
                plt.plot(ts, vals, label=self._pretty(key))
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(title)
        self._save_or_show(save_path)
        logger.info(f"Threshold sweep saved to: {save_path}") if save_path else None

    def plot_precision_recall_curve(
        self,
        save_path: Optional[Path] = None,
        title: str = "Precision–Recall curve",
        figsize: tuple = (7, 6)
    ) -> None:
        """If you computed (precision, recall) pairs for multiple thresholds, plot the PR curve."""
        prs = None
        p = self.threshold_sweep.get("precision")
        r = self.threshold_sweep.get("recall")
        if p is not None and r is not None and len(p) == len(r) and len(p) > 1:
            prs = (np.array(p, dtype=float), np.array(r, dtype=float))
        if prs is None:
            logger.warning("No precision/recall sweep in threshold_sweep; skipping PR curve.")
            return

        plt.figure(figsize=figsize)
        plt.plot(prs[1], prs[0])  # x: recall, y: precision
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.title(title)
        self._save_or_show(save_path)
        logger.info(f"PR curve saved to: {save_path}") if save_path else None

    # ---------------- Heatmap comparison ---------------- #

    def plot_heatmap_comparison(
        self,
        save_path: Optional[Path] = None,
        title: str = "Model comparison (heatmap)",
        figsize: tuple = (9, 6)
    ) -> None:
        """
        Heatmap for comparing multiple models across metrics.
        Expects heatmap_matrix shape (n_models, n_metrics) and labels lists.
        """
        M = self.heatmap_matrix
        if M is None or M.size == 0:
            logger.warning("heatmap_matrix not provided; skipping plot_heatmap_comparison.")
            return
        if not (self.heatmap_row_labels and self.heatmap_col_labels):
            logger.warning("Heatmap labels missing; skipping.")
            return

        plt.figure(figsize=figsize)
        im = plt.imshow(M, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(M.shape[1]), self.heatmap_col_labels, rotation=45, ha="right")
        plt.yticks(np.arange(M.shape[0]), self.heatmap_row_labels)
        plt.title(title)

        # annotate
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                plt.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="white" if M[i, j] < 0.5 else "black")

        plt.tight_layout()
        self._save_or_show(save_path)
        logger.info(f"Heatmap comparison saved to: {save_path}") if save_path else None

    # ---------------- Legacy (optional) ---------------- #

    def plot_metrics_radar(
        self,
        save_path: Optional[Path] = None,
        title: str = "Segmentation Metrics (radar)",
        figsize: tuple = (7, 7)
    ) -> None:
        """Keep radar as optional eye-candy."""
        metrics_names = ["IoU", "Dice", "Precision", "Recall", "Specificity"]
        metrics_values = [self.iou, self.dice_score, self.precision, self.recall, self.specificity]

        N = len(metrics_names)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        vals = metrics_values + metrics_values[:1]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))
        ax.plot(angles, vals, "o-", linewidth=2)
        ax.fill(angles, vals, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_title(title)
        self._save_or_show(save_path)

    # ---------------- Orchestrator ---------------- #

    def save_all_plots(
        self,
        base_path: Path,
        prefix: str = "metrics",
        title_prefix: str = "Model"
    ) -> None:
        """
        Save everything available. Only plots with the required inputs are generated.
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # 1) Always: simple summary bars
        self.plot_metrics_bar(
            save_path=base_path / f"{prefix}_bar.png",
            title=f"{title_prefix} - Summary"
        )

        # 2) If CV summary is available: error bars
        if self.kfold_summary:
            self.plot_kfold_errorbars(
                save_path=base_path / f"{prefix}_cv_errorbars.png",
                title=f"{title_prefix} - CV (mean±std)"
            )

        # 3) Per-image distributions
        if self.per_image:
            self.plot_boxplots_per_image(
                save_path=base_path / f"{prefix}_boxplots.png",
                title=f"{title_prefix} - Per-image distributions"
            )

        # 4) Latency
        if (self.latency_list and len(self.latency_list) > 0) or any("inference_time" in d for d in self.per_image):
            self.plot_latency_distribution(
                save_path=base_path / f"{prefix}_latency.png",
                title=f"{title_prefix} - Inference time"
            )

        # 5) Size relationship
        if self.lesion_areas is not None and self.per_image:
            self.plot_metric_vs_size(
                metric="dice_score",
                save_path=base_path / f"{prefix}_dice_vs_size.png",
                title=f"{title_prefix} - Dice vs lesion size"
            )
            self.plot_metric_vs_size(
                metric="iou",
                save_path=base_path / f"{prefix}_iou_vs_size.png",
                title=f"{title_prefix} - IoU vs lesion size"
            )

        # 6) Threshold sweep & PR
        if self.threshold_sweep:
            self.plot_threshold_sweep(
                save_path=base_path / f"{prefix}_threshold_sweep.png",
                title=f"{title_prefix} - Threshold sweep"
            )
            self.plot_precision_recall_curve(
                save_path=base_path / f"{prefix}_pr_curve.png",
                title=f"{title_prefix} - Precision–Recall"
            )

        # 7) Heatmap comparison across models
        if self.heatmap_matrix is not None and self.heatmap_matrix.size > 0:
            self.plot_heatmap_comparison(
                save_path=base_path / f"{prefix}_heatmap.png",
                title=f"{title_prefix} - Model comparison"
            )

        logger.info(f"All plots saved in: {base_path}")

    # ---------------- Utils ---------------- #

    @staticmethod
    def _pretty(key: str) -> str:
        return {
            "iou": "IoU",
            "dice_score": "Dice",
            "precision": "Precision",
            "recall": "Recall",
            "specificity": "Specificity",
            "dice": "Dice",
        }.get(key, key)

    @staticmethod
    def _save_or_show(save_path: Optional[Path]) -> None:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()
