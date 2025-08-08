"""
This module provides functionality to compute pixel-level segmentation metrics
from predictions and ground truth masks. It includes methods to calculate confusion matrix components, derive metrics like IoU, Dice score, precision, recall, and specificity, and evaluate a batch of predictions against their corresponding ground truth masks.
#             f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}"
Compute pixel-level segmentation metrics between prediction and ground truth masks.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from schemas.pipeline_schemas import SegmentationMetrics


class MetricsCalculator:
    @staticmethod
    def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert probabilities or grayscale masks into binary masks.
        """
        return (mask > threshold).astype(np.uint8)

    @staticmethod
    def confusion_matrix(
        pred: np.ndarray,
        gt: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[int, int, int, int]:
        """
        Compute TP, FP, FN, TN for a single pair of masks.
        """
        p = MetricsCalculator.binarize(pred, threshold)
        g = MetricsCalculator.binarize(gt, threshold)
        tp = int((p & g).sum())
        fp = int((p & ~g).sum())
        fn = int((~p & g).sum())
        tn = int((~p & ~g).sum())
        return tp, fp, fn, tn

    @staticmethod
    def from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
        """
        Compute all metrics from counts.
        """
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "iou": iou,
            "dice_score": dice,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
        }

    @staticmethod
    def evaluate_batch(
        preds: np.ndarray,
        gts: np.ndarray,
        times: List[float] = None,
        threshold: float = 0.5
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Evaluate a batch of N masks:
            - preds: shape (N, H, W) predicted continuous masks
            - gts:   shape (N, H, W) ground-truth continuous or binary masks
            - times: optional list of inference times per image

        Returns:
            - avg_metrics: dict of average over N images (including inference_time if times given)
        """
        n = preds.shape[0]
        per_image = []
        for i in range(n):
            tp, fp, fn, tn = MetricsCalculator.confusion_matrix(
                preds[i], gts[i], threshold
            )
            m = MetricsCalculator.from_confusion(tp, fp, fn, tn)
            if times is not None:
                m["inference_time"] = float(times[i])
            per_image.append(m)

        # Compute averages
        avg = {}
        for key in per_image[0]:
            vals = [m[key] for m in per_image]
            avg[key] = float(np.mean(vals))
        return avg

    @staticmethod
    def kfold_summary(metrics_list: List[SegmentationMetrics]) -> Dict[str, Dict[str, float]]:
        """
        Compute mean and standard deviation for each metric across all folds.
        Args:
            metrics_list (list[dict]): List of dictionaries containing metrics from each fold.
        Returns:
            dict: A dictionary with mean and std for each metric.
        """
        if not metrics_list:
            raise ValueError("metrics_list is empty. Cannot compute mean and std.")
        keys = metrics_list[0].model_dump().keys()
        return {
            key: {
                "mean": float(np.mean([m.model_dump()[key] for m in metrics_list])),
                "std": float(np.std([m.model_dump()[key] for m in metrics_list]))
            } for key in keys
        }

    @staticmethod
    def write_kfold_summary(
        metrics: Dict[str, Dict[str, float]],
        output_path: Path,
        stdout: bool = True
    ) -> None:
        """
        Write the summary of metrics to a file.
        Args:
            metrics (dict): Dictionary containing mean and std for each metric.
            output_path (Path): Path to save the summary.
        """
        with open(output_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: mean={value['mean']:.4f}, std={value['std']:.4f}\n")

        if stdout:
            for key, value in metrics.items():
                print(f"{key}: mean={value['mean']:.4f}, std={value['std']:.4f}")
        print(f"Metrics summary written to {output_path}")

        logger.info(f"Metrics summary written to {output_path}")