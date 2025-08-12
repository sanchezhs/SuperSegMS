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
        return (mask.astype(np.float32) > float(threshold))
    
    @staticmethod
    def confusion_matrix(
        pred: np.ndarray,
        gt: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[int, int, int, int]:
        """
        Compute TP, FP, FN, TN for a single pair of masks.
        Works in boolean space to avoid uint8 bitwise pitfalls.
        """
        p = MetricsCalculator.binarize(pred, threshold)  # bool
        g = MetricsCalculator.binarize(gt, threshold)    # bool

        tp = int(np.logical_and(p, g).sum())
        fp = int(np.logical_and(p, np.logical_not(g)).sum())
        fn = int(np.logical_and(np.logical_not(p), g).sum())
        tn = int(np.logical_and(np.logical_not(p), np.logical_not(g)).sum())
        return tp, fp, fn, tn

    @staticmethod
    def from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
        """
        Compute IoU, Dice, Precision, Recall, Specificity from counts.
        """
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "iou": float(iou),
            "dice_score": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
        }
    
    @staticmethod
    def evaluate_batch(
        preds: np.ndarray,      # shape (N,H,W) in [0,1] or logits if threshold applies
        gts: np.ndarray,        # shape (N,H,W) in {0,1} or [0,1]
        times: List[float] | None = None,
        threshold: float = 0.5
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Returns:
            avg_metrics: dict of averages over N
            per_image:   list of dicts per image with all metrics (+ inference_time if times provided)
        """
        n = int(preds.shape[0])
        per_image: List[Dict[str, float]] = []
        for i in range(n):
            tp, fp, fn, tn = MetricsCalculator.confusion_matrix(preds[i], gts[i], threshold)
            m = MetricsCalculator.from_confusion(tp, fp, fn, tn)
            if times is not None:
                m["inference_time"] = float(times[i])
            per_image.append(m)

        avg = {}
        for key in per_image[0].keys():
            vals = [m[key] for m in per_image if key in m]
            avg[key] = float(np.mean(vals)) if len(vals) > 0 else 0.0
        return avg, per_image

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