"""
This module provides functionality to compute pixel-level segmentation metrics
from predictions and ground truth masks. It includes methods to calculate confusion matrix components, derive metrics like IoU, Dice score, precision, recall, and specificity, and evaluate a batch of predictions against their corresponding ground truth masks.
#             f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}"
Compute pixel-level segmentation metrics between prediction and ground truth masks.
"""
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger
import numpy as np
import json

from schemas.pipeline_schemas import SegmentationMetrics

class MetricsCalculator:
    """
    A class to compute pixel-level segmentation metrics from predictions and ground truth masks.
    """
    @staticmethod
    def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Binarize a mask based on a threshold.
        Args:
            mask (np.ndarray): The input mask array.
            threshold (float): The threshold value to binarize the mask. Defaults to 0.5.
        Returns:
            np.ndarray: A binary mask where values greater than the threshold are set to True, and others to False.
        """
        return (mask.astype(np.float32) > float(threshold))

    @staticmethod
    def confusion_matrix(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> Tuple[int, int, int, int]:
        """
        Compute the confusion matrix components: TP, FP, FN, TN.
        Args:
            pred (np.ndarray): The predicted mask array.
            gt (np.ndarray): The ground truth mask array.
            threshold (float): The threshold value to binarize the masks. Defaults to 0.5.
        Returns:
            Tuple[int, int, int, int]: A tuple containing TP, FP, FN, TN counts.
        """
        p = MetricsCalculator.binarize(pred, threshold)
        g = MetricsCalculator.binarize(gt, threshold)
        tp = int(np.logical_and(p, g).sum())
        fp = int(np.logical_and(p, np.logical_not(g)).sum())
        fn = int(np.logical_and(np.logical_not(p), g).sum())
        tn = int(np.logical_and(np.logical_not(p), np.logical_not(g)).sum())
        return tp, fp, fn, tn

    @staticmethod
    def from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
        """
        Compute segmentation metrics from confusion matrix components.
        Args:
            tp (int): True Positives count.
            fp (int): False Positives count.
            fn (int): False Negatives count.
            tn (int): True Negatives count.
        Returns:
            Dict[str, float]: A dictionary containing IoU, Dice score, Precision, Recall, and Specificity.
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
    def _metrics_for_empty_gt(
        pred: np.ndarray,
        empty_gt_policy: str = "perfect"  # "perfect" | "skip" | "zero"
    ) -> Dict[str, float]:
        """
        Handle the special case where the ground-truth mask has no positives.
        - perfect: treat (gt=0, pred=0) as a perfect case (IoU=1, Dice=1, etc.)
        - skip:    return NaN for IoU/Dice/Precision/Recall so they don't affect the average; keep Specificity valid
        - zero:    keep zeros for IoU/Dice/Precision/Recall (penalizes even perfect empty predictions)
        """
        p_bin = MetricsCalculator.binarize(pred)
        has_pred_pos = bool(p_bin.any())

        # Compute specificity even for empty GT (depends on FP/TN)
        tn = int((~p_bin).sum())
        fp = int(p_bin.sum())
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if not has_pred_pos:
            if empty_gt_policy == "perfect":
                return {
                    "iou": 1.0, "dice_score": 1.0,
                    "precision": 1.0, "recall": 1.0,
                    "specificity": float(specificity)
                }
            elif empty_gt_policy == "skip":
                # Use NaN so np.nanmean ignores this image for those metrics
                return {
                    "iou": np.nan, "dice_score": np.nan,
                    "precision": np.nan, "recall": np.nan,
                    "specificity": float(specificity)
                }
            else:  # "zero"
                return {
                    "iou": 0.0, "dice_score": 0.0,
                    "precision": 0.0, "recall": 0.0,
                    "specificity": float(specificity)
                }
        else:
            # False positives on an empty GT: this should be penalized
            return {
                "iou": 0.0, "dice_score": 0.0,
                "precision": 0.0, "recall": 0.0,
                "specificity": float(specificity)
            }

    @staticmethod
    def evaluate_batch(
        preds: np.ndarray,      # (N,H,W) in [0,1] or logits if threshold applies
        gts: np.ndarray,        # (N,H,W) in {0,1} or [0,1]
        times: List[float] | None = None,
        threshold: float = 0.5,
        empty_gt_policy: str = "perfect"  # "perfect" | "skip" | "zero"
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Returns:
            avg_metrics: dict of averages over N (NaNs ignored via nanmean)
            per_image:   list of dicts per image with metrics (+ inference_time if times provided)
        Notes:
            - When GT is empty:
              * (pred empty) => handled by 'empty_gt_policy'
              * (pred non-empty) => penalized (FPs)
        """
        assert empty_gt_policy in {"perfect", "skip", "zero"}
        n = int(preds.shape[0])
        per_image: List[Dict[str, float]] = []

        for i in range(n):
            gt_bin = MetricsCalculator.binarize(gts[i], threshold)
            if not gt_bin.any():
                m = MetricsCalculator._metrics_for_empty_gt(preds[i], empty_gt_policy)
            else:
                tp, fp, fn, tn = MetricsCalculator.confusion_matrix(preds[i], gts[i], threshold)
                m = MetricsCalculator.from_confusion(tp, fp, fn, tn)

            if times is not None:
                m["inference_time"] = float(times[i])
            per_image.append(m)

        # Average with NaN-safe mean to support "skip" policy
        keys = per_image[0].keys()
        avg = {k: float(np.nanmean([m[k] for m in per_image])) for k in keys}

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
            f.write(json.dumps(metrics))

        if stdout:
            for key, value in metrics.items():
                print(f"{key}: mean={value['mean']:.4f}, std={value['std']:.4f}")

        logger.info(f"Metrics summary written to {output_path}")
