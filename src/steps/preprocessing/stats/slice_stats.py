import math
from typing import Dict, List
import cv2
import numpy as np
import json
from pathlib import Path
from schemas.pipeline_schemas import SliceStats


def compute_slice_stats(img_u8: np.ndarray, mask_u8: np.ndarray,
                        patient_id: str, timepoint: str, idx: int,
                        split: str, strategy: str, signature: str) -> SliceStats:
    """
    Compute statistics for a single slice given the image and mask.
    Args:
        img_u8 (np.ndarray): Grayscale image slice as a 2D numpy array (uint8).
        mask_u8 (np.ndarray): Binary mask slice as a 2D numpy array (uint8).
        patient_id (str): Identifier for the patient.
        timepoint (str): Timepoint identifier.
        idx (int): Slice index.
        split (str): Data split (e.g., 'train', 'val', 'test').
        strategy (str): Preprocessing strategy used.
        signature (str): Preprocessing signature.
    Returns:
        SliceStats: Dataclass containing computed statistics for the slice.
    """
    h, w = mask_u8.shape
    total_px = h * w

    # Ensure binary
    _, bin_m = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)

    lesion_area_px = int((bin_m == 255).sum())
    lesion_ratio = float(lesion_area_px) / float(total_px) if total_px > 0 else 0.0

    num_components = 0
    largest_component_area_px = 0
    bbox_x = bbox_y = bbox_w = bbox_h = 0
    bbox_ratio = 0.0

    if lesion_area_px > 0:
        contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_components = len(contours)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            largest_idx = int(np.argmax(areas))
            largest_component_area_px = int(max(areas))
            x, y, bw, bh = cv2.boundingRect(contours[largest_idx])
            bbox_x, bbox_y, bbox_w, bbox_h = int(x), int(y), int(bw), int(bh)
            bbox_ratio = float(bw * bh) / float(total_px) if total_px > 0 else 0.0

    # Intensity stats
    m_bool = (bin_m == 255)
    inv_bool = ~m_bool
    # Avoid empty masks for means
    if m_bool.any():
        mean_in = float(img_u8[m_bool].mean())
        std_in  = float(img_u8[m_bool].std())
    else:
        mean_in = float("nan")
        std_in  = float("nan")

    if inv_bool.any():
        mean_out = float(img_u8[inv_bool].mean())
        std_out  = float(img_u8[inv_bool].std())
    else:
        mean_out = float("nan")
        std_out  = float("nan")

    contrast = (mean_in - mean_out) if (not math.isnan(mean_in) and not math.isnan(mean_out)) else float("nan")

    return SliceStats(
        patient_id=patient_id, timepoint=timepoint, slice_idx=idx, split=split,
        strategy=strategy, preproc_signature=signature,
        lesion_area_px=lesion_area_px, lesion_ratio=lesion_ratio,
        num_components=num_components, largest_component_area_px=largest_component_area_px,
        bbox_x=bbox_x, bbox_y=bbox_y, bbox_w=bbox_w, bbox_h=bbox_h, bbox_ratio=bbox_ratio,
        mean_in=mean_in, std_in=std_in, mean_out=mean_out, std_out=std_out, contrast=contrast
    )

class StatsCollector:
    """
    Collect and write per-slice statistics.
    """
    def __init__(self) -> None:
        self.rows: List[Dict] = []

    def add(self, row: Dict) -> None:
        self.rows.append(row)

    def write_json(self, path: Path) -> None:
        """
        Write full per-slice statistics to JSON file (nan -> None).

        Args:
            path (Path): Path to the output JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        def _nan_to_none(x):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return x
        sanitized = [
            {k: _nan_to_none(v) for k, v in r.items()}
            for r in self.rows
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=2, ensure_ascii=False)

    def write_summary_json(self, path: Path) -> None:
        """
        Write aggregate statistics (summary) to JSON file (nan -> None).

        Args:
            path (Path): Path to the output JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text(json.dumps({"empty": True}, indent=2))
            return

        def _nan_to_none(x):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return x

        lesion_ratios = [r["lesion_ratio"] for r in self.rows if r["lesion_ratio"] is not None]
        import numpy as np
        arr_ratio = np.array(lesion_ratios, dtype=float)

        summary = {
            "slices_total": len(self.rows),
            "slices_pos": int(sum(1 for r in self.rows if (r["lesion_area_px"] or 0) > 0)),
            "slices_neg": int(sum(1 for r in self.rows if (r["lesion_area_px"] or 0) == 0)),
            "pos_ratio": float((sum(1 for r in self.rows if (r["lesion_area_px"] or 0) > 0) / max(1, len(self.rows)))),
            "lesion_ratio_percentiles": {
                "p50": _nan_to_none(float(np.percentile(arr_ratio, 50))) if arr_ratio.size else None,
                "p75": _nan_to_none(float(np.percentile(arr_ratio, 75))) if arr_ratio.size else None,
                "p90": _nan_to_none(float(np.percentile(arr_ratio, 90))) if arr_ratio.size else None,
                "p95": _nan_to_none(float(np.percentile(arr_ratio, 95))) if arr_ratio.size else None,
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
