from typing import List
import numpy as np

from interfaces.interfaces import SliceSelector


class AllSlicesSelector(SliceSelector):
    def select_slices(self, flair_volume: np.ndarray, mask_volume: np.ndarray) -> List[int]:
        # Return every slice index [0..Z-1].
        return list(range(flair_volume.shape[2]))


class LesionOnlySelector(SliceSelector):
    def __init__(self, min_area: int = 0):
        self.min_area = min_area

    def select_slices(self, flair_volume: np.ndarray, mask_volume: np.ndarray) -> List[int]:
        Z = mask_volume.shape[2]
        chosen = []
        if not self.min_area or self.min_area < 0:
            self.min_area = 0
        for z in range(Z):
            if np.sum(mask_volume[:, :, z] > 0) > self.min_area:
                chosen.append(z)
        return chosen


class TopCenteredBlockSelector(SliceSelector):
    def __init__(self, block_size: int = 5, min_area: int = 0):
        assert block_size % 2 == 1
        self.block_size = block_size
        self.min_area = min_area

    def select_slices(self, flair_volume: np.ndarray, mask_volume: np.ndarray) -> List[int]:
        _, _, Z = mask_volume.shape
        # Compute lesion area per slice
        area_list = [(i, np.sum(mask_volume[:, :, i] > 0)) for i in range(Z)]
        area_list = [(i, a) for (i, a) in area_list if a > self.min_area]
        if not area_list:
            center_idx = Z // 2
        else:
            center_idx = max(area_list, key=lambda x: x[1])[0]

        half = self.block_size // 2
        start = max(0, center_idx - half)
        end = min(Z, center_idx + half + 1)
        idxs = list(range(start, end))
        # Pad with “-1” if you want exactly block_size indices
        while len(idxs) < self.block_size:
            idxs.append(-1)
        return idxs

class BrainOnlySelector(SliceSelector):
    """
    Keep slices that contain brain tissue (discard black/air slices).
    Uses a simple intensity occupancy test on the FLAIR slice (post-normalization).
    """
    def __init__(self, min_tissue_ratio: float = 0.01, threshold: int = 5):
        """
        min_tissue_ratio: minimum fraction of pixels above threshold to keep the slice.
        threshold: pixel intensity (0-255) to consider "non-black" after your transforms.
        """
        self.min_tissue_ratio = float(min_tissue_ratio)
        self.threshold = int(threshold)

    def select_slices(self, flair_volume: np.ndarray, mask_volume: np.ndarray | None = None) -> list[int]:
        """
        flair_vol: (H, W, D) already loaded (pre-transform). We'll estimate tissue on raw then your pipeline resizes/normalizes later.
        Returns indices of slices likely containing brain tissue.
        """
        H, W, D = flair_volume.shape
        idxs: list[int] = []
        for k in range(D):
            sl = flair_volume[:, :, k]
            # robust scaling to 0..255 for the quick test (avoid relying on absolute NIfTI range)
            sl_norm = sl - np.nanmin(sl)
            den = (np.nanmax(sl_norm) + 1e-8)
            sl_u8 = (sl_norm / den * 255.0).astype(np.uint8)
            # count pixels above small threshold
            tissue = (sl_u8 > self.threshold).sum()
            if tissue / float(H * W) >= self.min_tissue_ratio:
                idxs.append(k)
        return idxs
