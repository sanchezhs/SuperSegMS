# pipeline/interfaces.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np

class PatientSplitter(ABC):
    @abstractmethod
    def split(self, all_patients: List[str]) -> Tuple[set[str], set[str], set[str]]:
        """
        Given a list of patient IDs, return three disjoint sets:
        (train_ids, val_ids, test_ids).
        """
        ...

class SliceSelector(ABC):
    @abstractmethod
    def select_slices(
        self,
        flair_volume: np.ndarray,
        mask_volume: np.ndarray
    ) -> List[int]:
        """
        Given a 3D FLAIR array (H×W×Z) and 3D mask (H×W×Z),
        return a list of slice indices to keep.
        """
        ...

class ImageSaver(ABC):
    @abstractmethod
    def save_flair(
        self,
        img_slice: np.ndarray,
        out_path: Path,
    ) -> None:
        """
        Given a single 2D FLAIR slice, write it to disk (including any resizing,
        normalization, super‐resolution). “out_path” includes the filename.
        """
        ...

    @abstractmethod
    def save_mask(
        self,
        mask_slice: np.ndarray,
        out_path: Path,
    ) -> None:
        """
        Given a binary mask slice, write it to disk (and—if needed—store YOLO .txt
        polygon format instead of raw PNG).
        """
        ...

class Transformer(ABC):
    @abstractmethod
    def transform(self, img: np.ndarray, is_flair: bool) -> np.ndarray:
        """
        Take a 2D NumPy array (flair or mask), apply a transformation,
        and return the new 2D array.
        Args:
            img: input 2D array
            is_flair: whether this is a FLAIR slice (so e.g. normalize differently)
        Returns:
            Transformed 2D array.
        """
        ...