import cv2
import numpy as np
from typing import List, Optional

from interfaces.interfaces import Transformer
from net.FSRCNN.fsrcnn import FSRCNN, FSRCNN_PATH
from schemas.pipeline_schemas import ResizeMethod, SuperScale

class ResizeTransformer(Transformer):
    def __init__(self, target_size: tuple[int, int], method: Optional[ResizeMethod] = None, super_scale: Optional[SuperScale] = None):
        """
        target_size = (width, height)
        method = ResizeMethod.NEAREST / LINEAR / CUBIC or None (defaults to CUBIC).
        """
        self.target_size = target_size
        self.method = method
        self.super_scale = super_scale if super_scale is not None else SuperScale.NONE

    def transform(self, img: np.ndarray, is_flair: bool) -> np.ndarray:
        scale = self.super_scale.value if not is_flair else 1
        w, h = self.target_size
        target_size = (w * scale, h * scale)

        interp = cv2.INTER_CUBIC if is_flair else cv2.INTER_NEAREST
        resized = cv2.resize(img, target_size, interpolation=interp)
        return resized



class SuperResolutionTransformer(Transformer):
    def __init__(self, super_scale: SuperScale, orig_size: tuple[int, int]):
        """
        If super_scale == SuperScale.NONE → no op. Otherwise apply FSRCNN model.
        """
        self.super_scale = super_scale
        self.orig_size = orig_size  # Original size for resizing before super-resolution
        if super_scale == SuperScale.NONE:
            self.model = None
        else:
            factor = int(super_scale.value)
            self.model = FSRCNN(scale_factor=factor, weights_path=FSRCNN_PATH[factor])

    def transform(self, img: np.ndarray, is_flair: bool) -> np.ndarray:
        if not is_flair or self.super_scale == SuperScale.NONE:
            return img
        if np.max(img) == 0:
            w, h = self.orig_size
            return cv2.resize(img, (int(w * self.super_scale.value), int(h * self.super_scale.value)),
                              interpolation=cv2.INTER_NEAREST)
        return self.model.apply(img)

class FlairNormalizeTransformer(Transformer):
    def transform(self, img: np.ndarray, is_flair: bool) -> np.ndarray:
        if not is_flair:
            return img
        # Normalize to [0,255] as uint8
        img = img.astype(np.float32)
        mn, mx = float(img.min()), float(img.max())
        if mx - mn < 1e-8:
            return np.zeros_like(img, dtype=np.uint8)
        norm = (img - mn) / (mx - mn)
        norm = np.clip(norm * 255, 0, 255).astype(np.uint8)
        return norm


class MaskBinarizeTransformer(Transformer):
    def transform(self, img: np.ndarray, is_flair: bool) -> np.ndarray:
        if is_flair:
            return img

        # Binarize any nonzero to 255
        return ((img > 0).astype(np.uint8)) * 255

# -------------- Composite Transformers -------------- #
class ComposeTransformer(Transformer):
    """Applies a sequence of transformations to an image."""
    def __init__(self, transforms: List[Transformer]):
        self.transforms = transforms

    def transform(self, img: np.ndarray, is_flair: bool) -> np.ndarray:
        out = img
        for t in self.transforms:
            out = t.transform(out, is_flair)
        return out

