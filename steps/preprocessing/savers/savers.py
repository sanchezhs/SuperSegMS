# pipeline/savers.py

import cv2
import numpy as np
from pathlib import Path

from interfaces.interfaces import ImageSaver


class PNGSaver(ImageSaver):
    """
    Simply write the 2D array as a PNG (assumes it's uint8 already).
    """
    def save_flair(self, img: np.ndarray, out_path: str) -> None:
        cv2.imwrite(out_path, img)

    def save_mask(self, mask: np.ndarray, out_path: str) -> None:
        cv2.imwrite(out_path, mask)


class YOLOSegSaver(ImageSaver):
    """
    Write mask as PNG, then convert to YOLO polygon format.
    Assumes mask is already a 2D uint8 where foreground==255 and background==0.
    """
    def __init__(self, class_id: int = 0):
        self.class_id = class_id

    def save_flair(self, img: np.ndarray, out_path: str) -> None:
        # same as PNGSaver
        cv2.imwrite(out_path, img)

    def save_mask(self, mask: np.ndarray, out_path: str) -> None:
        # 1) write the binary PNG
        cv2.imwrite(out_path, mask)

        # 2) find contours & write polygon txt
        mask_png = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        if mask_png is None:
            print(f"Warning: could not load mask at {out_path}")
            return

        h, w = mask_png.shape
        _, binary = cv2.threshold(mask_png, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        txt_path = Path(out_path).with_suffix(".txt")
        with open(txt_path, "w") as f:
            for cnt in contours:
                if cnt.shape[0] < 3:
                    continue
                pts = cnt.reshape(-1, 2)
                normalized = [(x / w, y / h) for (x, y) in pts]
                coords = " ".join(f"{x:.6f} {y:.6f}" for (x, y) in normalized)
                f.write(f"{self.class_id} {coords}\n")
