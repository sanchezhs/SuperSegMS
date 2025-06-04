import os
from typing import Optional
import cv2
import numpy as np
from net.FSRCNN.fsrcnn import FSRCNN, FSRCNN_PATH
from schemas.pipeline_schemas import ResizeMethod, SuperScale


def save_image(
    img: np.ndarray,
    img_path: str,
    super_scale: SuperScale,
    resize: tuple[int, int],
    is_flair: Optional[bool] = False,
    resize_method: Optional[ResizeMethod] = None,
) -> None:
    """
    Save a single image slice to disk after resizing and applying super-resolution if needed.
    Args:
        img: 2D numpy array representing the image slice.
        img_path: Path where the image will be saved.
        super_scale: Super-resolution scaling factor.
        resize: Tuple (width, height) for resizing the image.
        is_flair: If True, indicates this is a FLAIR image (for normalization).
        resize_method: Method to use for resizing (if None, defaults to cubic).
    """
    interpolation = {
        ResizeMethod.NEAREST: cv2.INTER_NEAREST,
        ResizeMethod.LINEAR: cv2.INTER_LINEAR,
        ResizeMethod.CUBIC: cv2.INTER_CUBIC,
    }.get(resize_method, cv2.INTER_CUBIC)

    img = cv2.resize(img, resize, interpolation=interpolation)
    
    if super_scale != SuperScale.NONE:
        fsrcnn_model = FSRCNN(super_scale, FSRCNN_PATH[super_scale])
        img = fsrcnn_model.apply(img)
    else:
        if is_flair:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = (img > 0).astype(np.uint8) * 255
    cv2.imwrite(img_path, img)

def convert_masks_to_yolo_seg_format(masks_dir: str, output_dir: str, class_id: int = 0) -> None:
    """
    Converts all binary mask images in a directory to YOLO segmentation label format.

    Args:
    - masks_dir: directory containing binary mask PNG images
    - output_dir: directory to write YOLO segmentation .txt label files
    - class_id: class ID to assign to all masks (default 0)
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(masks_dir):
        if not filename.lower().endswith(".png"):
            continue

        mask_path = os.path.join(masks_dir, filename)
        output_txt_path = os.path.join(output_dir, filename.replace(".png", ".txt"))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: could not load mask {mask_path}")
            continue

        height, width = mask.shape
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with open(output_txt_path, 'w') as f:
            for cnt in contours:
                if cnt.shape[0] < 3:
                    continue  # skip too-small polygons

                flattened = cnt.reshape(-1, 2)
                normalized = [(x / width, y / height) for x, y in flattened]
                coords = ' '.join(f"{x:.6f} {y:.6f}" for x, y in normalized)
                f.write(f"{class_id} {coords}\n")
