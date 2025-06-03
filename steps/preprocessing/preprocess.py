import os
from typing import Optional

from loguru import logger

from schemas.pipeline_schemas import (
    Net,
    PreprocessConfig,
    ResizeMethod,
    Strategy,
    SuperScale,
)
from steps.preprocessing.saving import convert_masks_to_yolo_seg_format
from steps.preprocessing.worker import _run_split_pipeline


def unet_process_imgs(
    src_path: str,
    dst_path: str,
    super_scale: SuperScale,
    split: float,
    resize: tuple[int, int],
    strategy: Strategy,
    threshold: int,
    resize_method: Optional[ResizeMethod] = None,
):
    _run_split_pipeline(
        src_path, dst_path, super_scale, split, resize, strategy, threshold, resize_method,
        postproc_fn=None,
    )

def yolo_process_imgs(
    src_path: str,
    dst_path: str,
    super_scale: SuperScale,
    split: float,
    resize: tuple[int, int],
    strategy: Strategy,
    threshold: int,
):
    _run_split_pipeline(
        src_path, dst_path, super_scale, split, resize, strategy, threshold, resize_method=None,
        postproc_fn=lambda lbl_dir: convert_masks_to_yolo_seg_format(lbl_dir, lbl_dir, class_id=0),
    )

def preprocess(config: PreprocessConfig) -> None:
    """
    Preprocess images based on the provided configuration.
    Args:
        config: PreprocessConfig object containing all necessary parameters.
    """
    net = config.net
    resize = config.resize
    resize_method = config.resize_method
    super_scale = config.super_scale
    split = config.split
    strategy = config.strategy
    threshold = config.threshold
    src_path = config.src_path
    dst_path = config.dst_path

    if not os.path.exists(src_path):
        logger.info(f"Source path {src_path} does not exist. Creating it.")
        os.makedirs(src_path, exist_ok=True)

    match net:
        case Net.UNET:
            unet_process_imgs(
                src_path=src_path,
                dst_path=dst_path,
                super_scale=super_scale,
                split=split,
                resize=resize,
                resize_method=resize_method,
                threshold=threshold,
                strategy=strategy,
            )
        case Net.YOLO:
            yolo_process_imgs(
                src_path=src_path,
                dst_path=dst_path,
                super_scale=super_scale,
                split=split,
                resize=resize,
                threshold=threshold,
                strategy=strategy,
            )
        case _:
            raise ValueError(f"Invalid net name {net}")

    config.write_config()