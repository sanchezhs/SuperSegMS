from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Callable, Optional
import nibabel as nib

from loguru import logger
import numpy as np
from tqdm import tqdm
from schemas.pipeline_schemas import ResizeMethod, Strategy, SuperScale
from steps.preprocessing.saving import save_image
from steps.preprocessing.slice_selection import get_all_lesion_slices, get_centered_lesion_block
from steps.preprocessing.splitting import create_dirs, split_patients_triple

def process_patient(
    patient: str,
    base_path_train: str,
    images_dir: str,
    labels_dir: str,
    strategy: Strategy,
    threshold: int,
    resize: tuple[int, int],
    super_scale: SuperScale,
    resize_method: Optional[ResizeMethod] = None,
) -> None:
    """
    Load one patient's timepoints, pick best slices, and save both FLAIR and mask PNGs.
    Args:
        patient: Patient ID (string).
        base_path_train: Base path to the training data.
        images_dir: Directory to save FLAIR images.
        labels_dir: Directory to save mask images.
        strategy: Strategy for selecting slices (e.g., all, lesion, top five).
        threshold: Minimum lesion area threshold for slice selection.
        resize: Tuple (width, height) for resizing images.
        super_scale: Super-resolution scaling factor.
        resize_method: Method to use for resizing (if None, defaults to cubic).
    """
    patient_path = os.path.join(base_path_train, patient)
    if not os.path.isdir(patient_path):
        logger.warning(f"Patient {patient} directory not found. Skipping.")
        return

    for timepoint in os.listdir(patient_path):
        flair_path = os.path.join(patient_path, timepoint, f"{patient}_{timepoint}_FLAIR.nii.gz")
        mask_path  = os.path.join(patient_path, timepoint, f"{patient}_{timepoint}_MASK.nii.gz")

        if not os.path.exists(flair_path) or not os.path.exists(mask_path):
            logger.warning(f"Missing FLAIR or mask for {patient}@{timepoint}, skipping.")
            continue

        flair_img = nib.load(flair_path).get_fdata()
        mask_img  = nib.load(mask_path).get_fdata()

        # decide which slices to keep
        match strategy:
            case Strategy.ALL_SLICES:
                _, slice_idxs = get_all_lesion_slices(flair_img, mask_img, only_lesion_slices=False)
            case Strategy.LESION:
                _, slice_idxs = get_all_lesion_slices(flair_img, mask_img, only_lesion_slices=True)
            case Strategy.LESION_BLOCK:
                _, slice_idxs = get_centered_lesion_block(flair_img, mask_img, block_size=5)
            case _:
                raise ValueError(f"Invalid strategy {strategy}")

        # save both flair & mask in parallel
        for idx in slice_idxs:
            base_fn = f"{patient}_{timepoint}_{idx}.png"
            save_image(
                img=flair_img[:, :, idx],
                img_path=os.path.join(images_dir, base_fn),
                super_scale=super_scale,
                resize=resize,
                is_flair=True,
                resize_method=resize_method,
            )
            # skip empty masks
            if np.max(mask_img[:, :, idx]) == 0:
                logger.warning(f"Empty mask for {patient}@{timepoint}@slice{idx}")
            save_image(
                img=mask_img[:, :, idx],
                img_path=os.path.join(labels_dir, base_fn),
                super_scale=super_scale,
                resize=resize,
                is_flair=False,
                resize_method=resize_method,
            )


def _batch_process_split(
    name: str,
    patients: set[str],
    base_path_train: str,
    images_dir: str,
    labels_dir: str,
    strategy: Strategy,
    threshold: int,
    resize: tuple[int, int],
    super_scale: SuperScale,
    resize_method: Optional[ResizeMethod],
    max_workers: Optional[int] = None,
) -> None:
    """
    Submit one thread per patient in this split.
    Args:
        name: Name of the split (e.g., "train", "val", "test").
        patients: Set of patient IDs to process.
        base_path_train: Base path to the training data.
        images_dir: Directory to save FLAIR images.
        labels_dir: Directory to save mask images.
        strategy: Strategy for selecting slices (e.g., all, lesion, top five).
        threshold: Minimum lesion area threshold for slice selection.
        resize: Tuple (width, height) for resizing images.
        super_scale: Super-resolution scaling factor.
        resize_method: Method to use for resizing (if None, defaults to cubic).
        max_workers: Maximum number of threads to use (default is number of CPU cores).
    """
    logger.info(f"Starting batch for split='{name}' with {len(patients)} patients.")
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(
                process_patient,
                pat,
                base_path_train,
                images_dir,
                labels_dir,
                strategy,
                threshold,
                resize,
                super_scale,
                resize_method,
            ): pat
            for pat in patients
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {name}"):
            pat = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.error(f"Patient {pat} failed: {e}")


def _run_split_pipeline(
    src_path: str,
    dst_path: str,
    super_scale: SuperScale,
    split: float,
    resize: tuple[int,int],
    strategy: Strategy,
    threshold: int,
    resize_method: Optional[ResizeMethod],
    postproc_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Common logic for UNet and YOLO pipelines. 
    If postproc_fn is provided, it will be called on each labels_dir after batch processing.
    """
    base = os.path.join(src_path, "train")
    dirs = create_dirs(dst_path)

    patients = [
        p for p in os.listdir(base)
        if os.path.isdir(os.path.join(base, p))
    ]
    train_p, val_p, test_p = split_patients_triple(patients, train_frac=split)

    for a, b in [(train_p,val_p), (train_p,test_p), (val_p,test_p)]:
        assert a.isdisjoint(b), "Overlap in train/val/test sets"

    splits = {
        "train": (train_p, dirs.images_train_dir, dirs.labels_train_dir),
        "val":   (val_p,   dirs.images_val_dir,   dirs.labels_val_dir),
        "test":  (test_p,  dirs.images_test_dir,  dirs.labels_test_dir),
    }

    for split_name, (patient_set, img_dir, lbl_dir) in splits.items():
        _batch_process_split(
            name=split_name,
            patients=patient_set,
            base_path_train=base,
            images_dir=img_dir,
            labels_dir=lbl_dir,
            strategy=strategy,
            threshold=threshold,
            resize=resize,
            super_scale=super_scale,
            resize_method=resize_method,
            max_workers=os.cpu_count(),
        )
        if postproc_fn is not None:
            postproc_fn(lbl_dir)