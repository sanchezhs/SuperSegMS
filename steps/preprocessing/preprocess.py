import os
import cv2
import nibabel as nib
import numpy as np
import json
from collections import namedtuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional

from schemas.pipeline_schemas import (
    PreprocessConfig,
    Net,
    ResizeMethod,
    SuperScale,
    Strategy,
)
from net.FSRCNN.fsrcnn import apply_fsrcnn
from loguru import logger

SplitDirs = namedtuple(
    "SplitDirs",
    [
        "images_train_dir",
        "images_val_dir",
        "images_test_dir",
        "labels_train_dir",
        "labels_val_dir",
        "labels_test_dir",
    ],
)

def create_dirs(dst_path: str) -> SplitDirs:
    """
    Create directories for images and labels for train, val, and test sets.

    :param dst_path: The base directory where the images and labels will be stored.
    :return: A tuple containing the paths to the created directories:
        - images_train_dir
        - images_val_dir
        - images_test_dir
        - labels_train_dir
        - labels_val_dir
        - labels_test_dir
    """
    images_train_dir = os.path.join(dst_path, "images", "train")
    images_val_dir = os.path.join(dst_path, "images", "val")
    images_test_dir = os.path.join(dst_path, "images", "test")
    labels_test_dir = os.path.join(dst_path, "labels", "test")
    labels_train_dir = os.path.join(dst_path, "labels", "train")
    labels_val_dir = os.path.join(dst_path, "labels", "val")

    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(images_test_dir, exist_ok=True)
    os.makedirs(labels_test_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    return namedtuple(
        "SplitDirs",
        [
            "images_train_dir",
            "images_val_dir",
            "images_test_dir",
            "labels_train_dir",
            "labels_val_dir",
            "labels_test_dir",
        ],
    )(
        images_train_dir,
        images_val_dir,
        images_test_dir,
        labels_train_dir,
        labels_val_dir,
        labels_test_dir,
    )

def split_patients_triple(
    patients: list[str], train_frac: float = 0.8, random_state: int = 42
) -> tuple[set[str], set[str], set[str]]:
    """
    Returns 3 sets (train, val, test) without overlap, with val and test being equal.
    """
    # 1) train vs tmp
    train_pat, tmp_pat = train_test_split(
        patients,
        test_size=1 - train_frac,
        random_state=random_state,
    )
    # 2) val vs test (50%–50% de tmp)
    val_pat, test_pat = train_test_split(
        tmp_pat,
        test_size=0.5,
        random_state=random_state,
    )
    return set(train_pat), set(val_pat), set(test_pat)

def get_centered_lesion_block(
    flair_data, mask_data, block_size=5, min_area_threshold=0
) -> tuple[list[np.ndarray], list[int]]:
    """
    Devuelve un bloque de slices centrado en el slice con más lesión.

    Parámetros:
    - flair_data: array 3D del FLAIR (H x W x Z)
    - mask_data: array 3D de la máscara binaria (H x W x Z)
    - block_size: número impar de slices a incluir (ej. 3, 5, 7)
    - min_area_threshold: umbral mínimo de píxeles de lesión

    Retorna:
    - bloque de slices FLAIR (lista de arrays 2D)
    - índices seleccionados
    """
    assert block_size % 2 == 1, "block_size debe ser impar"
    assert flair_data.shape == mask_data.shape, "Dimensiones no coinciden"

    lesion_areas = [
        (i, np.sum(mask_data[:, :, i] > 0)) for i in range(mask_data.shape[2])
    ]
    lesion_areas = [t for t in lesion_areas if t[1] > min_area_threshold]

    if not lesion_areas:
        center_idx = mask_data.shape[2] // 2
    else:
        center_idx = max(lesion_areas, key=lambda x: x[1])[0]

    half = block_size // 2
    start = max(0, center_idx - half)
    end = min(mask_data.shape[2], center_idx + half + 1)

    # Padding if needed (when at borders)
    slices = []
    indices = []
    for i in range(start, end):
        slices.append(flair_data[:, :, i])
        indices.append(i)

    # If not enough slices, pad with zeros (optional)
    while len(slices) < block_size:
        slices.append(np.zeros_like(flair_data[:, :, 0]))
        indices.append(-1)

    return slices, indices


def get_all_lesion_slices(
    flair_data, mask_data, min_area_threshold=50
) -> tuple[list[np.ndarray], list[int]]:
    """
    Devuelve todos los slices que contienen algún área de lesión significativa.
    """
    assert flair_data.shape == mask_data.shape, "Dimensiones no coinciden"

    selected_slices = []
    indices = []

    for i in range(mask_data.shape[2]):
        area = np.sum(mask_data[:, :, i] > 0)
        if area > min_area_threshold:
            selected_slices.append(flair_data[:, :, i])
            indices.append(i)

    if not selected_slices:
        mid = mask_data.shape[2] // 2
        selected_slices = [flair_data[:, :, mid]]
        indices = [mid]

    return selected_slices, indices


def save_image(
    img: np.ndarray,
    img_path: str,
    super_scale: SuperScale,
    resize: tuple[int, int],
    is_flair: Optional[bool] = False,
    resize_method: Optional[ResizeMethod] = None,
) -> None:
    
    interpolation = {
        ResizeMethod.NEAREST: cv2.INTER_NEAREST,
        ResizeMethod.LINEAR: cv2.INTER_LINEAR,
        ResizeMethod.CUBIC: cv2.INTER_CUBIC,
    }.get(resize_method, cv2.INTER_CUBIC)

    if super_scale != SuperScale.NONE:
        img = cv2.resize(img, resize, interpolation=interpolation)
        img = apply_fsrcnn(img, super_scale)
    else:
        img = cv2.resize(img, resize, interpolation=interpolation)

        if is_flair:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = (img > 0).astype(np.uint8) * 255
    cv2.imwrite(img_path, img)


def unet_process_imgs(
    src_path: str,
    dst_path: str,
    super_scale: SuperScale,
    split: float,
    resize: tuple[int, int],
    strategy: Strategy,
    threshold: int,
    resize_method: Optional[ResizeMethod],
) -> None:
    base_path_train = f"{src_path}/train"

    split_dirs = create_dirs(dst_path)
    images_train_dir = split_dirs.images_train_dir
    images_val_dir = split_dirs.images_val_dir
    images_test_dir = split_dirs.images_test_dir
    labels_train_dir = split_dirs.labels_train_dir
    labels_val_dir = split_dirs.labels_val_dir
    labels_test_dir = split_dirs.labels_test_dir

    patients = [
        p
        for p in os.listdir(base_path_train)
        if os.path.isdir(os.path.join(base_path_train, p))
    ]

    train_patients, val_patients, test_patients = split_patients_triple(
        patients, train_frac=split
    )

    # We make sure that the patients in train, val, and test sets are disjoint
    assert train_patients.isdisjoint(val_patients), "Train and val patients overlap"
    assert train_patients.isdisjoint(test_patients), "Train and test patients overlap"
    assert val_patients.isdisjoint(test_patients), "Val and test patients overlap"

    splits = {
        "train": (train_patients, images_train_dir, labels_train_dir),
        "val": (val_patients, images_val_dir, labels_val_dir),
        "test": (test_patients, images_test_dir, labels_test_dir),
    }

    for split, (current_patients, images_dir, labels_dir) in splits.items():
        for patient in tqdm(current_patients, desc=f"Processing unet: {split}"):
            patient_path = os.path.join(base_path_train, patient)

            if not os.path.isdir(patient_path):
                logger.warning(f"Patient {patient} directory not found. Skipping.")
                continue

            for timepoint in os.listdir(patient_path):
                if timepoint.lower() == ".ds_store":
                    # logger.warning(f"Skipping .DS_Store file in {patient_path}")
                    continue

                flair_path = os.path.join(
                    patient_path, timepoint, f"{patient}_{timepoint}_FLAIR.nii.gz"
                )
                mask_path = os.path.join(
                    patient_path, timepoint, f"{patient}_{timepoint}_MASK.nii.gz"
                )

                if not os.path.exists(flair_path) or not os.path.exists(mask_path):
                    logger.warning(
                        f"FLAIR or mask file not found for {patient} at {timepoint}. Skipping."
                    )
                    continue

                flair_nifti = nib.load(flair_path)
                flair_image = flair_nifti.get_fdata()
                mask_nifti = nib.load(mask_path)
                mask_image = mask_nifti.get_fdata()

                if strategy.lower() == Strategy.ALL_SLICES.value:
                    _, best_slice_idxs = get_all_lesion_slices(
                        flair_data=flair_image,
                        mask_data=mask_image,
                        min_area_threshold=threshold,
                    )
                elif strategy.lower() == Strategy.TOP_FIVE.value:
                    _, best_slice_idxs = get_centered_lesion_block(
                        flair_data=flair_image,
                        mask_data=mask_image,
                        block_size=5,
                    )
                else:
                    raise ValueError(f"Invalid strategy {strategy}")

                for best_slice_idx in best_slice_idxs:
                    img_filename = f"{patient}_{timepoint}_{best_slice_idx}.png"
                    save_image(
                        img=flair_image[:, :, best_slice_idx],
                        img_path=os.path.join(images_dir, img_filename),
                        is_flair=True,
                        resize=resize,
                        super_scale=super_scale,
                        resize_method=resize_method,
                    )
                    mask_img_filename = f"{patient}_{timepoint}_{best_slice_idx}.png"
                    save_image(
                        img=mask_image[:, :, best_slice_idx],
                        img_path=os.path.join(labels_dir, mask_img_filename),
                        is_flair=False,
                        resize=resize,
                        super_scale=super_scale,
                        resize_method=resize_method,
                    )

def convert_masks_to_yolo_seg_format(masks_dir: str, output_dir: str, class_id: int = 0):
    """
    Converts all binary mask images in a directory to YOLO segmentation label format.

    Parameters:
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

def process_single_patient(
    patient: str,
    base_path_train: str,
    images_dir: str,
    labels_dir: str,
    strategy: Strategy,
    threshold: int,
    resize: tuple[int, int],
    super_scale: SuperScale,
):
    patient_path = os.path.join(base_path_train, patient)
    for timepoint in os.listdir(patient_path):
        flair_path = os.path.join(patient_path, timepoint, f"{patient}_{timepoint}_FLAIR.nii.gz")
        mask_path = os.path.join(patient_path, timepoint, f"{patient}_{timepoint}_MASK.nii.gz")

        if not os.path.exists(flair_path) or not os.path.exists(mask_path):
            continue

        flair_nifti = nib.load(flair_path)
        flair_image = flair_nifti.get_fdata()
        mask_nifti = nib.load(mask_path)
        mask_image = mask_nifti.get_fdata()

        match strategy.lower():
            case Strategy.ALL_SLICES.value:
                _, best_slice_idxs = get_all_lesion_slices(
                    flair_data=flair_image,
                    mask_data=mask_image,
                    min_area_threshold=threshold,
                )
            case Strategy.TOP_FIVE.value:
                _, best_slice_idxs = get_centered_lesion_block(
                    flair_data=flair_image,
                    mask_data=mask_image,
                    block_size=5,
                )

        for best_slice_idx in best_slice_idxs:
            img_filename = f"{patient}_{timepoint}_{best_slice_idx}.png"
            save_image(
                img=flair_image[:, :, best_slice_idx],
                img_path=os.path.join(images_dir, img_filename),
                is_flair=True,
                resize=resize,
                super_scale=super_scale,
            )

            mask_img_filename = f"{patient}_{timepoint}_{best_slice_idx}.png"
            if np.max(mask_image[:, :, best_slice_idx]) == 0:
                logger.warning(f"Mask for {patient} at {timepoint} is empty. Skipping.")
            save_image(
                img=mask_image[:, :, best_slice_idx],
                img_path=os.path.join(labels_dir, mask_img_filename),
                is_flair=False,
                resize=resize,
                super_scale=super_scale,
            )

def yolo_process_imgs(
    src_path: str,
    dst_path: str,
    super_scale: SuperScale,
    split: float,
    resize: tuple[int, int],
    strategy: Strategy,
    threshold: int,
) -> None:
    base_path_train = f"{src_path}/train"

    split_dirs = create_dirs(dst_path)
    images_train_dir = split_dirs.images_train_dir
    images_val_dir = split_dirs.images_val_dir
    images_test_dir = split_dirs.images_test_dir
    labels_train_dir = split_dirs.labels_train_dir
    labels_val_dir = split_dirs.labels_val_dir
    labels_test_dir = split_dirs.labels_test_dir

    patients = [
        p
        for p in os.listdir(base_path_train)
        if os.path.isdir(os.path.join(base_path_train, p))
    ]

    train_patients, val_patients, test_patients = split_patients_triple(
        patients, train_frac=split
    )

    # We make sure that the patients in train, val, and test sets are disjoint
    assert train_patients.isdisjoint(val_patients), "Train and val patients overlap"
    assert train_patients.isdisjoint(test_patients), "Train and test patients overlap"
    assert val_patients.isdisjoint(test_patients), "Val and test patients overlap"

    splits = {
        "train": (train_patients, images_train_dir, labels_train_dir),
        "val": (val_patients, images_val_dir, labels_val_dir),
        "test": (test_patients, images_test_dir, labels_test_dir),
    }

    for split, (current_patients, images_dir, labels_dir) in splits.items():
        for patient in tqdm(current_patients, desc=f"Processing yolo: {split}"):
            patient_path = os.path.join(base_path_train, patient)
            for timepoint in os.listdir(patient_path):
                flair_path = os.path.join(
                    patient_path, timepoint, f"{patient}_{timepoint}_FLAIR.nii.gz"
                )
                mask_path = os.path.join(
                    patient_path, timepoint, f"{patient}_{timepoint}_MASK.nii.gz"
                )

                if not os.path.exists(flair_path) or not os.path.exists(mask_path):
                    continue

                flair_nifti = nib.load(flair_path)
                flair_image = flair_nifti.get_fdata()
                mask_nifti = nib.load(mask_path)
                mask_image = mask_nifti.get_fdata()

                match strategy.lower():
                    case Strategy.ALL_SLICES.value:
                        _, best_slice_idxs = get_all_lesion_slices(
                            flair_data=flair_image,
                            mask_data=mask_image,
                            min_area_threshold=threshold,
                        )
                    case Strategy.TOP_FIVE.value:
                        _, best_slice_idxs = get_centered_lesion_block(
                            flair_data=flair_image,
                            mask_data=mask_image,
                            block_size=5,
                        )

                for best_slice_idx in best_slice_idxs:
                    img_filename = f"{patient}_{timepoint}_{best_slice_idx}.png"
                    save_image(
                        img=flair_image[:, :, best_slice_idx],
                        img_path=os.path.join(images_dir, img_filename),
                        is_flair=True,
                        resize=resize,
                        super_scale=super_scale,
                    )

                    mask_img_filename = f"{patient}_{timepoint}_{best_slice_idx}.png"
                    if np.max(mask_image[:, :, best_slice_idx]) == 0:
                        logger.warning(
                            f"Mask for {patient} at {timepoint} is empty. Skipping."
                        )
                    save_image(
                        img=mask_image[:, :, best_slice_idx],
                        img_path=os.path.join(labels_dir, mask_img_filename),
                        is_flair=False,
                        resize=resize,
                        super_scale=super_scale,
                    )
        convert_masks_to_yolo_seg_format(
            masks_dir=labels_dir,
            output_dir=labels_dir,
            class_id=0,
        )

def write_preprocess_params(config: PreprocessConfig) -> None:
    with open(os.path.join(config.dst_path, "preprocess_params.json"), "w") as f:
        json.dump(config.model_dump(), f, indent=4)

    logger.info("Preprocess parameters saved to preprocess_params.json")

def preprocess(config: PreprocessConfig) -> None:
    net = config.net
    resize = config.resize
    resize_method = config.resize_method
    super_scale = config.super_scale
    split = config.split
    strategy = config.strategy
    threshold = config.threshold
    src_path = config.src_path
    dst_path = config.dst_path

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

    write_preprocess_params(config)