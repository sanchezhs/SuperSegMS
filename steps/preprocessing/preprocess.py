import os
import cv2
import nibabel as nib
import numpy as np
import json
from collections import namedtuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional, List
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def process_patient_batch(
    patient_batch: List[str],
    base_path_train: str,
    images_dir: str,
    labels_dir: str,
    strategy: Strategy,
    threshold: int,
    resize: tuple[int, int],
    super_scale: SuperScale,
    max_workers: int = 4,  # adjust based on your CPU
):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_patient,
                patient,
                base_path_train,
                images_dir,
                labels_dir,
                strategy,
                threshold,
                resize,
                super_scale,
            )
            for patient in patient_batch
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.exception(f"Exception in patient processing: {e}")


def process_patient_batch_old(
    patient_batch: List[str],
    base_path_train: str,
    images_dir: str,
    labels_dir: str,
    strategy: Strategy,
    threshold: int,
    resize: tuple[int, int],
    super_scale: SuperScale,
):
    for patient in patient_batch:
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
    batch_size: int = 50,
) -> None:
    base_path_train = f"{src_path}/train"
    split_dirs = create_dirs(dst_path)

    patients = [
        p for p in os.listdir(base_path_train)
        if os.path.isdir(os.path.join(base_path_train, p))
    ]

    train_patients, val_patients, test_patients = split_patients_triple(
        patients, train_frac=split
    )

    assert train_patients.isdisjoint(val_patients), "Train and val patients overlap"
    assert train_patients.isdisjoint(test_patients), "Train and test patients overlap"
    assert val_patients.isdisjoint(test_patients), "Val and test patients overlap"

    splits = {
        "train": (train_patients, split_dirs.images_train_dir, split_dirs.labels_train_dir),
        "val": (val_patients, split_dirs.images_val_dir, split_dirs.labels_val_dir),
        "test": (test_patients, split_dirs.images_test_dir, split_dirs.labels_test_dir),
    }

    for split, (current_patients, images_dir, labels_dir) in splits.items():
        current_patients = list(current_patients)
        for i in tqdm(range(0, len(current_patients), batch_size), desc=f"Processing yolo: {split}"):
            logger.info(f"Processing batch {i // batch_size + 1} of {len(current_patients) // batch_size + 1} for {split} set")
            patient_batch = current_patients[i:i+batch_size]
            process_patient_batch(
                patient_batch=patient_batch,
                base_path_train=base_path_train,
                images_dir=images_dir,
                labels_dir=labels_dir,
                strategy=strategy,
                threshold=threshold,
                resize=resize,
                super_scale=super_scale,
            )

        convert_masks_to_yolo_seg_format(
            masks_dir=labels_dir,
            output_dir=labels_dir,
            class_id=0,
        )

# def yolo_process_imgs(
#     src_path: str,
#     dst_path: str,
#     super_scale: SuperScale,
#     split: float,
#     resize: tuple[int, int],
#     strategy: Strategy,
#     threshold: int,
# ) -> None:
#     base_path_train = f"{src_path}/train"

#     split_dirs = create_dirs(dst_path)
#     images_train_dir = split_dirs.images_train_dir
#     images_val_dir = split_dirs.images_val_dir
#     images_test_dir = split_dirs.images_test_dir
#     labels_train_dir = split_dirs.labels_train_dir
#     labels_val_dir = split_dirs.labels_val_dir
#     labels_test_dir = split_dirs.labels_test_dir

#     patients = [
#         p
#         for p in os.listdir(base_path_train)
#         if os.path.isdir(os.path.join(base_path_train, p))
#     ]

#     train_patients, val_patients, test_patients = split_patients_triple(
#         patients, train_frac=split
#     )

#     # We make sure that the patients in train, val, and test sets are disjoint
#     assert train_patients.isdisjoint(val_patients), "Train and val patients overlap"
#     assert train_patients.isdisjoint(test_patients), "Train and test patients overlap"
#     assert val_patients.isdisjoint(test_patients), "Val and test patients overlap"

#     splits = {
#         "train": (train_patients, images_train_dir, labels_train_dir),
#         "val": (val_patients, images_val_dir, labels_val_dir),
#         "test": (test_patients, images_test_dir, labels_test_dir),
#     }

#     for split, (current_patients, images_dir, labels_dir) in splits.items():
#         for patient in tqdm(current_patients, desc=f"Processing yolo: {split}"):
#             patient_path = os.path.join(base_path_train, patient)
#             for timepoint in os.listdir(patient_path):
#                 flair_path = os.path.join(
#                     patient_path, timepoint, f"{patient}_{timepoint}_FLAIR.nii.gz"
#                 )
#                 mask_path = os.path.join(
#                     patient_path, timepoint, f"{patient}_{timepoint}_MASK.nii.gz"
#                 )

#                 if not os.path.exists(flair_path) or not os.path.exists(mask_path):
#                     continue

#                 flair_nifti = nib.load(flair_path)
#                 flair_image = flair_nifti.get_fdata()
#                 mask_nifti = nib.load(mask_path)
#                 mask_image = mask_nifti.get_fdata()

#                 match strategy.lower():
#                     case Strategy.ALL_SLICES.value:
#                         _, best_slice_idxs = get_all_lesion_slices(
#                             flair_data=flair_image,
#                             mask_data=mask_image,
#                             min_area_threshold=threshold,
#                         )
#                     case Strategy.TOP_FIVE.value:
#                         _, best_slice_idxs = get_centered_lesion_block(
#                             flair_data=flair_image,
#                             mask_data=mask_image,
#                             block_size=5,
#                         )

#                 for best_slice_idx in best_slice_idxs:
#                     img_filename = f"{patient}_{timepoint}_{best_slice_idx}.png"
#                     save_image(
#                         img=flair_image[:, :, best_slice_idx],
#                         img_path=os.path.join(images_dir, img_filename),
#                         is_flair=True,
#                         resize=resize,
#                         super_scale=super_scale,
#                     )

#                     mask_img_filename = f"{patient}_{timepoint}_{best_slice_idx}.png"
#                     if np.max(mask_image[:, :, best_slice_idx]) == 0:
#                         logger.warning(
#                             f"Mask for {patient} at {timepoint} is empty. Skipping."
#                         )
#                     save_image(
#                         img=mask_image[:, :, best_slice_idx],
#                         img_path=os.path.join(labels_dir, mask_img_filename),
#                         is_flair=False,
#                         resize=resize,
#                         super_scale=super_scale,
#                     )
#         convert_masks_to_yolo_seg_format(
#             masks_dir=labels_dir,
#             output_dir=labels_dir,
#             class_id=0,
#         )

    # todo borrar abajo
        # convert_segment_masks_to_yolo_seg(
        #     masks_dir=labels_dir,
        #     output_dir=labels_dir,
        #     classes=255,
        # )
    # Mask images pixels are either 0 or 255
    # convert_segment_masks_to_yolo_seg needs masks to be 0 or 1 so we tell it that we have
    # 255 classes. Therefore we need the convert the classes to 0.
    # for split in [labels_train_dir, labels_val_dir, labels_test_dir]:
    #     for filename in os.listdir(split):
    #         if filename.endswith(".txt"):
    #             file_path = os.path.join(split, filename)
    #             fixed_lines = []

    #             with open(file_path, "r") as f:
    #                 lines = f.readlines()

    #             for line in lines:
    #                 parts = line.strip().split()
    #                 if not parts:
    #                     continue
    #                 try:
    #                     float(parts[0])  # check it's a number
    #                     parts[0] = "0"  # force class to 0
    #                     fixed_lines.append(" ".join(parts))
    #                 except ValueError:
    #                     continue

    #             with open(file_path, "w") as f:
    #                 f.write("\n".join(fixed_lines))

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




# def get_top_lesion_slices(flair_data, mask_data, num_slices=1, min_area_threshold=0):
#     """
#     Finds the `num_slices` with the largest lesion areas and returns the corresponding full FLAIR slices.

#     Parameters:
#     - mask_data: Numpy array containing the lesion mask data
#     - flair_data: Numpy array containing the FLAIR image data
#     - num_slices: Number of slices with the largest lesion areas to select
#     - min_area_threshold: Minimum area threshold to consider a lesion significant

#     Returns:
#     - flair_slices: List of NumPy arrays containing the selected full FLAIR slices
#     - selected_indices: List of indices corresponding to the selected slices
#     """
#     assert np.all(
#         np.isin(mask_data, [0, 1])
#     ), "Error: Mask data should only contain 0s and 1s"
#     assert (
#         mask_data.shape == flair_data.shape
#     ), "Error: Mask and FLAIR dimensions do not match"

#     # If the number of slices is less than or equal to 0, return all slices
#     lesion_areas = []
#     for i in range(mask_data.shape[2]):
#         slice_sum = np.sum(mask_data[:, :, i] > 0)
#         lesion_areas.append((i, slice_sum))
#     lesion_areas = [t for t in lesion_areas if t[1] > min_area_threshold]

#     if not lesion_areas:
#         middle_idx = mask_data.shape[2] // 2
#         return [flair_data[:, :, middle_idx]], [middle_idx]

#     lesion_areas.sort(key=lambda x: x[1], reverse=True)
#     selected_slices = lesion_areas[: min(num_slices, len(lesion_areas))]
#     selected_indices = [idx for idx, _ in selected_slices]

#     flair_slices = [flair_data[:, :, idx] for idx in selected_indices]

#     return flair_slices, selected_indices


# https://github.com/diuzhevVlad/mask2yolo/blob/main/converter.py
# def mask_to_yolo(
#     mask: np.ndarray,
#     super_scale: SuperScale,
#     resize: tuple[int, int],
# ) -> list[str]:
#     """Converts a binary mask to YOLO format."""
#     scaled_resize = (resize[1] * super_scale, resize[0] * super_scale)
#     mask = cv2.resize(mask, scaled_resize, interpolation=cv2.INTER_NEAREST)
#     mask = (mask > 0).astype(np.uint8)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     h, w = mask.shape
#     yolo_labels = []

#     for contour in contours:
#         if len(contour) >= 3:
#             polygon = []
#             for point in contour:
#                 x, y = point[0]
#                 polygon.append(f"{x / w:.6f} {y / h:.6f}")

#             # YOLO-Seg format: <class> <x_center> <y_center> <width> <height> <x1> <y1> <x2> <y2> ... <xN> <yN>
#             x_min, y_min = contour.min(axis=0)[0]
#             x_max, y_max = contour.max(axis=0)[0]
#             x_center = (x_min + x_max) / 2 / w
#             y_center = (y_min + y_max) / 2 / h
#             width = (x_max - x_min) / w
#             height = (y_max - y_min) / h

#             label = (
#                 f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} "
#                 + " ".join(polygon)
#             )
#             yolo_labels.append(label)

#     return yolo_labels