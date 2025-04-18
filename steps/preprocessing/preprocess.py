import os
import cv2
import nibabel as nib
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional

from schemas.pipeline_schemas import PreprocessConfig, Net, ResizeMethod, SuperScale
from net.FSRCNN.fsrcnn import apply_fsrcnn
from loguru import logger

def create_dirs(dst_path: str) -> tuple[str, str, str, str, str]:
    images_train_dir = os.path.join(dst_path, "images", "train")
    images_val_dir = os.path.join(dst_path, "images", "val")
    images_test_dir = os.path.join(dst_path, "images", "test")
    labels_train_dir = os.path.join(dst_path, "labels", "train")
    labels_val_dir = os.path.join(dst_path, "labels", "val")

    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(images_test_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    return (
        images_train_dir,
        images_val_dir,
        images_test_dir,
        labels_train_dir,
        labels_val_dir,
    )


def split_patients(patients: list[str], test_size: Optional[float] = 0.2) -> tuple:
    train_patients, val_patients = train_test_split(
        patients, test_size=test_size, random_state=42
    )
    return set(train_patients), set(val_patients)

def get_top_lesion_slices(flair_data, mask_data, num_slices=1, min_area_threshold=0):
    """
    Finds the `num_slices` with the largest lesion areas and returns the corresponding full FLAIR slices.
    
    Parameters:
    - mask_data: Numpy array containing the lesion mask data
    - flair_data: Numpy array containing the FLAIR image data
    - num_slices: Number of slices with the largest lesion areas to select
    - min_area_threshold: Minimum area threshold to consider a lesion significant
    
    Returns:
    - flair_slices: List of NumPy arrays containing the selected full FLAIR slices
    - selected_indices: List of indices corresponding to the selected slices
    """
    assert np.all(np.isin(mask_data, [0, 1])), "Error: Mask data should only contain 0s and 1s"
    assert mask_data.shape == flair_data.shape, "Error: Mask and FLAIR dimensions do not match"
    
    # If the number of slices is less than or equal to 0, return all slices
    if num_slices <= 0:
        return [flair_data], list(range(flair_data.shape[2]))

    lesion_areas = []
    for i in range(mask_data.shape[2]):
        slice_sum = np.sum(mask_data[:, :, i] > 0)
        lesion_areas.append((i, slice_sum))
    lesion_areas = [t for t in lesion_areas if t[1] > min_area_threshold]
    
    if not lesion_areas:
        middle_idx = mask_data.shape[2] // 2
        return [flair_data[:, :, middle_idx]], [middle_idx]
    
    lesion_areas.sort(key=lambda x: x[1], reverse=True)
    selected_slices = lesion_areas[:min(num_slices, len(lesion_areas))]
    selected_indices = [idx for idx, _ in selected_slices]
    
    flair_slices = [flair_data[:, :, idx] for idx in selected_indices]
    
    return flair_slices, selected_indices


def save_image(
    img: np.ndarray,
    img_path: str,
    super_scale: SuperScale,
    resize: tuple[int, int],
    is_flair: Optional[bool] = False,
    resize_method: Optional[ResizeMethod] = None,
) -> None:
    if super_scale != SuperScale.NONE:
        # if resize:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)
        img = apply_fsrcnn(img, super_scale)
    else:
        # if resize:
        #     interpolation = {
        #         ResizeMethod.NEAREST: cv2.INTER_NEAREST,
        #         ResizeMethod.LINEAR: cv2.INTER_LINEAR,
        #         ResizeMethod.CUBIC: cv2.INTER_CUBIC,
        #     }.get(resize_method, cv2.INTER_LINEAR)
        img = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)

        if is_flair:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        else:
            img = (img > 0).astype(np.uint8) * 255

    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(img_path, img)


def unet_process_train_val(
    src_path: str,
    dst_path: str,
    super_scale: SuperScale,
    split: float,
    resize: tuple[int, int],
    top_slices: int,
    resize_method: Optional[ResizeMethod],
) -> None:
    base_path_train = f"{src_path}/train"
    images_train_dir, images_val_dir, _, labels_train_dir, labels_val_dir = create_dirs(
        dst_path
    )
    patients = [
        p
        for p in os.listdir(base_path_train)
        if os.path.isdir(os.path.join(base_path_train, p))
    ]
    train_patients, val_patients = split_patients(patients, test_size=1 - split)

    for patient in tqdm(patients, desc="Processing unet"):
        patient_path = os.path.join(base_path_train, patient)
        patient_split = "train" if patient in train_patients else "val"
        images_dir = images_train_dir if patient_split == "train" else images_val_dir
        labels_dir = labels_train_dir if patient_split == "train" else labels_val_dir

        for timepoint in os.listdir(patient_path):
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
            
            # Load the Flair and mask images
            flair_nifti = nib.load(flair_path)
            flair_image = flair_nifti.get_fdata()
            mask_nifti = nib.load(mask_path)
            mask_image = mask_nifti.get_fdata()
            if patient == "P3" and timepoint == "T3":
                pass
            _, best_slice_idxs = get_top_lesion_slices(
                flair_image, mask_image, num_slices=top_slices
            )

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

                # flair_img_filename = f"{patient}_{timepoint}_{best_slice_idx}.png"
                # save_image(
                #     best_flair_slice,
                #     os.path.join(images_dir, flair_img_filename),
                #     is_flair=True,
                #     resize=resize,
                #     super_scale=super_scale,
                #     resize_method=resize_method,
                # )

                # mask_img_filename = f"{patient}_{timepoint}_{best_slice_idx}.png"
                # save_image(
                #     mask_image[:, :, best_slice_idx],
                #     os.path.join(labels_dir, mask_img_filename),
                #     is_flair=False,
                #     resize=resize,
                #     super_scale=super_scale,
                #     resize_method=resize_method,
                # )


def unet_process_test(
    src_path: str,
    dst_path: str,
    super_scale: SuperScale,
    resize: tuple[int, int],
    resize_method: Optional[ResizeMethod],
) -> None:
    base_path_test = f"{src_path}/test"
    _, _, images_test_dir, _, _ = create_dirs(dst_path)

    for patient in tqdm(os.listdir(base_path_test), desc="Processing test for unet"):
        patient_path = os.path.join(base_path_test, patient)
        if not os.path.isdir(patient_path):
            continue

        flair_path = os.path.join(patient_path, f"{patient}_FLAIR.nii.gz")

        if not os.path.exists(flair_path):
            continue

        flair_nifti = nib.load(flair_path)
        flair_image = flair_nifti.get_fdata()

        best_slice_idx = (
            flair_image.shape[2] // 2
        )
        flair_slice = flair_image[:, :, best_slice_idx]
        img_filename = f"{patient}.png"
        save_image(
            img=flair_slice,
            img_path=os.path.join(images_test_dir, img_filename),
            is_flair=True,
            resize=resize,
            resize_method=resize_method,
            super_scale=super_scale,
        )


def yolo_process_train_val(
    src_path: str,
    dst_path: str,
    super_scale: SuperScale,
    split: float,
    resize: tuple[int, int],
    top_slices: int
) -> None:
    base_path_train = f"{src_path}/train"
    images_train_dir, images_val_dir, _, labels_train_dir, labels_val_dir = create_dirs(
        dst_path
    )
    patients = [
        p
        for p in os.listdir(base_path_train)
        if os.path.isdir(os.path.join(base_path_train, p))
    ]
    train_patients, val_patients = split_patients(patients, test_size=1 - split)

    for patient in tqdm(patients, desc="Processing yolo"):
        patient_path = os.path.join(base_path_train, patient)
        patient_split = "train" if patient in train_patients else "val"
        images_dir = images_train_dir if patient_split == "train" else images_val_dir
        labels_dir = labels_train_dir if patient_split == "train" else labels_val_dir

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

            _, best_slice_idxs = get_top_lesion_slices(
                flair_image, mask_image, num_slices=top_slices
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
                
                mask_img_filename = f"{patient}_{timepoint}_{best_slice_idx}.txt"
                yolo_labels = mask_to_yolo(mask_image[:, : ,best_slice_idx], super_scale, resize)
                
                if yolo_labels:
                    with open(os.path.join(labels_dir, mask_img_filename), "w") as f:
                        f.write("\n".join(yolo_labels))


def yolo_process_test(
    src_path: str,
    dst_path: str,
    super_scale: SuperScale,
    resize: tuple[int, int],
) -> None:
    base_path_test = f"{src_path}/test"
    _, _, images_test_dir, _, _ = create_dirs(dst_path)

    for patient in tqdm(os.listdir(base_path_test), desc="Processing test for yolo"):
        patient_path = os.path.join(base_path_test, patient)
        if not os.path.isdir(patient_path):
            continue

        flair_path = os.path.join(patient_path, f"{patient}_FLAIR.nii.gz")

        if not os.path.exists(flair_path):
            continue

        flair_nifti = nib.load(flair_path)
        flair_image = flair_nifti.get_fdata()

        middle_slice_idx = flair_image.shape[2] // 2
        flair_slice = flair_image[:, :, middle_slice_idx]
        img_filename = f"{patient}.png"
        save_image(
            img=flair_slice,
            img_path=os.path.join(images_test_dir, img_filename),
            is_flair=True,
            resize=resize,
            super_scale=super_scale,
        )


def mask_to_yolo(
    mask: np.ndarray,
    super_scale: SuperScale,
    resize: tuple[int, int],
) -> list[str]:
    """Converts a binary mask to YOLO format."""
    scaled_resize = (resize[1] * super_scale, resize[0] * super_scale)
    mask = cv2.resize(mask, scaled_resize, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    yolo_labels = []

    for contour in contours:
        if len(contour) >= 3:
            polygon = []
            for point in contour:
                x, y = point[0]
                polygon.append(f"{x / w:.6f} {y / h:.6f}")

            # YOLO-Seg format: <class> <x_center> <y_center> <width> <height> <x1> <y1> <x2> <y2> ... <xN> <yN>
            x_min, y_min = contour.min(axis=0)[0]
            x_max, y_max = contour.max(axis=0)[0]
            x_center = (x_min + x_max) / 2 / w
            y_center = (y_min + y_max) / 2 / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h

            label = (
                f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} "
                + " ".join(polygon)
            )
            yolo_labels.append(label)

    return yolo_labels

def write_preprocess_params(config: PreprocessConfig) -> None:
    with open(os.path.join(config.dst_path, "preprocess_params.json"), "w") as f:
        json.dump(config.as_dict(), f, indent=4)

    logger.info("Preprocess parameters saved to preprocess_params.json")

def preprocess(config: PreprocessConfig) -> None:
    net = config.net
    resize = config.resize
    resize_method = config.resize_method
    super_scale = config.super_scale
    split = config.split
    top_slices = config.top_slices
    src_path = config.src_path
    dst_path = config.dst_path

    match net:
        case Net.UNET:
            unet_process_train_val(
                src_path=src_path, dst_path=dst_path, super_scale=super_scale, split=split, resize=resize, resize_method=resize_method, top_slices=top_slices
            )
            unet_process_test(src_path=src_path, dst_path=dst_path, super_scale=super_scale, resize=resize, resize_method=resize_method)
        case Net.YOLO:
            yolo_process_train_val(src_path=src_path, dst_path=dst_path, super_scale=super_scale, split=split, resize=resize, top_slices=top_slices)
            yolo_process_test(src_path=src_path, dst_path=dst_path, super_scale=super_scale, resize=resize)
        case _:
            raise ValueError(f"Invalid net name {net}")

    write_preprocess_params(config)