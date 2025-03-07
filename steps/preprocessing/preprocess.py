import os
import cv2
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional

from schemas.pipeline_schemas import PreprocessConfig, Net, ResizeMethod
from net.FSRCNN.fsrcnn import apply_fsrcnn


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


def save_image(
    img: np.ndarray,
    img_path: str,
    is_flair: Optional[bool] = False,
    resize: Optional[tuple[int, int]] = None,
    super_scale: Optional[int] = None,
    resize_method: Optional[ResizeMethod] = None,
) -> None:
    if super_scale:
        if resize:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)
        img = apply_fsrcnn(img, super_scale)
    else:
        if resize:
            interpolation = {
                ResizeMethod.NEAREST: cv2.INTER_NEAREST,
                ResizeMethod.LINEAR: cv2.INTER_LINEAR,
                ResizeMethod.CUBIC: cv2.INTER_CUBIC,
            }.get(resize_method, cv2.INTER_LINEAR)
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
    resize: Optional[tuple[int, int]],
    resize_method: Optional[ResizeMethod],
    super_scale: Optional[int],
    split: float,
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
                continue

            # Load the Flair image
            flair_nifti = nib.load(flair_path)
            flair_image = flair_nifti.get_fdata()

            mask_nifti = nib.load(mask_path)
            mask_image = mask_nifti.get_fdata()

            middle_slice_idx = flair_image.shape[2] // 2
            flair_slice = flair_image[:, :, middle_slice_idx]
            img_filename = f"{patient}_{timepoint}_{middle_slice_idx}.png"
            save_image(
                flair_slice,
                os.path.join(images_dir, img_filename),
                is_flair=True,
                resize=resize,
                super_scale=super_scale,
                resize_method=resize_method,
            )

            mask_slice = mask_image[:, :, middle_slice_idx]
            label_filename = f"{patient}_{timepoint}_{middle_slice_idx}.png"
            save_image(
                mask_slice,
                os.path.join(labels_dir, label_filename),
                is_flair=False,
                resize=resize,
                super_scale=super_scale,
                resize_method=resize_method,
            )

            # Select the best slices
            # We will only consider slices with more than 1% of lesion
            # filtered_slices = []
            # for i in range(flair_image.shape[2]):
            #     lesion_mask = mask_image[:, :, i]
            #     lesion_ratio = np.sum(lesion_mask > 0) / lesion_mask.size

            #     if lesion_ratio > 0.01:
            #         filtered_slices.append(i)

            # if len(filtered_slices) == 0:
            #     continue

            # for i in filtered_slices:
            #     flair_slice = flair_image[:, :, i]
            #     img_filename = f"{patient}_{timepoint}_{i}.png"
            #     save_image(
            #         flair_slice,
            #         os.path.join(images_dir, img_filename),
            #         is_flair=True,
            #         resize=resize,
            #         super_scale=super_scale,
            #         resize_method=resize_method,
            #     )
            #     mask_slice = mask_image[:, :, i]
            #     label_filename = f"{patient}_{timepoint}_{i}.png"
            #     save_image(
            #         mask_slice,
            #         os.path.join(labels_dir, label_filename),
            #         is_flair=False,
            #         resize=resize,
            #         super_scale=super_scale,
            #         resize_method=resize_method,
            #     )


def unet_process_test(
    src_path: str,
    dst_path: str,
    resize: Optional[tuple[int, int]],
    super_scale: Optional[int],
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
        )  # np.argmax(np.sum(flair_image, axis=(0, 1)))
        flair_slice = flair_image[:, :, best_slice_idx]
        img_filename = f"{patient}.png"
        save_image(
            flair_slice,
            os.path.join(images_test_dir, img_filename),
            is_flair=True,
            resize=resize,
            resize_method=resize_method,
            super_scale=super_scale,
        )


def yolo_process_train_val(
    src_path: str, dst_path: str, resize: Optional[tuple[int, int]], super_scale: Optional[int], split: float
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

            middle_slice_idx = flair_image.shape[2] // 2
            flair_slice = flair_image[:, :, middle_slice_idx]
            img_filename = f"{patient}_{timepoint}_{middle_slice_idx}.png"
            save_image(
                flair_slice,
                os.path.join(images_dir, img_filename),
                is_flair=True,
                resize=resize,
                super_scale=super_scale,
            )

            mask_slice = mask_image[:, :, middle_slice_idx]
            label_filename = f"{patient}_{timepoint}_{middle_slice_idx}.txt"
            yolo_labels = mask_to_yolo(mask_slice, resize, super_scale)
            if yolo_labels:
                with open(os.path.join(labels_dir, label_filename), "w") as f:
                    f.write("\n".join(yolo_labels))


def yolo_process_test(
    src_path: str, dst_path: str, resize: Optional[tuple[int, int]], super_scale: Optional[int]
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
            flair_slice,
            os.path.join(images_test_dir, img_filename),
            is_flair=True,
            resize=resize,
            super_scale=super_scale,
        )


def mask_to_yolo(
    mask: np.ndarray, resize: Optional[tuple[int, int]] = None, super_scale: Optional[int] = None
) -> list[str]:
    """Converts a binary mask to YOLO format."""
    if resize is not None:
        scaled_resize = (resize[1]*super_scale, resize[0]*super_scale)
        mask = cv2.resize(mask, scaled_resize, interpolation=cv2.INTER_NEAREST)
        # if super_scale:
        #     mask = apply_fsrcnn(mask, super_scale)

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


def preprocess(config: PreprocessConfig) -> None:
    net = config.net
    resize = config.resize
    resize_method = config.resize_method
    super_scale = config.super_scale
    split = config.split
    src_path = config.src_path
    dst_path = config.dst_path

    match net:
        case Net.UNET:
            unet_process_train_val(
                src_path, dst_path, resize, resize_method, super_scale, split
            )
            unet_process_test(src_path, dst_path, resize, super_scale, resize_method)
        case Net.YOLO:
            yolo_process_train_val(src_path, dst_path, resize, super_scale, split)
            yolo_process_test(src_path, dst_path, resize, super_scale)
        case _:
            raise ValueError(f"Invalid net name {net}")
