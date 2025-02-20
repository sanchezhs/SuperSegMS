import os
import cv2
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional

from schemas.pipeline_schemas import PreprocessConfig, Model

def create_dirs(output_path: str) -> tuple:
    images_train_dir = os.path.join(output_path, "images", "train")
    images_val_dir = os.path.join(output_path, "images", "val")
    images_test_dir = os.path.join(output_path, "images", "test")
    labels_train_dir = os.path.join(output_path, "labels", "train")
    labels_val_dir = os.path.join(output_path, "labels", "val")
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(images_test_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    
    return images_train_dir, images_val_dir, images_test_dir, labels_train_dir, labels_val_dir

def split_patients(patients, test_size: Optional[float] =0.2):
    train_patients, val_patients = train_test_split(patients, test_size=test_size, random_state=42)
    return set(train_patients), set(val_patients)

def save_image(img: np.ndarray, img_path: str, is_flair: Optional[bool] = False, resize: Optional[tuple[int, int]] = None) -> None:
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)

    if is_flair:
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    else:
        img = (img > 0).astype(np.uint8) * 255

    cv2.imwrite(img_path, img)

    # if is_flair:
    #     cv2.imwrite(img_path, (img / img.max() * 255).astype(np.uint8))
    # else:
    #     image = (img > 0).astype(np.uint8) * 255
    #     cv2.imwrite(img_path, image)

def unet_process_train_val(resize: tuple[int, int], split: int, input_path: str, output_path: str):
    base_path_train = f"{input_path}/train"
    images_train_dir, images_val_dir, _, labels_train_dir, labels_val_dir = create_dirs(output_path)
    patients = [p for p in os.listdir(base_path_train) if os.path.isdir(os.path.join(base_path_train, p))]
    train_patients, val_patients = split_patients(patients, test_size=1 - split)
    
    for patient in tqdm(patients, desc="Processing unet"):
        patient_path = os.path.join(base_path_train, patient)
        split = "train" if patient in train_patients else "val"
        images_dir = images_train_dir if split == "train" else images_val_dir
        labels_dir = labels_train_dir if split == "train" else labels_val_dir
        
        for timepoint in os.listdir(patient_path):
            flair_path = os.path.join(patient_path, timepoint, f"{patient}_{timepoint}_FLAIR.nii.gz")
            mask_path = os.path.join(patient_path, timepoint, f"{patient}_{timepoint}_MASK.nii.gz")
            
            if not os.path.exists(flair_path) or not os.path.exists(mask_path):
                continue
            
            # Load the Flair image
            flair_nifti = nib.load(flair_path)
            flair_image = flair_nifti.get_fdata()
            
            mask_nifti = nib.load(mask_path)
            mask_image = mask_nifti.get_fdata()
            
            # Select the best slices
            # We will only consider slices with more than 1% of lesion
            filtered_slices = []
            for i in range(flair_image.shape[2]):
                lesion_mask = mask_image[:, :, i]
                lesion_ratio = np.sum(lesion_mask > 0) / lesion_mask.size 
                
                if lesion_ratio > 0.01:
                    filtered_slices.append(i)

            if len(filtered_slices) == 0:
                continue

            for i in filtered_slices:
                flair_slice = flair_image[:, :, i]
                img_filename = f"{patient}_{timepoint}_{i}.png"
                save_image(flair_slice, os.path.join(images_dir, img_filename), is_flair=True, resize=resize)

                mask_slice = mask_image[:, :, i]
                label_filename = f"{patient}_{timepoint}_{i}.png"
                save_image(mask_slice, os.path.join(labels_dir, label_filename), is_flair=False, resize=resize)


def unet_process_test(resize: tuple[int, int], input_path: str, output_path: str):
    base_path_test = f"{input_path}/test"
    _, _, images_test_dir, _, _ = create_dirs(output_path)

    for patient in tqdm(os.listdir(base_path_test), desc="Processing test for unet"):
        patient_path = os.path.join(base_path_test, patient)
        if not os.path.isdir(patient_path):
            continue
        
        flair_path = os.path.join(patient_path, f"{patient}_FLAIR.nii.gz")
        
        if not os.path.exists(flair_path):
            continue
        
        flair_nifti = nib.load(flair_path)
        flair_image = flair_nifti.get_fdata()

        best_slice_idx = np.argmax(np.sum(flair_image, axis=(0, 1)))
        flair_slice = flair_image[:, :, best_slice_idx]
        img_filename = f"{patient}.png"
        save_image(flair_slice, os.path.join(images_test_dir, img_filename), is_flair=True, resize=resize)

def preprocess(config: PreprocessConfig) -> None:
    model = config.model
    resize = config.resize
    split = config.split
    input_path = config.dataset_path
    output_path = config.processed_dataset_path

    match model:
        case Model.UNET:
            unet_process_train_val(resize, split, input_path, output_path)
            unet_process_test(resize, input_path, output_path)
        case Model.YOLO:
            raise NotImplementedError("Yolo preprocessing not implemented")
            # process_yolo(input_path, output_path)
        case _:
            raise ValueError(f"Invalid model name {model}")