import os
import cv2
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Directorios del dataset
BASE_PATH_TRAIN = "MSLesSeg-Dataset/train"
BASE_PATH_TEST = "MSLesSeg-Dataset/test"
OUTPUT_PATH = "datasets"

# Tipos de dataset compatibles
FORMATS = {"yolo": "dataset_yolo", "unet": "dataset_unet"}

# Tamaño de salida de las imágenes
TARGET_SIZE = (640, 640)

def create_dirs(dataset_format):
    dataset_path = os.path.join(OUTPUT_PATH, FORMATS[dataset_format])
    images_train_dir = os.path.join(dataset_path, "images", "train")
    images_val_dir = os.path.join(dataset_path, "images", "val")
    images_test_dir = os.path.join(dataset_path, "images", "test")
    labels_train_dir = os.path.join(dataset_path, "labels", "train")
    labels_val_dir = os.path.join(dataset_path, "labels", "val")
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(images_test_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    
    return images_train_dir, images_val_dir, images_test_dir, labels_train_dir, labels_val_dir

def split_patients(patients, test_size=0.2):
    train_patients, val_patients = train_test_split(patients, test_size=test_size, random_state=42)
    return set(train_patients), set(val_patients)

def mask_to_yolo(mask, img_shape):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img_shape[:2]
    yolo_labels = []
    
    for contour in contours:
        if len(contour) >= 3:
            normalized_contour = [(x / w, y / h) for [[x, y]] in contour]
            flat_contour = [str(coord) for point in normalized_contour for coord in point]
            yolo_labels.append("0 " + " ".join(flat_contour))
    
    return yolo_labels

def process_train_val(dataset_format):
    images_train_dir, images_val_dir, _, labels_train_dir, labels_val_dir = create_dirs(dataset_format)
    patients = [p for p in os.listdir(BASE_PATH_TRAIN) if os.path.isdir(os.path.join(BASE_PATH_TRAIN, p))]
    train_patients, val_patients = split_patients(patients)
    
    for patient in tqdm(patients, desc=f"Processing {dataset_format}"):
        patient_path = os.path.join(BASE_PATH_TRAIN, patient)
        split = "train" if patient in train_patients else "val"
        images_dir = images_train_dir if split == "train" else images_val_dir
        labels_dir = labels_train_dir if split == "train" else labels_val_dir
        
        for timepoint in os.listdir(patient_path):
            flair_path = os.path.join(patient_path, timepoint, f"{patient}_{timepoint}_FLAIR.nii.gz")
            mask_path = os.path.join(patient_path, timepoint, f"{patient}_{timepoint}_MASK.nii.gz")
            
            if not os.path.exists(flair_path) or not os.path.exists(mask_path):
                continue
            
            flair_nifti = nib.load(flair_path)
            mask_nifti = nib.load(mask_path)
            flair_image = flair_nifti.get_fdata()
            mask_image = mask_nifti.get_fdata()
            
            best_slice_idx = np.argmax(np.sum(flair_image, axis=(0, 1)))
            flair_slice = flair_image[:, :, best_slice_idx]
            mask_slice = mask_image[:, :, best_slice_idx]
            
            resized_flair = cv2.resize(flair_slice, TARGET_SIZE)
            # resized_mask = cv2.resize(mask_slice, TARGET_SIZE)
            resized_mask = cv2.resize(mask_slice, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            
            img_filename = f"{patient}_{timepoint}.png"
            img_path = os.path.join(images_dir, img_filename)
            cv2.imwrite(img_path, (resized_flair / resized_flair.max() * 255).astype(np.uint8))
            
            if dataset_format == "yolo":
                mask_filename = f"{patient}_{timepoint}.txt"
                mask_path = os.path.join(labels_dir, mask_filename)
                yolo_labels = mask_to_yolo(resized_mask.astype(np.uint8), TARGET_SIZE)
                
                if yolo_labels:
                    with open(mask_path, "w") as f:
                        f.write("\n".join(yolo_labels))
            elif dataset_format == "unet":
                mask_filename = f"{patient}_{timepoint}.png"
                mask_path = os.path.join(labels_dir, mask_filename)
                cv2.imwrite(mask_path, (resized_mask * 255).astype(np.uint8))

def process_test(dataset_format):
    _, _, images_test_dir, _, _ = create_dirs(dataset_format)
    for patient in tqdm(os.listdir(BASE_PATH_TEST), desc=f"Processing test for {dataset_format}"):
        patient_path = os.path.join(BASE_PATH_TEST, patient)
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
        img_path = os.path.join(images_test_dir, img_filename)
        cv2.imwrite(img_path, (flair_slice / flair_slice.max() * 255).astype(np.uint8))

def process_dataset(dataset_format):
    print(f"Procesando conjunto de entrenamiento y validación para {dataset_format}...")
    process_train_val(dataset_format)
    print(f"Procesando conjunto de prueba para {dataset_format}...")
    process_test(dataset_format)
    print(f"Conversión completada. Dataset listo para {dataset_format}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MSLesSeg dataset")
    parser.add_argument("--net", type=str, choices=["yolo", "unet"],
                        help="Format of the dataset to process")
    args = parser.parse_args()

    if args.net:
        process_dataset(args.net)
    else:
        raise ValueError("You must specify a dataset format to process")