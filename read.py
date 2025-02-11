import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import gzip
import shutil
import argparse
from tqdm import tqdm


BASE_PATH = "MSLesSeg-Dataset"
OUTPUT_PATH = "processed_slices_v3"
os.makedirs(OUTPUT_PATH, exist_ok=True)
INTENSITY_THRESHOLD = 10
MASK_THRESHOLD = 0.95

MODALITIES = ["FLAIR", "T1", "T2"]

def decompress_nii_gz(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".nii.gz"):
                gz_path = os.path.join(root, file)
                nii_path = gz_path[:-3]  
                
                if not os.path.exists(nii_path):  
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(nii_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"Descomprimido: {gz_path} -> {nii_path}")


def is_relevant_slice(slice_image, is_mask=False):
    if is_mask:
        return np.mean(slice_image) < MASK_THRESHOLD
    return np.sum(slice_image) >= INTENSITY_THRESHOLD

def reorient_to_ras(image_nifti):
    orientation = nib.orientations.axcodes2ornt(nib.aff2axcodes(image_nifti.affine))
    standard_orientation = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = nib.orientations.ornt_transform(orientation, standard_orientation)
    return nib.orientations.apply_orientation(image_nifti.get_fdata(), transform)

def extract_slices(split, patient_id, timepoint, modality, orientation="axial"):
    input_base = os.path.join(BASE_PATH, split, patient_id, timepoint, f"{patient_id}_{timepoint}_{modality}")
    input_path = input_base + ".nii"
    mask_path = os.path.join(BASE_PATH, split, patient_id, timepoint, f"{patient_id}_{timepoint}_MASK.nii")
    
    if not os.path.exists(input_path):
        input_path += ".gz"
    if not os.path.exists(mask_path):
        mask_path += ".gz"
    
    if not os.path.exists(input_path):
        print(f"Archivo no encontrado para {split} {patient_id} {timepoint} {modality}")
        return
    
    # image = nib.load(input_path).get_fdata()
    # mask = nib.load(mask_path).get_fdata() if os.path.exists(mask_path) else None
    image_nifti = nib.load(input_path)
    image = reorient_to_ras(image_nifti)  # Reorientamos la imagen
    mask = reorient_to_ras(nib.load(mask_path)) if os.path.exists(mask_path) else None


    output_patient_path = os.path.join(OUTPUT_PATH, split, patient_id, orientation, modality)
    os.makedirs(output_patient_path, exist_ok=True)
    
    if orientation == "axial":
        num_slices = image.shape[2]
    elif orientation == "coronal":
        num_slices = image.shape[1]
    elif orientation == "sagittal":
        num_slices = image.shape[0]
    else:
        raise ValueError("Orientación no válida. Usa 'axial', 'coronal' o 'sagittal'.")

    best_slice_idx = None
    best_intensity = -np.inf
    best_slice = None
    best_mask = None
    
    for i in range(num_slices):
        if orientation == "axial":
            slice_image = image[:, :, i]
        elif orientation == "coronal":
            slice_image = image[:, i, :]
        elif orientation == "sagittal":
            slice_image = image[i, :, :]
        
        if not is_relevant_slice(slice_image):
            continue

        slice_intensity = np.sum(slice_image)
        if slice_intensity > best_intensity:
            best_intensity = slice_intensity
            best_slice_idx = i
            best_slice = slice_image
            best_mask = mask[:, :, i] if mask is not None else None
    
    if best_slice is not None:
        np.save(os.path.join(output_patient_path, f"{modality}_best_slice.npy"), best_slice)
        plt.imsave(os.path.join(output_patient_path, f"{modality}_best_slice.png"), best_slice, cmap="gray")
        
        if best_mask is not None and is_relevant_slice(best_mask, is_mask=True):
            np.save(os.path.join(output_patient_path, f"mask_best_slice.npy"), best_mask)
            plt.imsave(os.path.join(output_patient_path, f"mask_best_slice.png"), best_mask, cmap="Reds")
    
    print(f"Procesado: {split} {patient_id} {timepoint} {modality}, Mejor Slice: {best_slice_idx} en {orientation}")

def extract_slices_test(split, patient_id, modality, orientation="axial"):
    input_base = os.path.join(BASE_PATH, split, patient_id, f"{patient_id}_{modality}.nii")
    
    if not os.path.exists(input_base):
        input_base += ".gz"
    
    if not os.path.exists(input_base):
        print(f"Archivo no encontrado para {split} {patient_id} {modality}")
        return
    
    image = nib.load(input_base).get_fdata()
    
    output_patient_path = os.path.join(OUTPUT_PATH, split, patient_id, orientation, modality)
    os.makedirs(output_patient_path, exist_ok=True)
    
    if orientation == "axial":
        num_slices = image.shape[2]
    elif orientation == "coronal":
        num_slices = image.shape[1]
    elif orientation == "sagittal":
        num_slices = image.shape[0]
    else:
        raise ValueError("Orientación no válida. Usa 'axial', 'coronal' o 'sagittal'.")
    
    best_slice_idx = None
    best_intensity = -np.inf
    best_slice = None
    
    for i in range(num_slices):
        if orientation == "axial":
            slice_image = image[:, :, i]
        elif orientation == "coronal":
            slice_image = image[:, i, :]
        elif orientation == "sagittal":
            slice_image = image[i, :, :]

        # if not is_relevant_slice(slice_image):
        #     continue

        slice_intensity = np.sum(slice_image)
        if slice_intensity > best_intensity:
            best_intensity = slice_intensity
            best_slice_idx = i
            best_slice = slice_image
    
    if best_slice is not None:
        np.save(os.path.join(output_patient_path, f"{modality}_best_slice.npy"), best_slice)
        plt.imsave(os.path.join(output_patient_path, f"{modality}_best_slice.png"), best_slice, cmap="gray")
    
    print(f"Procesado: {split} {patient_id} {modality}, Mejor Slice: {best_slice_idx}")

def process_dataset(split="all", orientation="axial"):
    splits = ["train", "test"] if split == "all" else [split]
    for split in splits:
        split_path = os.path.join(BASE_PATH, split)
        
        for patient in tqdm(os.listdir(split_path)):
            patient_path = os.path.join(split_path, patient)
            if not os.path.isdir(patient_path):
                continue
            
            for modality in MODALITIES:
                if split == "train":
                    for timepoint in os.listdir(patient_path):
                        extract_slices(split, patient, timepoint, modality, orientation)
                else:
                    extract_slices_test(split, patient, modality, orientation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar imágenes médicas en diferentes orientaciones.")
    parser.add_argument("--orientation", type=str, default="all", choices=["axial", "coronal", "sagittal", "all"],
                        help="Orientación de los cortes a extraer (axial, coronal, sagital, all)")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test"],
                        help="Conjunto de datos a procesar (train, test)")
    args = parser.parse_args()
    
    if args.orientation == "all":
        for orient in ["axial", "coronal", "sagittal"]:
            process_dataset(split=args.split, orientation=orient)
    else:
        process_dataset(split=args.split, orientation=args.orientation)
