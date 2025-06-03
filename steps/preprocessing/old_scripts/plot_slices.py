import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys

def view_mri_and_mask(mri_path, mask_path):
    # Cargar los archivos NIfTI
    mri_img = nib.load(mri_path)
    mask_img = nib.load(mask_path)
    mri_data = mri_img.get_fdata()
    mask_data = mask_img.get_fdata()

    # Normalizar la imagen de MRI
    mri_data = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))

    # Verificar que las dimensiones coincidan
    assert mri_data.shape == mask_data.shape, "MRI and mask must have the same shape"

    # Slice con mayor suma en la máscara
    num_slices = mri_data.shape[2]
    current_slice = np.argmax(np.sum(mask_data, axis=(0, 1)))

    def update_slice(slice_idx):
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(mri_data[:, :, slice_idx], cmap="gray")
        plt.title(f"MRI Slice {slice_idx + 1}/{num_slices}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask_data[:, :, slice_idx], cmap="gray")
        plt.title(f"Mask Slice {slice_idx + 1}/{num_slices}")
        plt.axis("off")

        plt.draw()

    def on_key(event):
        nonlocal current_slice
        if event.key == "right" and current_slice < num_slices - 1:
            current_slice += 1
        elif event.key == "left" and current_slice > 0:
            current_slice -= 1
        update_slice(current_slice)

    # Mostrar la imagen inicial
    plt.figure(figsize=(12, 6))
    update_slice(current_slice)
    plt.connect("key_press_event", on_key)
    plt.show()

# Ejecutar desde línea de comandos
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python view_nii.py <mri_file.nii.gz> <mask_file.nii.gz>")
    else:
        view_mri_and_mask(sys.argv[1], sys.argv[2])
