import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys

def view_slices(nii_path):
    # Cargar el archivo NIfTI
    img = nib.load(nii_path)
    data = img.get_fdata()

    # Normalizar la imagen para que esté en el rango [0,1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Obtener el número de slices en el eje axial
    num_slices = data.shape[2]
    current_slice = num_slices // 2  # Empezar en el centro

    def update_slice(slice_idx):
        plt.clf()
        plt.imshow(data[:, :, slice_idx], cmap="gray")
        plt.title(f"Slice {slice_idx + 1}/{num_slices}")
        plt.axis("off")
        plt.draw()

    # Función para manejar las teclas
    def on_key(event):
        nonlocal current_slice
        if event.key == "right" and current_slice < num_slices - 1:
            current_slice += 1
        elif event.key == "left" and current_slice > 0:
            current_slice -= 1
        update_slice(current_slice)

    # Mostrar la primera imagen
    plt.figure(figsize=(6, 6))
    update_slice(current_slice)
    plt.connect("key_press_event", on_key)
    plt.show()

def view_mask(nii_mask_path):
    # Cargar el archivo NIfTI
    img = nib.load(nii_mask_path)
    data = img.get_fdata()

    # Obtener el número de slices en el eje axial
    num_slices = data.shape[2]
    # Get mask with max value
    current_slice = np.argmax(np.sum(data, axis=(0, 1)))
    
    # current_slice = num_slices // 2  # Empezar en el centro

    def update_slice(slice_idx):
        plt.clf()
        plt.imshow(data[:, :, slice_idx], cmap="gray")
        plt.title(f"Slice {slice_idx + 1}/{num_slices}")
        plt.axis("off")
        plt.draw()

    # Función para manejar las teclas
    def on_key(event):
        nonlocal current_slice
        if event.key == "right" and current_slice < num_slices - 1:
            current_slice += 1
        elif event.key == "left" and current_slice > 0:
            current_slice -= 1
        update_slice(current_slice)

    # Mostrar la primera imagen
    plt.figure(figsize=(6, 6))
    update_slice(current_slice)
    plt.connect("key_press_event", on_key)
    plt.show()

# Ejecutar desde línea de comandos
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_nii.py <file_nii.gz>")
    else:
        view_mask(sys.argv[1])
