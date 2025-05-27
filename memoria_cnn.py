import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Ruta al archivo NIfTI
nifti_path = "datasets/MSLesSeg-Dataset/train/P1/T1/P1_T1_T1.nii.gz"

# Cargar el archivo NIfTI
img = nib.load(nifti_path)
volume = img.get_fdata()

# Simulación simple de un volumen 3D con una "lesión" en el centro
# volume_shape = (256, 256, 20)
# volume = np.zeros(volume_shape)

# Añadir una lesión simulada en algunos cortes
# for z in range(7, 13):  # zona central
#     volume[100:150, 100:150, z] = 1

# Crear una figura con 3 estrategias visuales
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# BASE: todos los slices
base_slices = [volume[:, :, z] for z in range(volume.shape[2])]
axes[0].imshow(np.sum(base_slices, axis=0), cmap='gray')
axes[0].set_title("BASE: todos los cortes")
axes[0].axis('off')

# FILTERED: solo slices con lesión > umbral
threshold = 100
filtered_slices = [volume[:, :, z] for z in range(volume.shape[2])
                   if np.sum(volume[:, :, z]) > threshold]
axes[1].imshow(np.sum(filtered_slices, axis=0), cmap='gray')
axes[1].set_title("FILTERED: cortes con lesión significativa")
axes[1].axis('off')

# TOP 5 SLICES: bloque centrado en el corte con mayor lesión
lesion_areas = [np.sum(volume[:, :, z]) for z in range(volume.shape[2])]
center_z = np.argmax(lesion_areas)
half_block = 2
top5_indices = range(max(0, center_z - half_block), min(volume.shape[2], center_z + half_block + 1))
top5_slices = [volume[:, :, z] for z in top5_indices]
axes[2].imshow(np.sum(top5_slices, axis=0), cmap='gray')
axes[2].set_title("TOP 5: bloque centrado en la mayor lesión")
axes[2].axis('off')

plt.tight_layout()
plt.show()
