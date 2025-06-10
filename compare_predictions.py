# COMPARAR PREDICCIONES DE UNET Y YOLO CON IMÁGENES DE MRI
# import sys
# import os
# import cv2
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from loguru import logger

# # Global index to track which patient is displayed
# current_index = 0
# patient_ids = []

# def load_image(image_path, is_mask=False):
#     """Load an image as grayscale."""
#     if not os.path.exists(image_path):
#         logger.warning(f"❌ Error: File not found -> {image_path}")
#         return None
    
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         logger.warning(f"❌ Error: Unable to read image -> {image_path}")
#         return None

#     if is_mask:
#         img = (img > 127).astype(np.uint8)

#     return img

# def overlay_mask_on_mri(mri, mask, alpha=0.5):
#     """Overlay mask on MRI with transparency."""
#     colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
#     colored_mask[mask > 0] = [255, 0, 0]  # Red color for mask

#     mri_colored = cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR)
#     overlay = cv2.addWeighted(mri_colored, 1 - alpha, colored_mask, alpha, 0)
    
#     return overlay

# def update_display(fig, axes, mask_dir, mri_dir):
#     """Update display for the current patient index."""
#     global current_index, patient_ids

#     if not patient_ids:
#         return
    
#     patient_id = patient_ids[current_index]
#     pred_mask_img_path = os.path.join(mask_dir, f"{patient_id}.png")
#     mri_img_path = os.path.join(mri_dir, f"{patient_id}.png")

#     mask = load_image(pred_mask_img_path, is_mask=True)
#     mri = load_image(mri_img_path)

#     if mask is None or mri is None:
#         return

#     overlay = overlay_mask_on_mri(mri, mask)

#     axes[0].imshow(mri, cmap="gray")
#     axes[0].set_title("MRI")
#     axes[0].axis("off")

#     axes[1].imshow(overlay)
#     axes[1].set_title("Predicted Mask Overlay")
#     axes[1].axis("off")

#     fig.suptitle(f"Patient {patient_id}", fontsize=14)

#     plt.draw()

# def on_key(event, fig, axes, mask_dir, mri_dir):
#     """Handle key presses for navigation."""
#     global current_index, patient_ids

#     if event.key == "right":
#         current_index = min(current_index + 1, len(patient_ids) - 1)
#     elif event.key == "left":
#         current_index = max(current_index - 1, 0)

#     update_display(fig, axes, mask_dir, mri_dir)

# def compare_directories(mask_dir, mri_dir):
#     """Compare images in the given directories."""
#     global patient_ids, current_index

#     # Extract patient IDs from filenames (assuming format "P{number}.png")
#     mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.startswith("P") and f.endswith(".png")])
#     patient_ids = sorted([f[:-4] for f in mask_filenames])

#     if not patient_ids:
#         print(f"❌ No valid patient images found in {mask_dir}")
#         sys.exit(1)

#     current_index = 0

#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#     update_display(fig, axes, mask_dir, mri_dir)

#     fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, fig, axes, mask_dir, mri_dir))
#     plt.show()

# # python steps/evaluation/compare_predictions.py results/unet/unet_all_100_x2/predictions datasets/unet_all_100_x2/images/test
# # python steps/evaluation/compare_predictions.py results/yolo/yolo_all_100_x2/masks datasets/unet_all_100_x2/images/test
# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python compare.py pred_mask_dir mri_dir")
#         sys.exit(1)

#     mask_dir = sys.argv[1]
#     mri_dir = sys.argv[2]

#     if not os.path.isdir(mask_dir) or not os.path.isdir(mri_dir):
#         print(f"❌ Error: One of the provided paths is not a directory -> {mask_dir}, {mri_dir}")
#         sys.exit(1)

#     print(f"📂 Comparing predicted masks from '{mask_dir}' with ground truth MRIs from '{mri_dir}'")
#     compare_directories(mask_dir, mri_dir)

# # COMPARAR IMAGENES INDIVIDUALES DE PREDICCIONES Y MRIs
# import sys
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from loguru import logger

# def load_image(image_path, is_mask=False):
#     """Load an image as grayscale."""
#     if not os.path.exists(image_path):
#         logger.error(f"❌ Error: File not found -> {image_path}")
#         return None
    
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         logger.error(f"❌ Error: Unable to read image -> {image_path}")
#         return None

#     if is_mask:
#         img = (img > 127).astype(np.uint8)

#     return img

# def overlay_mask_on_mri(mri, mask, alpha=0.5):
#     """Overlay mask on MRI with transparency."""
#     colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
#     colored_mask[mask > 0] = [255, 0, 0]  # Red color for mask

#     mri_colored = cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR)
#     overlay = cv2.addWeighted(mri_colored, 1 - alpha, colored_mask, alpha, 0)
    
#     return overlay

# def display_overlay(mri_path, mask_path):
#     """Display MRI, mask, and overlay."""
#     mri = load_image(mri_path)
#     mask = load_image(mask_path, is_mask=True)

#     if mri is None or mask is None:
#         print("❌ Error loading images.")
#         return

#     overlay = overlay_mask_on_mri(mri, mask)

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     axes[0].imshow(mri, cmap="gray")
#     axes[0].set_title("MRI")
#     axes[0].axis("off")

#     axes[1].imshow(mask, cmap="gray")
#     axes[1].set_title("Máscara")
#     axes[1].axis("off")

#     axes[2].imshow(overlay)
#     axes[2].set_title("Superposición")
#     axes[2].axis("off")

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python overlay.py path/to/mri.png path/to/mask.png")
#         sys.exit(1)

#     mri_path = sys.argv[1]
#     mask_path = sys.argv[2]

#     print(f"📂 Overlaying: MRI='{mri_path}' with Mask='{mask_path}'")
#     display_overlay(mri_path, mask_path)

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def mostrar_segmentacion(imagen_path, mascara_real_path, pred_unet_path, pred_yolo_path, output_path="salida.png"):
    # Cargar imágenes como arrays en escala de grises
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    mascara_real = cv2.imread(mascara_real_path, cv2.IMREAD_GRAYSCALE)
    pred_unet = cv2.imread(pred_unet_path, cv2.IMREAD_GRAYSCALE)
    pred_yolo = cv2.imread(pred_yolo_path, cv2.IMREAD_GRAYSCALE)

    # Verificar que todas las imágenes se cargaron correctamente
    if imagen is None or mascara_real is None or pred_unet is None or pred_yolo is None:
        raise ValueError("No se pudo cargar alguna de las imágenes proporcionadas.")

    # Crear figura
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(imagen, cmap='gray')
    axs[0].set_title("Imagen original")
    axs[0].axis('off')

    axs[1].imshow(mascara_real, cmap='gray')
    axs[1].set_title("Máscara real")
    axs[1].axis('off')

    axs[2].imshow(pred_unet, cmap='gray')
    axs[2].set_title("Predicción U-Net")
    axs[2].axis('off')

    axs[3].imshow(pred_yolo, cmap='gray')
    axs[3].set_title("Predicción YOLO")
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

"""
Por mostrar
P13 -> (P13_T1_98.png, P13_T1_100.png, P13_T2_96.png)
P33 -> (P33_T1_97.png, P33_T1_99.png, P33_T2_101.png)
P7 -> (P7_T1_94.png, P7_T2_95.png, P7_T2_96.png)

"""

patients = [
    "P13_T1_98.png",
    "P13_T1_100.png",
    "P13_T2_96.png",
    "P33_T1_97.png",
    "P33_T1_99.png",
    "P33_T2_101.png",
    "P7_T1_94.png",
    "P7_T2_95.png",
    "P7_T2_96.png"
]

# for patient in patients:
#     imagen_path = f"datasets/unet_top5_slice_x2/images/test/{patient}"
#     mascara_real_path = f"datasets/unet_top5_slice_x2/labels/test/{patient}"
#     pred_unet_path = f"results/unet_OLD/unet_top5_slice_x2/predictions/{patient}"
#     pred_yolo_path = f"results/yolo_OLD/yolo_top5_slice_x2/predictions/masks/{patient}"

#     # Extraer el ID del paciente del nombre del archivo
#     patient_id = patient.split('.')[0]  # Asumiendo que el formato es "P{number}_T{number}.png"
#     output_path = f"results/compare_predictions/{patient_id}.png"
    
#     # Asegurarse de que el directorio de salida exista
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     mostrar_segmentacion(imagen_path, mascara_real_path, pred_unet_path, pred_yolo_path, output_path)



def cargar_imagenes(imagen_path, mascara_real_path, pred_unet_path, pred_yolo_path):
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    mascara_real = cv2.imread(mascara_real_path, cv2.IMREAD_GRAYSCALE)
    pred_unet = cv2.imread(pred_unet_path, cv2.IMREAD_GRAYSCALE)
    pred_yolo = cv2.imread(pred_yolo_path, cv2.IMREAD_GRAYSCALE)

    if imagen is None or mascara_real is None or pred_unet is None or pred_yolo is None:
        raise ValueError(f"No se pudo cargar alguna imagen: {imagen_path}, {mascara_real_path}, {pred_unet_path}, {pred_yolo_path}")

    return imagen, mascara_real, pred_unet, pred_yolo

# Lista de pacientes
patients = [
    "P13_T1_98.png",
    "P13_T1_100.png",
    "P13_T2_96.png",

    "P33_T1_97.png",
    "P33_T1_99.png",
    "P33_T2_101.png",

    "P7_T1_94.png",
    "P7_T2_95.png",
    "P7_T2_96.png"
]

# Crear figura con una fila por paciente y 4 columnas
num_patients = len(patients)
fig, axs = plt.subplots(num_patients, 4, figsize=(16, 4 * num_patients))

for i, patient in enumerate(patients):
    patient_id = patient.split('.')[0]  # Extraer el ID del paciente
    patient_time = patient.split('_')[1]  # Extraer el tiempo del paciente
    patient_slice = patient.split('_')[2]  # Extraer el slice del paciente
    imagen_path = f"datasets/unet_top5_slice_x2/images/test/{patient}"
    mascara_real_path = f"datasets/unet_top5_slice_x2/labels/test/{patient}"
    pred_unet_path = f"results/unet_OLD/unet_top5_slice_x2/predictions/{patient}"
    pred_yolo_path = f"results/yolo_OLD/yolo_top5_slice_x2/predictions/masks/{patient}"

    imagen, mascara, unet, yolo = cargar_imagenes(imagen_path, mascara_real_path, pred_unet_path, pred_yolo_path)

    axs[i, 0].imshow(imagen, cmap='gray')
    axs[i, 0].set_title("Imagen original")
    axs[i, 1].imshow(mascara, cmap='gray')
    axs[i, 1].set_title("Máscara real")
    axs[i, 2].imshow(unet, cmap='gray')
    axs[i, 2].set_title("Predicción U-Net")
    axs[i, 3].imshow(yolo, cmap='gray')
    axs[i, 3].set_title("Predicción YOLO")

    for j in range(4):
        axs[i, j].axis('off')

plt.tight_layout()
os.makedirs("results/compare_predictions", exist_ok=True)
plt.savefig("results/compare_predictions/mosaico_completo.png", dpi=150)
plt.close()