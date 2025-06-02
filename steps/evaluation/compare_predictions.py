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

## BUSCAR CASOS DONDE U-NET SUPERA A YOLO EN DICE COEFFICIENT
# import os
# import cv2
# import numpy as np
# from tqdm import tqdm

# def dice_coefficient(pred, true):
#     pred = pred.astype(bool)
#     true = true.astype(bool)
#     intersection = np.logical_and(pred, true).sum()
#     return 2.0 * intersection / (pred.sum() + true.sum() + 1e-8)

# # Rutas
# gt_dir = "datasets/unet_all_100_x2/labels/test"
# unet_dir = "results/unet/unet_all_100_x2/predictions"
# yolo_dir = "results/yolo/yolo_all_100_x2/masks"

# # Umbral de diferencia de Dice para considerar "mejor"
# dice_threshold = 0.01

# # Resultados
# mejores_unet = []

# for fname in tqdm(os.listdir(gt_dir)):
#     if not fname.endswith(".png"):
#         continue

#     # Cargar máscaras binarizadas
#     path_gt = os.path.join(gt_dir, fname)
#     path_unet = os.path.join(unet_dir, fname)
#     path_yolo = os.path.join(yolo_dir, fname)

#     if not os.path.exists(path_unet) or not os.path.exists(path_yolo):
#         continue

#     gt = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE) > 0
#     unet = cv2.imread(path_unet, cv2.IMREAD_GRAYSCALE) > 0
#     yolo = cv2.imread(path_yolo, cv2.IMREAD_GRAYSCALE) > 0

#     # Calcular Dice
#     dice_unet = dice_coefficient(unet, gt)
#     dice_yolo = dice_coefficient(yolo, gt)

#     if dice_unet - dice_yolo > dice_threshold:
#         mejores_unet.append((fname, dice_unet, dice_yolo))

# # Mostrar resultados
# print(f"Total casos donde U-Net supera a YOLO por más de {dice_threshold}: {len(mejores_unet)}\n")
# # Ordenamos por mayor diferencia de Dice
# mejores_unet.sort(key=lambda x: x[1] - x[2], reverse=True)
# for fname, d_unet, d_yolo in mejores_unet:
#     print(f"{fname}: U-Net Dice={d_unet:.3f}, YOLO Dice={d_yolo:.3f}")

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as gridspec

# # Directorio de las máscaras del conjunto de test
gt_dirs = [
    "datasets/unet_all_kfolds/labels/train",
    "datasets/unet_all_kfolds/labels/test",
    "datasets/unet_all_kfolds/labels/val",
]

# # Lista para almacenar áreas de lesión
# lesion_areas = []

# # Recorrer los archivos del directorio
# for fname in tqdm(os.listdir(gt_dir)):
#     if not fname.endswith(".png"):
#         continue

#     path = os.path.join(gt_dir, fname)
#     mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     binary_mask = (mask > 0).astype(np.uint8)

#     area = np.sum(binary_mask)
#     lesion_areas.append(area)

# # Crear histograma
# plt.figure(figsize=(8, 5))
# plt.hist(lesion_areas, bins=50, color='gray', edgecolor='black')
# plt.axvline(100, color='red', linestyle='--', label='Umbral = 100 píxeles')
# plt.xlabel('Área de la lesión (píxeles)')
# plt.ylabel('Frecuencia')
# plt.title('Distribución del área de las lesiones en el conjunto de test')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Umbral de área mínima
# area_threshold = 100

# # Almacenar las máscaras con área < 100
# small_masks = []

# # Recoger rutas de imágenes con área pequeña
# for fname in tqdm(os.listdir(gt_dir)):
#     if not fname.endswith(".png"):
#         continue

#     path = os.path.join(gt_dir, fname)
#     mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     binary_mask = (mask > 0).astype(np.uint8)
#     area = np.sum(binary_mask)

#     if area > 0 and area < area_threshold:
#         small_masks.append((fname, binary_mask))

# # Mostrar las primeras 12 en una cuadrícula
# n = min(12, len(small_masks))
# cols = 4
# rows = int(np.ceil(n / cols))

# fig = plt.figure(figsize=(12, 3 * rows))
# gs = gridspec.GridSpec(rows, cols, wspace=0.3, hspace=0.4)

# for i in range(n):
#     fname, mask = small_masks[i]
#     ax = fig.add_subplot(gs[i])
#     ax.imshow(mask, cmap='gray')
#     ax.set_title(fname, fontsize=8)
#     ax.axis('off')

# plt.suptitle(f'Máscaras con área < {area_threshold} píxeles', fontsize=14)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# Contadores
empty_count = 0
non_empty_count = 0

for gt_dir in gt_dirs:
    # Recorrer los archivos del directorio
    for fname in tqdm(os.listdir(gt_dir)):
        if not fname.endswith(".png"):
            continue

        path = os.path.join(gt_dir, fname)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        # Verificar si la máscara está vacía
        if np.sum(mask > 0) == 0:
            empty_count += 1
        else:
            non_empty_count += 1

    # for fname in os.listdir(gt_dir):
    #     if not fname.endswith(".png"):
    #         continue

    #     mask = cv2.imread(os.path.join(gt_dir, fname), cv2.IMREAD_GRAYSCALE)
    #     if np.sum(mask > 0) == 0:
    #         empty_count += 1
    #     else:
    #         non_empty_count += 1

print(f"Máscaras vacías: {empty_count}")
print(f"Máscaras con lesión: {non_empty_count}")

# Graficar
# labels = ['Máscaras vacías', 'Máscaras con lesión']
# counts = [empty_count, non_empty_count]

# plt.figure(figsize=(6, 6))
# plt.bar(labels, counts)
# plt.title('Frecuencia de máscaras vacías vs. con lesión - Estrategia BASE')
# plt.ylabel('Número de imágenes')
# plt.tight_layout()
# plt.show()