import cv2
import numpy as np
import nibabel as nib
import os

# Load Image
img_path = "datasets/MSLesSeg-Dataset/test/P74/P74_T2.nii.gz"
#img_path = "datasets/MSLesSeg-Dataset/train/P1/T1/P1_T1_MASK.nii.gz"

img = nib.load(img_path)
img = img.get_fdata()
img = np.array(img)
img = img.astype(np.uint8)

print(np.unique(img))


# Get image size
print(img.shape)


# # Aplicar CLAHE
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# enhanced_image = clahe.apply(image)

# # Guardar imagen mejorada
# cv2.imwrite("P74_contraste.png", enhanced_image)

# # Aplicar filtro gaussiano
# blurred = cv2.GaussianBlur(enhanced_image, (5,5), 0)

# # Aplicar detección de bordes
# edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# cv2.imwrite("P74_bordes.png", edges)
