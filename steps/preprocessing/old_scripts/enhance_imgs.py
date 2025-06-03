import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Cargar la imagen MRI
# image_path = "datasets/yolo_single_x2/images/train/P2_T1_91.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Aplicar CLAHE para mejorar el contraste
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# clahe_image = clahe.apply(image)

# # Aplicar filtro de realce de bordes (Laplaciano)
# laplacian = cv2.Laplacian(clahe_image, cv2.CV_64F)
# laplacian = np.uint8(np.absolute(laplacian))

# # Aplicar filtro bilateral para reducir ruido manteniendo bordes
# bilateral = cv2.bilateralFilter(clahe_image, 9, 75, 75)

# # Aplicar transformación gamma para mejorar visibilidad de lesiones
# gamma = 1.5  # Mayor que 1 aclara la imagen, menor que 1 la oscurece
# gamma_corrected = np.power(clahe_image / 255.0, gamma) * 255
# gamma_corrected = np.uint8(gamma_corrected)

# # Mostrar las imágenes procesadas
# fig, axs = plt.subplots(1, 5, figsize=(20, 5))

# axs[0].imshow(image, cmap='gray')
# axs[0].set_title("Original MRI")
# axs[0].axis("off")

# axs[1].imshow(clahe_image, cmap='gray')
# axs[1].set_title("CLAHE")
# axs[1].axis("off")

# axs[2].imshow(laplacian, cmap='gray')
# axs[2].set_title("Laplaciano (Bordes)")
# axs[2].axis("off")

# axs[3].imshow(bilateral, cmap='gray')
# axs[3].set_title("Bilateral (Ruido Red.)")
# axs[3].axis("off")

# axs[4].imshow(gamma_corrected, cmap='gray')
# axs[4].set_title("Corrección Gamma")
# axs[4].axis("off")

# plt.show()
def apply_clahe(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img)
    return enhanced_img



def normalize_zscore(image):
    mean, std = np.mean(image), np.std(image)
    return ((image - mean) / std * 255).astype(np.uint8)


def enhance_edges(image):
    blurred = cv2.GaussianBlur(image, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return cv2.addWeighted(image, 1.5, edges, -0.5, 0)

def enhance_mri(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    
    img_norm = normalize_zscore(img_clahe)
    img_edges = enhance_edges(img_norm)
    
    return cv2.addWeighted(img_norm, 0.8, img_edges, 0.2, 0)

# Cargar la imagen MRI
image_path = "datasets/yolo_single_x2/images/train/P2_T1_91.png"


img = enhance_mri(image_path)

plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("MRI Mejorada")
plt.show()