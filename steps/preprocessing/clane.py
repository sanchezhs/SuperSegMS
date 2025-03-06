import cv2
import matplotlib.pyplot as plt

img_path = "datasets/unet/images/train/P4_T1_74.png"
mask_path = "datasets/unet/labels/train/P4_T1_74.png"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Imagen FLAIR")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Máscara Ground Truth")

plt.show()
