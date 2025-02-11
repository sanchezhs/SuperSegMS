import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar imagen
image_path = "datasets/dataset_yolo/images/train/P6_T1.png"
label_path = "datasets/dataset_yolo/labels/train/P6_T1.txt"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# imprimir tamaño de la imagen
print(image.shape)

# Leer etiquetas
with open(label_path, "r") as f:
    labels = f.readlines()

# Dibujar las etiquetas sobre la imagen
plt.figure(figsize=(6,6))
plt.imshow(image, cmap="gray")

for label in labels:
    points = list(map(float, label.strip().split()[1:]))  # Omitir la clase (0)
    points = [(points[i] * image.shape[1], points[i+1] * image.shape[0]) for i in range(0, len(points), 2)]
    
    # Dibujar contorno
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    image = cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

plt.imshow(image)
plt.axis("off")
plt.title("Verificación de etiquetas YOLO")
plt.show()
