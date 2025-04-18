import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger

def draw_yolo_labels(image_path: str, label_path: str, class_names: list[str] = None) -> None:
    """
    Reads an image and its YOLO label file, then draws the bounding box,
    polygon, and class label on the image.
    
    Parameters:
    - image_path: path to the image file.
    - label_path: path to the YOLO label text file.
    - class_names: Optional list of class names for labeling; if provided,
      the class id in the label is replaced by the corresponding name.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Error: Cannot load image from {image_path}")
        return

    # Convert image to RGB for matplotlib visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Read the YOLO labels from file
    if not os.path.exists(label_path):
        logger.error(f"Error: Label file {label_path} not found.")
        return
    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()
    
    # Process each label in the file
    for line in lines:
        # Split the line into parts
        parts = line.split()
        if len(parts) < 5:
            continue  # Skip if there are not enough parts
        
        # The first five values correspond to:
        # <class> <x_center> <y_center> <width> <height> (all normalized)
        class_id = parts[0]
        x_center_norm, y_center_norm, box_width_norm, box_height_norm = map(float, parts[1:5])
        
        # Denormalize the bounding box dimensions
        x_center = x_center_norm * width
        y_center = y_center_norm * height
        box_width = box_width_norm * width
        box_height = box_height_norm * height
        
        # Calculate top-left and bottom-right coordinates of the bounding box
        x_min = int(x_center - box_width / 2)
        y_min = int(y_center - box_height / 2)
        x_max = int(x_center + box_width / 2)
        y_max = int(y_center + box_height / 2)
        
        # Draw the bounding box (red rectangle)
        cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
        
        # The remaining values are the polygon coordinates (normalized)
        polygon_coords = list(map(float, parts[5:]))
        if len(polygon_coords) % 2 != 0:
            logger.warning("Warning: Polygon coordinates count is not even.")
            continue
        
        # Convert normalized polygon points to pixel coordinates
        polygon_points = []
        for i in range(0, len(polygon_coords), 2):
            x_point = int(polygon_coords[i] * width)
            y_point = int(polygon_coords[i+1] * height)
            polygon_points.append((x_point, y_point))
        
        # Draw the polygon (green line)
        if len(polygon_points) > 0:
            pts = np.array(polygon_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image_rgb, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Optionally, add the class label text near the bounding box (blue text)
        label_text = class_names[int(class_id)] if class_names and int(class_id) < len(class_names) else class_id
        cv2.putText(image_rgb, label_text, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Display the resulting image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("YOLO Labels Visualization")
    plt.show()

def process_yolo_visualization(image_dir, label_dir, output_dir):
    """
    Procesa todas las imágenes de un directorio y les superpone las etiquetas YOLO correspondientes.

    :param image_dir: Carpeta de imágenes MRI.
    :param label_dir: Carpeta con los archivos de etiquetas YOLO.
    :param output_dir: Carpeta donde se guardarán las imágenes con etiquetas superpuestas.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Obtener lista de imágenes
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))])

    for image_file in tqdm(image_files, desc="Procesando imágenes"):
        base_name = os.path.splitext(image_file)[0]
        label_file = f"{base_name}.txt"

        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)
        output_path = os.path.join(output_dir, f"{base_name}_labeled.png")

        draw_yolo_labels(image_path, label_path, None)

    logger.info("Proceso completado.")

# Ejecutar el procesamiento con rutas ajustadas
image_dir = "datasets/yolo_single_x2/images/train"
label_dir = "datasets/yolo_single_x2/labels/train"
output_dir = "datasets/yolo_single_x2/visualizations"

process_yolo_visualization(image_dir, label_dir, output_dir)
