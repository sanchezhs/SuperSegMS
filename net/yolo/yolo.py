import yaml
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO as uYOLO


class YOLO:
    def __init__(self, config) -> None:
        self.config = config
        self.src_path = config.src_path
        self.dst_path = config.dst_path
        self.yaml_path = os.path.join(self.dst_path, "data.yaml")
        # self.model_path = "./net/yolo/models/yolo11m-seg.pt"
        self.model_path = config.model_path

        os.makedirs(self.dst_path, exist_ok=True)
        self.create_yaml()

    def predict(self) -> None:
        # model = uYOLO(self.model_path)

        # model.predict(
        #     source=os.path.join(self.src_path, "images", "test"),
        #     project=self.dst_path,
        #     save_txt=True,
        #     save_conf=True,
        #     save_crop=False,
        #     device="cuda",
        # )
        self.draw_predictions()
        # self.visualize_predictions()

    def train(self) -> None:
        model = uYOLO(self.model_path)

        model.train(
            data=self.yaml_path,
            epochs=self.config.epochs,
            batch=self.config.batch_size,
            save=True,
            imgsz=256,
            project=self.dst_path,
            device="cuda",
            verbose=True,
        )

    def create_yaml(self) -> None:
        """Crea el archivo data.yaml dentro del directorio de salida."""
        data_yaml = {
            "path": "yolo_single",#self.src_path,
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["lesion"],
            "task": "segment",
        }

        with open(self.yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

    def draw_predictions(self) -> None:

        output_dir = "yolo_res_single/predictions/masks/"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_dir = "datasets/yolo_single/images/test/"
        prediction_dir = "yolo_res_single/predictions/predict/labels/"
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))])
        prediction_files = sorted([f for f in os.listdir(prediction_dir) if f.endswith(".txt")])
        img_size = (256, 256)

        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]  # Nombre sin extensión
            prediction_file = f"{base_name}.txt"

            image_path = os.path.join(image_dir, image_file)
            prediction_path = os.path.join(prediction_dir, prediction_file)

            if not os.path.exists(prediction_path):
                print(f"Advertencia: No se encontró predicción para {image_file}")
                continue

            # Cargar la imagen
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                continue

            img = cv2.resize(img, img_size)  # Redimensionar si es necesario
            mask = np.zeros_like(img)

            # Leer las predicciones
            with open(prediction_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                values = list(map(float, line.split()))

                # Si la línea tiene más de dos valores, se interpreta como coordenadas de segmentación
                if len(values) > 2:
                    class_id = int(values[0])
                    coords = values[1:]

                    if len(coords) % 2 != 0:
                        print(f"Advertencia: Número impar de coordenadas en {line}, descartando último valor.")
                        coords = coords[:-1]

                    if len(coords) >= 4:
                        points = np.array(coords).reshape(-1, 2)
                        points[:, 0] *= img_size[1]
                        points[:, 1] *= img_size[0]
                        points = points.astype(np.int32)

                        # Dibujar la segmentación en la máscara
                        cv2.polylines(mask, [points], isClosed=True, color=255, thickness=1)
                        cv2.fillPoly(mask, [points], color=255)

            # Guardar la imagen con la máscara
            dst_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(dst_path, mask)
            print(f"Guardada: {dst_path}")


    # def visualize_predictions(self):
    #     yolo_predictions_path = "yolo_res_single/predictions/predict/labels"
    #     test_images_path = "datasets/yolo_single/images/test"
    #     command = f"yolo predict model={self.model_path} task=segment overlap_mask=True imgsz=256"

    #     if os.path.exists(yolo_predictions_path) and os.path.exists(test_images_path):
    #         test_images = sorted([f for f in os.listdir(test_images_path) if f.endswith((".png", ".jpg"))])

    #         for image in test_images:
    #             image_path = os.path.join(test_images_path, image)
    #             command += f" source={image_path}"
    #             os.system(command) 
    #     else:
    #         print("Predictions or test images directory not found.")
