import yaml
import os
from ultralytics import YOLO as uYOLO


class YOLO:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.dataset_path
        self.output_path = config.output_path  # Directorio de salida
        self.yaml_path = os.path.join(self.output_path, "data.yaml")  # Archivo YAML

        # Crear directorio de salida si no existe
        os.makedirs(self.output_path, exist_ok=True)

        self.create_yaml()

    def train(self):
        model = uYOLO("./net/yolo/models/yolo11m-seg.pt")

        model.train(
            data=self.yaml_path,
            epochs=self.config.epochs,
            batch=self.config.batch_size,
            save=True,
            project=self.output_path,
            device="cuda",
        )

    def create_yaml(self):
        """Crea el archivo data.yaml dentro del directorio de salida."""
        data_yaml = {
            "path": "yolo", #self.dataset_path,
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["lesion"],
            "task": "segment",
        }

        with open(self.yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
