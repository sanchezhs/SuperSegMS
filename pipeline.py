import yaml
from enum import StrEnum

from preprocess.preprocess import create_pipeline_from_config

class Steps(StrEnum):
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PREDICTIONS = "prediction"

def main(config: dict) -> None:
    print(config)

    pipeline = create_pipeline_from_config(config)
    steps = [
        x.get('name') for x in config.get('steps', [])
    ]

    if Steps.PREPROCESSING in steps:
        print("Preprocesamiento")
        pipeline.process()
    if Steps.TRAINING in steps:
        print("Entrenamiento")
    if Steps.EVALUATION in steps:
        print("Evaluación")
    if Steps.PREDICTIONS in steps:
        print("Predicciones")

    # pipeline.process()

    # 1. Preprocesamiento
    # if config["preprocessing"]["method"] == "fsrcnn":
    #     print("Aplicando super-resolución con FSRCNN...")
    #     enhance_images()
    # elif config["preprocessing"]["method"] == "basic":
    #     print("Aplicando preprocesado básico...")
    #     preprocess_images()
    # else:
    #     print("No se aplicará preprocesado adicional.")

    # # 2. Preparación del dataset
    # print("Preparando dataset...")
    # prepare_dataset()

    # # 3. Entrenamiento del modelo
    # model_type = config["model"]["type"]
    # if model_type == "yolo":
    #     print("Entrenando modelo YOLO...")
    #     train_yolo()
    # elif model_type == "unet":
    #     print("Entrenando modelo UNet...")
    #     train_unet()
    # else:
    #     raise ValueError(f"Modelo '{model_type}' no reconocido.")

    # # 4. Evaluación del modelo
    # print("Evaluando modelo...")
    # evaluate_model(model_type)

    # # 5. Predicciones
    # print("Realizando predicciones...")
    # make_predictions(model_type)

def load_config() -> dict:
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print("No se ha encontrado el archivo de configuración 'config.yaml'.")
        sys.exit(1)

if __name__ == "__main__":
    config = load_config()
    main(config)
