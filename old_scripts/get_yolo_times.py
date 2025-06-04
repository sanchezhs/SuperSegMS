import os
import json
import numpy as np

# Ruta base donde se encuentran las configuraciones
base_path = "results/yolo"

# Listado de configuraciones con K-Folds
configurations = [d for d in os.listdir(base_path) if d.endswith("kfolds")]
inference_times_summary = {}
for config in configurations:
    config_path = os.path.join(base_path, config)
    fold_dirs = [f for f in os.listdir(config_path) if f.startswith("fold_") and os.path.isdir(os.path.join(config_path, f))]
    
    all_times = []

    for fold in fold_dirs:
        inference_path = os.path.join(config_path, fold, "predictions", "predict", "inference_times.json")
        if os.path.exists(inference_path):
            with open(inference_path, "r") as f:
                times = json.load(f)
                all_times.extend(times)

    if all_times:
        mean_time = np.mean(all_times)
        std_time = np.std(all_times)
        inference_times_summary[config] = {
            "mean": round(mean_time, 4),
            "std": round(std_time, 4),
            "n_samples": len(all_times)
        }

# Guardar el resumen en un archivo JSON
summary_path = os.path.join(base_path, "inference_times_summary.json")

with open(summary_path, "w") as f:
    json.dump(inference_times_summary, f, indent=4)