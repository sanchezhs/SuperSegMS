import subprocess
import os
from datetime import datetime
from loguru import logger

# Crear carpeta de logs
os.makedirs("logs", exist_ok=True)

# Configurar loguru para guardar todo en un solo archivo + consola
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_log_path = f"logs/experiments_{timestamp}.log"

logger.remove()  # Quitamos el handler por defecto
logger.add(main_log_path, level="INFO", enqueue=True, backtrace=True, diagnose=True)
logger.add(lambda msg: print(msg, end=""), level="INFO")  # También por consola

# Experimentos A–H y pasos por ejecutar
experiments = [
    #"B", # hecho
    # "E", # hecho
    "F", # todo
    # "I", # hecho
    "J", # todo
]
steps = [
    "preprocess", 
    "train", 
    "predict", 
    "evaluate"
]

for exp_id in experiments:
    logger.info(f"\n=== Ejecutando experimento {exp_id} ===")
    for step in steps:
        logger.info(f">>> Ejecutando: python main.py {exp_id} {step}")
        log_file = f"logs/{exp_id}_{step}_{timestamp}.log"

        with open(log_file, "w") as lf:
            process = subprocess.Popen(
                ["python", "main.py", exp_id, step],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            for line in process.stdout:
                print(line, end="")         # Mostrar en consola
                lf.write(line)              # Guardar en archivo

            process.wait()

        if process.returncode != 0:
            logger.error(f"❌ Error en experimento {exp_id}, paso {step}. Detalles en {log_file}")
            exit(1)

        logger.info(f"✅ {exp_id} {step} completado. Log guardado en {log_file}")
