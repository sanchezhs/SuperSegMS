# Pasos a seguir
1. Preprocesamiento del dataset:
    **- Convertir volúmenes 3D en slices 2D correctamente.**
    - Aplicar la super-resolución con FSRCNN.
    - Normalizar y ajustar los datos para la red YOLO.

2. Implementación del modelo:
    - Adaptar YOLO para segmentación semántica en lugar de detección.
    - Diseñar la estrategia de entrenamiento y validación.
    - Ajustar hiperparámetros.

3. Entrenamiento y evaluación:
    - Definir métricas de evaluación (IoU, Dice Score, precisión).
    - Comparar el rendimiento de ambas estrategias (con y sin FSRCNN).

4. Resultados y análisis:
    - Visualización y comparación de segmentaciones.
    - Análisis de errores y posibles mejoras.

5. Redacción y documentación:
    - Redactar la memoria con gráficos, tablas y explicaciones.

# Descripción de las Modalidades de Imagen en el Dataset MSLesSeg

## **FLAIR (Fluid-Attenuated Inversion Recovery)**
FLAIR es una secuencia de resonancia magnética utilizada principalmente en imágenes cerebrales. Su principal característica es que **suprime el líquido cefalorraquídeo (LCR)** para resaltar lesiones en la sustancia blanca.
- **Uso en Esclerosis Múltiple (EM):** Es la modalidad más utilizada para la detección de lesiones, ya que mejora el contraste entre las lesiones y el tejido sano.
- **Apariencia:** Las lesiones suelen aparecer **hiperintensas (brillantes)** respecto a la sustancia blanca circundante.

## **T1 (T1-Weighted Imaging)**
Las imágenes ponderadas en T1 resaltan las diferencias en la relajación longitudinal de los tejidos.
- **Uso en Esclerosis Múltiple:** Se usa principalmente para evaluar la atrofia cerebral y las lesiones crónicas.
- **Apariencia:** Los tejidos con alto contenido de grasa (como la mielina) aparecen **brillantes**, mientras que los líquidos (LCR) se ven **oscuros**.
- **Importancia Clínica:** Es útil para detectar lesiones "negras" o agujeros en T1, que indican daño tisular permanente.

## **T2 (T2-Weighted Imaging)**
Las imágenes T2 ponderadas resaltan las diferencias en la relajación transversal de los tejidos.
- **Uso en Esclerosis Múltiple:** Se utiliza para detectar inflamación activa y lesiones en la sustancia blanca.
- **Apariencia:** El LCR y las lesiones suelen aparecer **hiperintensos (brillantes)**, lo que facilita su detección.

## **MASK (Máscara de Segmentación)**
Las imágenes MASK no son una modalidad de resonancia magnética, sino segmentaciones manuales o automáticas que indican la presencia de lesiones.
- **Uso en Machine Learning:** Sirven como "ground truth" para entrenar modelos de segmentación de lesiones.
- **Apariencia:**
  - **Pixeles blancos (valor alto):** Indican la presencia de una lesión.
  - **Pixeles negros (valor bajo):** Representan tejido sano o áreas sin interés.

## Orientaciones en las Imágenes NIfTI
Las imágenes NIfTI pueden almacenarse en diferentes orientaciones espaciales:
- Axial (Horizontal): Secciones transversales desde la parte superior a la inferior de la cabeza.
- Coronal (Frontal): Secciones que van de la parte frontal a la parte trasera de la cabeza.
- Sagital (Lateral): Secciones que dividen el cerebro en mitades izquierda y derecha.

Formato imágenes: (182, 218, 182) -> (Píxeles, Píxeles, Cantidad de cortes)

# Avances
- La YOLO rinde peor que la UNET

Estado | ID  | Red    | Slice Selection  | Super Resolución
[X]     | A   | UNet   | BASE             | Sin SR 
[X]     | B   | YOLO   | BASE             | Sin SR
[X]     | C   | UNet   | ALL              | Sin SR 
[X]     | D   | UNet   | ALL              | x2
[X]     | E   | YOLO   | ALL              | Sin SR
[X]     | F   | YOLO   | ALL              | x2
[X]     | G   | UNet   | Top 5 slices     | Sin SR
[X]     | H   | UNet   | Top 5 slices     | x2
[X]     | I   | YOLO   | Top 5 slices     | Sin SR
[X]     | J   | YOLO   | Top 5 slices     | x2

# GCLOUD
## Buckets
- Crear bucket para datasets: `gs://tfm-training-datasets/`
    - Copiar los archivos: `gcloud storage cp --recursive ./datasets gs://tfm-training-datasets/datasets/`
- Crear bucket para resultados: `gs://tfm-training-results/`

## Artifacts
- Crear repositorio: `tfm-med-seg-docker`
- Construir la imagen con el tag
`docker build -t europe-west1-docker.pkg.dev/snappy-surf-392910/tfm-med-seg-docker/train-nets:v1 .`
`docker build -t europe-southwest1-docker.pkg.dev/snappy-surf-392910/tfm-med-seg-docker/train-nets:v1 .`
- Subir la imagen con el tag
`docker push europe-southwest1-docker.pkg.dev/snappy-surf-392910/tfm-med-seg-docker/train-nets:v1`

## Instances
- Crear la VM:
```bash
gcloud compute instances create-with-container tfm-vm \
  --zone=us-west2-c \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --boot-disk-size=100GB \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --container-image=europe-southwest1-docker.pkg.dev/snappy-surf-392910/tfm-med-seg-docker/train-nets:v1 \
  --maintenance-policy=TERMINATE \
  --restart-on-failure
```

### Conectarte a la VM y montar los buckets con gcsfuse

gcloud compute ssh tfm-vm --zone=us-west2-c

- Una vez dentro de la VM:
    1. Instalar gcsfuse:
```bash
sudo apt-get update
sudo apt-get install -y gcsfuse
```

b) Crear puntos de montaje y montar buckets:

sudo mkdir -p /mnt/datasets
sudo mkdir -p /mnt/results

gcsfuse tfm-training-datasets /mnt/datasets
gcsfuse tfm-training-results /mnt/results

    Puedes verificar con ls /mnt/datasets

🛠️ 3. (Re)ejecutar tu contenedor con montajes de volumen

Dentro de la VM, para asegurarte de que tu contenedor usa los buckets montados:

docker run -it --rm --gpus all \
  europe-southwest1-docker.pkg.dev/snappy-surf-392910/tfm-med-seg-docker/train-nets:v1

