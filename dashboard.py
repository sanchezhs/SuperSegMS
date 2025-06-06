import streamlit as st
import os
import json
from PIL import Image
import nibabel as nib
import numpy as np
import cv2
import tempfile
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

def get_experiments(base_path):
    if not os.path.exists(base_path):
        return []
    return sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

def get_images(pred_path, ext=".png"):
    if not os.path.exists(pred_path):
        return []
    return sorted([f for f in os.listdir(pred_path) if f.endswith(ext)])

def get_prediction_image_path(base_path, exp, subdirs, img_name):
    path = os.path.join(base_path, exp, *subdirs, img_name)
    return path if os.path.exists(path) else None

def display_image(col, img_path, caption):
    if img_path and os.path.exists(img_path):
        col.image(Image.open(img_path), caption=caption, use_container_width=True)
    else:
        col.warning(f"No hay imagen para {caption}.")

def parse_filename(filename):
    base = filename.replace(".png", "")
    patient, tp, slice_id = base.split("_")
    return patient, tp, int(slice_id)

def get_gt_mask_image(patient, timepoint, slice_id):
    nii_path = os.path.join("datasets", "MSLesSeg-Dataset", "train", patient, timepoint, f"{patient}_{timepoint}_MASK.nii.gz")

    if not os.path.exists(nii_path):
        st.error(f"No se encontró el archivo de máscara para {patient} {timepoint}. Asegúrate de que el archivo exista.")
        return None

    try:
        nii = nib.load(nii_path)
        volume = nii.get_fdata()
        if slice_id >= volume.shape[2]:
            return None
        mask_slice = volume[:, :, slice_id]
        mask_img = np.uint8((mask_slice > 0) * 255)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(tmp_file.name, mask_img)
        return tmp_file.name
    except Exception:
        return None

def get_flair_slice(patient, timepoint, slice_id):
    flair_path = os.path.join("datasets", "MSLesSeg-Dataset", "train", patient, timepoint, f"{patient}_{timepoint}_FLAIR.nii.gz")
    if not os.path.exists(flair_path):
        return None

    try:
        nii = nib.load(flair_path)
        volume = nii.get_fdata()
        if slice_id >= volume.shape[2]:
            return None
        flair = volume[:, :, slice_id]
        flair_norm = cv2.normalize(flair, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(flair_norm)
    except Exception:
        st.error(f"Error al cargar la imagen FLAIR para {patient} {timepoint} slice {slice_id}. Asegúrate de que el archivo sea un NIfTI válido.")
        return None

def compute_mask_metrics(pred_mask, gt_mask):
    if pred_mask is None or gt_mask is None:
        return None, None, None, None, None, None

    if pred_mask.dtype == bool:
        pred_mask = pred_mask.astype(np.uint8)
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    if gt_mask.dtype == bool:
        gt_mask = gt_mask.astype(np.uint8)

    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
    dice_score = 2 * np.sum(intersection) / (np.sum(pred_mask) + np.sum(gt_mask)) if (np.sum(pred_mask) + np.sum(gt_mask)) > 0 else 0.0
    precision = np.sum(intersection) / np.sum(pred_mask) if np.sum(pred_mask) > 0 else 0.0
    recall = np.sum(intersection) / np.sum(gt_mask) if np.sum(gt_mask) > 0 else 0.0
    specificity = (np.sum(gt_mask) - np.sum(intersection)) / (np.sum(gt_mask) - np.sum(gt_mask * pred_mask)) if (np.sum(gt_mask) - np.sum(gt_mask * pred_mask)) > 0 else 0.0

    return iou, dice_score, precision, recall, specificity

def load_mask_img(path):
    if not os.path.exists(path):
        return None
    try:
        return np.array(Image.open(path).convert("L")) > 127
    except Exception:
        st.error(f"Error al cargar la máscara desde {path}. Asegúrate de que el archivo sea una imagen válida.")
        return None

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

# Configuración de modelos
MODELS = {
    "U-Net": {
        "base_path": "results/unet",
        "pred_subdirs": ["predictions"],
    },
    "YOLO": {
        "base_path": "results/yolo",
        "pred_subdirs": ["predictions", "masks"],
    },
}

# Sidebar
st.sidebar.title("Opciones de visualización")
experiments = {model: get_experiments(cfg["base_path"]) for model, cfg in MODELS.items()}
selected_exps = {
    model: st.sidebar.selectbox(f"Modelo {model}", experiments[model], key=f"exp_{model}")
    for model in MODELS
}
main_model = "U-Net"
main_pred_path = os.path.join(
    MODELS[main_model]["base_path"],
    selected_exps[main_model],
    *MODELS[main_model]["pred_subdirs"]
)
available_images = get_images(main_pred_path)
available_images = get_images(main_pred_path)
parsed = [parse_filename(f) for f in available_images]

# Construir diccionario: {P1: {T1: [slices]}}
pacientes_dict = {}
for patient, tp, slice_id in parsed:
    pacientes_dict.setdefault(patient, {}).setdefault(tp, []).append((slice_id, f"{patient}_{tp}_{slice_id}.png"))

# Ordenar pacientes y timepoints
for patient in pacientes_dict:
    # Ordenar pacientes por nombre
    pacientes_dict[patient] = dict(sorted(pacientes_dict[patient].items(), key=lambda x: x[0]))
    for tp in pacientes_dict[patient]:
        pacientes_dict[patient][tp].sort(key=lambda x: x[0])  # Ordenar slices por ID

# Sidebar selector por paciente y timepoint
selected_patient = st.sidebar.selectbox("Paciente", sorted(pacientes_dict.keys()))
selected_tp = st.sidebar.selectbox("Timepoint", sorted(pacientes_dict[selected_patient].keys()))

# Selector de slice
slice_options = pacientes_dict[selected_patient][selected_tp]
slice_ids = [s[0] for s in slice_options]
selected_slice = st.sidebar.slider("Slice", min_value=min(slice_ids), max_value=max(slice_ids), value=slice_ids[len(slice_ids)//2])
selected_img = f"{selected_patient}_{selected_tp}_{selected_slice}.png"


# Visualización
st.title(f"Visualización para {selected_img}")
cols = st.columns(len(MODELS) + 1)

# Mostrar imágenes
for idx, (model, cfg) in enumerate(MODELS.items()):
    cols[idx].header(model)
    pred_path = os.path.join(cfg["base_path"], selected_exps[model], *cfg["pred_subdirs"])
    img_path = os.path.join(pred_path, selected_img)
    display_image(cols[idx], img_path, selected_exps[model])

# Mostrar GT
patient, tp, slice_id = parse_filename(selected_img)
gt_img_path = get_gt_mask_image(patient, tp, slice_id)
cols[-1].header("GT Mask")
display_image(cols[-1], gt_img_path, f"{patient}_{tp}_MASK (slice {slice_id})")

st.markdown("---")
st.markdown("## Superposición de máscaras sobre FLAIR")

# Leyenda visual con parches de color
legend_patches = [
    Patch(facecolor=(1, 0, 0), edgecolor='k', label='U-Net (Rojo)'),
    Patch(facecolor=(0, 0, 1), edgecolor='k', label='YOLO (Azul)'),
    Patch(facecolor=(0, 1, 0), edgecolor='k', label='GT (Verde)'),
]

fig_legend, ax_legend = plt.subplots(figsize=(4, 0.5))
ax_legend.axis('off')
ax_legend.legend(handles=legend_patches, loc='center', ncol=3, frameon=False)
st.pyplot(fig_legend)

# Procesamiento
flair_slice = get_flair_slice(patient, tp, slice_id)
if flair_slice is None:
    st.warning("No se pudo cargar la imagen FLAIR para esta slice.")
else:
    H, W = flair_slice.shape
    flair_rgb = np.stack([flair_slice] * 3, axis=-1)

    gt_path = get_gt_mask_image(patient, tp, slice_id)
    gt_mask = load_mask_img(gt_path) if gt_path else None
    gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 if gt_mask is not None else None

    color_gt = (0, 255, 0)
    colors_pred = {
        "U-Net": (255, 0, 0),
        "YOLO": (0, 0, 255),
    }

    model_metrics = {}
    overlays = {}
    titles = {}

    for model in ["U-Net", "YOLO"]:
        cfg = MODELS[model]
        pred_dir = os.path.join(cfg["base_path"], selected_exps[model], *cfg["pred_subdirs"])
        mask_path = os.path.join(pred_dir, selected_img)
        pred_mask = load_mask_img(mask_path)
        model_metrics[model] = {}

        if pred_mask is None:
            st.warning(f"No se encontró la predicción de {model}.")
            overlays[model] = None
            continue

        # Redimensionar y binarizar
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[:, :, 0]
        pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

        # Calcular métricas
        iou, dice_score, precision, recall, specificity = compute_mask_metrics(pred_mask_resized, gt_mask_resized)
        model_metrics[model] = {
            "iou": iou,
            "dice_score": dice_score,
            "precision": precision,
            "recall": recall,
            "specificity": specificity
        }

        # Crear overlay
        overlay = flair_rgb.copy()
        if gt_mask_resized is not None:
            overlay[gt_mask_resized] = (0.6 * overlay[gt_mask_resized] + 0.4 * np.array(color_gt)).astype(np.uint8)
        overlay[pred_mask_resized] = (0.6 * overlay[pred_mask_resized] + 0.4 * np.array(colors_pred[model])).astype(np.uint8)

        overlays[model] = overlay
        titles[model] = f"{model} + GT"

    # Fila de métricas
    metric_cols = st.columns(len(overlays))
    for idx, model in enumerate(["U-Net", "YOLO"]):
        with metric_cols[idx]:
            if model in model_metrics and overlays[model] is not None:
                st.markdown(f"**{model}**")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("IoU", f"{model_metrics[model]['iou']:.3f}")
                    st.metric("Dice", f"{model_metrics[model]['dice_score']:.3f}")
                with m2:
                    st.metric("Recall", f"{model_metrics[model]['recall']:.3f}")
                    st.metric("Precision", f"{model_metrics[model]['precision']:.3f}")
            else:
                st.warning(f"No métricas disponibles para {model}.")

    # Fila de imágenes
    image_cols = st.columns(len(overlays))
    for idx, model in enumerate(["U-Net", "YOLO"]):
        with image_cols[idx]:
            if overlays[model] is not None:
                st.image(overlays[model], caption=titles[model], use_container_width=True)
            else:
                st.warning(f"No se pudo mostrar la superposición para {model}.")

# Métricas y configuración
st.markdown("---")
st.markdown("## Métricas y configuración de cada modelo")

# Guardar métricas para graficar
all_metrics = {}

for model, cfg in MODELS.items():
    st.subheader(f"{model} — {selected_exps[model]}")
    pred_dir = os.path.join(cfg["base_path"], selected_exps[model], *cfg["pred_subdirs"])

    if model == "YOLO":
        pred_dir = pred_dir.replace("masks", "")

    # Cargar métricas
    metrics = load_json(os.path.join(pred_dir, "metrics.json"))
    if metrics:
        all_metrics[model] = metrics  # Guardar para graficar

        st.markdown("**Métricas:**")
        metric_cols = st.columns(3)
        metric_cols[0].metric("IoU", f"{metrics['iou']:.3f}")
        metric_cols[0].metric("Dice", f"{metrics['dice_score']:.3f}")
        metric_cols[1].metric("Precision", f"{metrics['precision']:.3f}")
        metric_cols[1].metric("Recall", f"{metrics['recall']:.3f}")
        metric_cols[2].metric("F1 Score", f"{metrics['f1_score']:.3f}")
        metric_cols[2].metric("Specificity", f"{metrics['specificity']:.3f}")
        st.caption(f"Inference time: {metrics['inference_time']:.4f} s")
    else:
        st.warning("No se encontraron métricas.")
        continue

    # Configuración
    config_path = os.path.join(cfg["base_path"], selected_exps[model], "train_params.json")
    train_cfg = load_json(config_path)
    if train_cfg:
        with st.expander("📄 Configuración del entrenamiento (train_params.json)", expanded=False):
            st.json(train_cfg)
    else:
        st.warning("No se encontró el archivo de configuración.")

# Comparar métricas en gráfico
if len(all_metrics) > 1:
    st.markdown("### 📊 Comparación gráfica de métricas")
    metric_names = ["iou", "dice_score", "precision", "recall", "f1_score", "specificity"]
    display_names = {
        "iou": "IoU",
        "dice_score": "Dice",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1",
        "specificity": "Specificity"
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_width = 0.35
    index = np.arange(len(metric_names))

    for i, (model, metrics) in enumerate(all_metrics.items()):
        values = [metrics[k] for k in metric_names]
        ax.bar(index + i * bar_width, values, bar_width, label=model)

    ax.set_xlabel("Métrica")
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de métricas por modelo")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([display_names[k] for k in metric_names])
    ax.set_ylim(0, 1)
    ax.legend()

    st.pyplot(fig)