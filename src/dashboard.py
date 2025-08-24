# app.py
import io
import re
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import nibabel as nib
import streamlit as st

# =========================
# Config & helpers
# =========================
st.set_page_config(page_title="MS Lesion Viz", layout="wide")

st.session_state.setdefault("mosaic_rows", [])
if "mosaic_rows" not in st.session_state:
    st.session_state.mosaic_rows = []   # lista de dicts con info mínima para regenerar

DEFAULT_DATASET_DIR = Path("datasets/MSLesSeg-Dataset/train")
DEFAULT_RESULTS_UNET = Path("results/unet")
DEFAULT_RESULTS_YOLO = Path("results/yolo")

MASK_COLORS = {
    "gt": (255, 0, 0),         # rojo
    "unet": (0, 255, 0),       # verde (UNet A)
    "unet_b": (255, 0, 255),   # magenta (UNet B)
    "yolo": (0, 128, 255),     # azul (YOLO A)
    "yolo_b": (255, 165, 0),   # naranja (YOLO B)
}
ALPHA = {"gt": 110, "unet": 110, "unet_b": 110, "yolo": 110, "yolo_b": 110}
EXPERIMENTS_META = {
    "A": {"net": "unet", "sel": "Base",    "sr": "No",  "title": "A · U-Net · Base · No"},
    "B": {"net": "yolo", "sel": "Base",    "sr": "No",  "title": "B · YOLO · Base · No"},
    "C": {"net": "unet", "sel": "Base",    "sr": "x2",  "title": "C · U-Net · Base · x2"},
    "D": {"net": "yolo", "sel": "Base",    "sr": "x2",  "title": "D · YOLO · Base · x2"},
    "E": {"net": "unet", "sel": "Lesión",  "sr": "No",  "title": "E · U-Net · Lesión · No"},
    "F": {"net": "unet", "sel": "Lesión",  "sr": "x2",  "title": "F · U-Net · Lesión · x2"},
    "G": {"net": "yolo", "sel": "Lesión",  "sr": "No",  "title": "G · YOLO · Lesión · No"},
    "H": {"net": "yolo", "sel": "Lesión",  "sr": "x2",  "title": "H · YOLO · Lesión · x2"},
    "I": {"net": "unet", "sel": "Cerebro", "sr": "No",  "title": "I · U-Net · Cerebro · No"},
    "J": {"net": "unet", "sel": "Cerebro", "sr": "x2",  "title": "J · U-Net · Cerebro · x2"},
    "K": {"net": "yolo", "sel": "Cerebro", "sr": "No",  "title": "K · YOLO · Cerebro · No"},
    "L": {"net": "yolo", "sel": "Cerebro", "sr": "x2",  "title": "L · YOLO · Cerebro · x2"},
}


def list_px_ty(dataset_dir: Path) -> List[Tuple[str, str]]:
    """Escanea train/ para pares (PX, TY) que tengan FLAIR y MASK."""
    pairs = []
    if not dataset_dir.exists():
        return pairs
    for px_dir in sorted([d for d in dataset_dir.iterdir() if d.is_dir()]):
        for ty_dir in sorted([d for d in px_dir.iterdir() if d.is_dir()]):
            flair = get_nii_path(ty_dir, f"{px_dir.name}_{ty_dir.name}_FLAIR")
            mask  = get_nii_path(ty_dir, f"{px_dir.name}_{ty_dir.name}_MASK")
            if flair and mask:
                pairs.append((px_dir.name, ty_dir.name))
    return pairs

def normalize_to_uint8(slice2d: np.ndarray) -> np.ndarray:
    """Min-max normaliza a [0,255] uint8, robusto a outliers."""
    x = slice2d.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        lo, hi = x.min(), x.max()
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo) if (hi - lo) > 0 else np.zeros_like(x)
    return (x * 255).astype(np.uint8)

def get_nii_path(base_dir: Path, filename: str) -> Optional[Path]:
    """
    Devuelve la ruta de un archivo .nii o .nii.gz si existe.
    base_dir: carpeta (ej. PX/TY)
    filename: nombre base sin extensión (ej. "P01_T1_FLAIR")
    """
    nii = base_dir / f"{filename}.nii"
    nii_gz = base_dir / f"{filename}.nii.gz"
    if nii.exists():
        return nii
    elif nii_gz.exists():
        return nii_gz
    else:
        return None

def nii_get_slice(nii_path: Path, z: int) -> np.ndarray:
    img = nib.load(str(nii_path))
    vol = np.asanyarray(img.dataobj)
    if vol.ndim != 3:
        raise ValueError(f"Esperaba un volumen 3D, got shape {vol.shape}")
    if z < 0 or z >= vol.shape[2]:
        raise IndexError(f"Slice Z={z} fuera de rango (0..{vol.shape[2]-1})")
    return vol[..., z]

def to_pil_gray(arr_uint8: np.ndarray) -> Image.Image:
    return Image.fromarray(arr_uint8, mode="L")

def load_png_mask(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        return None
    img = Image.open(path).convert("L")
    arr = np.array(img)
    bin_arr = (arr >= 128).astype(np.uint8) * 255
    return Image.fromarray(bin_arr, mode="L")

def overlay_masks(base: Image.Image, overlays: List[Tuple[Image.Image, Tuple[int,int,int], int]]) -> Image.Image:
    base_rgb = base.convert("RGB")
    w, h = base_rgb.size
    comp = base_rgb.copy()
    for m, color, alpha in overlays:
        m_resized = m.resize((w, h), resample=Image.NEAREST)
        m_arr = np.array(m_resized)
        if m_arr.ndim == 3:
            m_arr = m_arr[..., 0]
        color_arr = np.zeros((h, w, 3), dtype=np.uint8)
        color_arr[m_arr > 0] = np.array(color, dtype=np.uint8)
        color_img = Image.fromarray(color_arr, mode="RGB")
        alpha_arr = np.zeros((h, w), dtype=np.uint8)
        alpha_arr[m_arr > 0] = alpha
        alpha_img = Image.fromarray(alpha_arr, mode="L")
        comp = Image.composite(color_img, comp, alpha_img)
    return comp

def find_unet_experiments(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

def find_unet_folds(unet_root: Path, exp: str) -> List[str]:
    """
    Estructura esperada: results/unet/<exp>/<exp>_fold_X/predictions/
    Devuelve nombres de carpetas '<exp>_fold_X'
    """
    exp_dir = unet_root / exp
    if not exp_dir.exists():
        return []
    pattern = re.compile(rf"^{re.escape(exp)}_fold_(\d+)$")
    folds = []
    for d in exp_dir.iterdir():
        if d.is_dir() and pattern.match(d.name) and (d / "predictions").exists():
            folds.append(d.name)
    return sorted(folds, key=lambda n: int(n.split("_")[-1]))

def find_yolo_experiments(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

def find_yolo_folds(yolo_root: Path, exp: str) -> List[str]:
    exp_dir = yolo_root / exp
    if not exp_dir.exists():
        return []
    pattern = re.compile(r"^fold_(\d+)$")
    folds = []
    for d in exp_dir.iterdir():
        if d.is_dir() and pattern.match(d.name) and (d / "predictions" / "masks").exists():
            folds.append(d.name)
    return sorted(folds, key=lambda n: int(n.split("_")[-1]))

def build_pred_paths(px: str, ty: str, z: int,
                     unet_root: Path, unet_exp: str, unet_fold_dirname: str,
                     yolo_root: Path, yolo_exp: str, yolo_fold_dirname: str) -> Tuple[Path, Path]:
    unet_path = unet_root / unet_exp / unet_fold_dirname / "predictions" / f"{px}_{ty}_{z}.png"
    yolo_path = yolo_root / yolo_exp / yolo_fold_dirname / "predictions" / "masks" / f"{px}_{ty}_{z}.png"
    return unet_path, yolo_path

# --- Fuente con soporte de tildes (ajusta si prefieres otra)
def load_unicode_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/gnu-free/FreeSerifBold.otf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/TTF/Iosevka-Regular.ttc", size, index=0)
        except Exception:
            return ImageFont.load_default()

def make_row_1x4(img_original: Image.Image,
                 img_gt: Optional[Image.Image],
                 img_pred: Optional[Image.Image],
                 title_pred: str,
                 pad: int = 12,
                 title_h: int = 28,
                 color_overlay: Tuple[int, int, int] = (0, 255, 0),
                 alpha_overlay: int = 110) -> Image.Image:
    """Devuelve una imagen 1x3 (Original, GT, Pred) con títulos."""
    base_w, base_h = img_original.size
    def ensure(img):
        if img is None:
            return Image.new("L", (base_w, base_h), 0)
        return img.resize((base_w, base_h), resample=Image.NEAREST)

    A = ensure(img_original.convert("L")).convert("RGB")
    B = ensure(img_gt.convert("L") if img_gt else None).convert("RGB")
    C = ensure(img_pred.convert("L") if img_pred else None).convert("RGB")

    titles = ("Imagen original", "Máscara real", title_pred)
    cells = [A, B, C]

    cell_w, cell_h = base_w, base_h + title_h
    grid_w = cell_w * 3 + pad * 4
    grid_h = cell_h + pad * 2

    canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = load_unicode_font(16)

    def paste_with_title(img, x, y, title):
        tx, ty = x + 6, y + 6
        draw.rectangle([x, y, x + cell_w, y + title_h], fill=(240, 240, 240))
        draw.text((tx, ty), title, fill=(0, 0, 0), font=font)
        canvas.paste(img, (x, y + title_h))

    x = pad
    for img, title in zip(cells, titles):
        paste_with_title(img, x, pad, title)
        x += cell_w + pad

    return canvas

def stack_rows_vertically(rows: List[Image.Image], pad: int = 12) -> Image.Image:
    """Apila imágenes 1x4 verticalmente con padding."""
    if not rows:
        return None
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows) + pad * (len(rows) + 1)
    canvas = Image.new("RGB", (w + pad * 2, h), (20, 20, 20))
    y = pad
    for r in rows:
        # centrar cada fila
        x = (canvas.width - r.width) // 2
        canvas.paste(r, (x, y))
        y += r.height + pad
    return canvas

# ---- util: slices con lesión por GT
def compute_lesion_slices(gt_path: Path) -> List[int]:
    """Devuelve lista de Z donde la GT tiene pixeles >0."""
    try:
        gt_img = nib.load(str(gt_path))
        vol = np.asanyarray(gt_img.dataobj)
        if vol.ndim != 3:
            return []
        # Un slice tiene lesión si cualquiera de sus píxeles > 0
        lesion_bool = (vol > 0).reshape(-1, vol.shape[2]).any(axis=0)
        return [int(z) for z, has in enumerate(lesion_bool) if has]
    except Exception:
        return []

def normalize_slice(
    slice2d: np.ndarray,
    mode: str = "minmax_slice",   # "minmax_slice" | "percentile_slice" | "zscore_slice" | "minmax_volume" | "percentile_volume"
    p_low: float = 1.0,
    p_high: float = 99.0,
    vol_stats: tuple | None = None,   # (lo, hi) precomputados para el volumen
) -> np.ndarray:
    x = slice2d.astype(np.float32)

    if mode == "minmax_slice":
        lo, hi = float(x.min()), float(x.max())

    elif mode == "percentile_slice":
        lo, hi = np.percentile(x, [p_low, p_high]).astype(np.float32)

    elif mode == "zscore_slice":
        mu, sigma = float(x.mean()), float(x.std() + 1e-8)
        x = (x - mu) / sigma
        lo, hi = -2.5, 2.5  # ventana típica
    elif mode in ("minmax_volume", "percentile_volume"):
        assert vol_stats is not None, "Faltan estadísticas del volumen"
        lo, hi = vol_stats
    else:
        lo, hi = float(x.min()), float(x.max())

    if hi <= lo:
        x = np.zeros_like(x, dtype=np.float32)
    else:
        x = np.clip(x, lo, hi)
        x = (x - lo) / (hi - lo)

    return (x * 255).astype(np.uint8)

def apply_clahe(u8: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(u8)

def load_unicode_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/gnu-free/FreeSerifBold.otf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/TTF/Iosevka-Regular.ttc", size, index=0)
        except Exception:
            return ImageFont.load_default()

def draw_cell_with_title(canvas, x, y, cell_w, cell_h, title_h, img: Image.Image, title: str, font):
    draw = ImageDraw.Draw(canvas)
    # barra de título
    draw.rectangle([x, y, x + cell_w, y + title_h], fill=(240, 240, 240))
    draw.text((x + 6, y + 6), title, fill=(0, 0, 0), font=font)
    # imagen
    canvas.paste(img.convert("RGB"), (x, y + title_h))

def make_experiments_matrix(
    flair_img: Image.Image,
    gt_img: Optional[Image.Image],
    preds: List[Tuple[str, Image.Image]],   # lista de (titulo, pred_img)
    cols_pred: int = 4,
    pad: int = 12,
    title_h: int = 28,
) -> Optional[Image.Image]:
    """
    Construye una imagen tipo:
    [ Original | GT | Pred_1 | Pred_2 | ... (cols_pred por fila) ]
    con Original y GT 'rowspan' (pintados una vez, a la izquierda, ocupando toda la altura).
    """
    if flair_img is None:
        return None
    base_w, base_h = flair_img.size
    font = load_unicode_font(16)

    # Normaliza tamaños
    def ensure(img):
        if img is None:
            return Image.new("L", (base_w, base_h), 0)
        return img.resize((base_w, base_h), resample=Image.NEAREST)

    flair_rgb = ensure(flair_img).convert("RGB")
    gt_rgb    = ensure(gt_img if gt_img else None).convert("RGB")

    # Preparar celdas de predicción (convertidas a RGB, tamaño base)
    preds = [(t, ensure(p).convert("RGB")) for (t, p) in preds]
    if len(preds) == 0:
        # solo dibuja original + gt
        rows = 1
    else:
        rows = int(np.ceil(len(preds) / float(cols_pred)))

    # Geometría
    cell_w, cell_h = base_w, base_h + title_h
    left_cols = 2  # Original y GT
    total_cols = left_cols + max(cols_pred, 1)

    grid_w = pad * (total_cols + 1) + cell_w * total_cols
    grid_h = pad * (rows + 1) + cell_h * rows

    canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))

    # Pintar Original y GT como 'rowspan': alto = grid_h - 2*pad
    left_x1 = pad
    left_x2 = pad * 2 + cell_w
    top_y = pad

    # Celda "Original" (altura total)
    draw_cell_with_title(canvas, left_x1, top_y, cell_w, cell_h * rows + pad * (rows - 1), title_h, flair_rgb, "Imagen original", font)
    # Celda "GT" (altura total)
    draw_cell_with_title(canvas, left_x2, top_y, cell_w, cell_h * rows + pad * (rows - 1), title_h, gt_rgb, "Máscara real", font)

    # Pintar predicciones en cuadrícula a la derecha
    start_x = pad * 3 + cell_w * 2
    for idx, (title, pred_img) in enumerate(preds):
        r = idx // cols_pred
        c = idx % cols_pred
        x = start_x + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        draw_cell_with_title(canvas, x, y, cell_w, cell_h, title_h, pred_img, title, font)

    return canvas


# =========================
# Sidebar (paths & options)
# =========================
with st.sidebar:
    st.header("Rutas")
    dataset_dir = Path(st.text_input("Carpeta dataset (PX/TY/...)", value=str(DEFAULT_DATASET_DIR)))
    unet_root = Path(st.text_input("Carpeta resultados UNet", value=str(DEFAULT_RESULTS_UNET)))
    yolo_root = Path(st.text_input("Carpeta resultados YOLO", value=str(DEFAULT_RESULTS_YOLO)))
    st.markdown("---")
    only_pairs_with_lesion = st.checkbox("Mostrar solo PX/TY con al menos un corte con lesión (GT)", value=False)
    st.caption("El filtro de lesión usa la máscara **GT**.")
    st.markdown("---")
    st.caption("Si cambias las rutas, pulsa el botón para refrescar las listas.")
    refresh = st.button("🔄 Refrescar listas")

# =========================
# Scan filesystem (cached)
# =========================
@st.cache_data(show_spinner=False)
def scan_all(dataset_dir: Path, unet_root: Path, yolo_root: Path):
    pairs = list_px_ty(dataset_dir)
    unet_exps = find_unet_experiments(unet_root)
    yolo_exps = find_yolo_experiments(yolo_root)
    folds_map_unet = {exp: find_unet_folds(unet_root, exp) for exp in unet_exps}
    folds_map_yolo = {exp: find_yolo_folds(yolo_root, exp) for exp in yolo_exps}

    slices_by_pair: Dict[Tuple[str,str], int] = {}
    lesion_slices_by_pair: Dict[Tuple[str,str], List[int]] = {}
    for px, ty in pairs:
        flair_path = get_nii_path(dataset_dir / px / ty, f"{px}_{ty}_FLAIR")
        gt_path    = get_nii_path(dataset_dir / px / ty, f"{px}_{ty}_MASK")
        try:
            img = nib.load(str(flair_path))
            vol = np.asanyarray(img.dataobj)
            n_slices = vol.shape[2] if vol.ndim == 3 else 0
        except Exception:
            n_slices = 0
        slices_by_pair[(px, ty)] = n_slices
        lesion_slices_by_pair[(px, ty)] = compute_lesion_slices(gt_path)
    return pairs, unet_exps, yolo_exps, folds_map_unet, folds_map_yolo, slices_by_pair, lesion_slices_by_pair

if refresh:
    st.cache_data.clear()

(pairs, unet_exps, yolo_exps,
 folds_map_unet, folds_map_yolo,
 slices_by_pair, lesion_slices_by_pair) = scan_all(dataset_dir, unet_root, yolo_root)

# =========================
# UI selectors
# =========================
st.title("Visualizador de Segmentación EM")

colA, colB, colC = st.columns([1, 1, 1.2])

st.subheader("Modo de comparación")
comparison_mode = st.radio(
    "Elige cómo comparar",
    options=["UNet vs YOLO", "UNet A vs UNet B", "YOLO A vs YOLO B"],
    index=0,
    horizontal=True
)


st.subheader("Contraste / Windowing")
colw1, colw2, colw3 = st.columns([1.2,1,1])
mode = colw1.selectbox(
    "Modo de normalización",
    ["minmax_slice", "percentile_slice", "zscore_slice", "minmax_volume", "percentile_volume"],
    index=0,  # por defecto como antes: min–max por slice
    format_func=lambda s: {
        "minmax_slice":"Min–Max (slice)",
        "percentile_slice":"Percentiles (slice)",
        "zscore_slice":"Z-Score (slice)",
        "minmax_volume":"Min–Max (volumen)",
        "percentile_volume":"Percentiles (volumen)",
    }[s]
)
p_low = colw2.number_input("pₗ (%)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
p_high = colw3.number_input("pₕ (%)", min_value=80.0, max_value=100.0, value=99.0, step=0.5)

clahe_on = st.checkbox("Aplicar CLAHE", value=False)
clip_limit = st.slider("CLAHE clip limit", 1.0, 5.0, 2.0, 0.1, disabled=not clahe_on)
tile_size  = st.slider("CLAHE tiles", 4, 16, 8, 1, disabled=not clahe_on)

with colA:
    st.subheader("Paciente / Timepoint")
    # Filtrar PX/TY si se pide
    display_pairs = pairs
    if only_pairs_with_lesion:
        display_pairs = [(px, ty) for (px, ty) in pairs if len(lesion_slices_by_pair.get((px, ty), [])) > 0]

    if not display_pairs:
        st.warning("No hay PX/TY disponibles con los criterios seleccionados.")
        # fallback mínimo para evitar errores de índice
        px, ty, z = "P01", "T1", 0
    else:
        options_pair = [f"{px} / {ty}" for (px, ty) in display_pairs]
        # options_pair = sorted([f"{px} / {ty}" for (px, ty) in display_pairs], key=lambda x: (int(x.split(" / ")[0][1:]), x.split(" / ")[1]))
        choice_pair = st.selectbox("PX / TY", options_pair, index=0)
        px, ty = display_pairs[options_pair.index(choice_pair)]
        n_slices = slices_by_pair.get((px, ty), 0)

        # Filtro de Z con lesión
        only_z_with_lesion = st.checkbox("Mostrar solo slices (Z) con lesión (GT) para este PX/TY", value=False)
        lesion_zs = lesion_slices_by_pair.get((px, ty), [])
        if only_z_with_lesion and lesion_zs:
            z = st.selectbox("Slice Z (con lesión)", lesion_zs, index=0)
        else:
            z = st.number_input("Slice Z", min_value=0, max_value=max(0, n_slices-1),
                                value=min(0, max(0, n_slices//2)))

with colB:
    st.subheader("UNet")
    unet_exp = st.selectbox("Experimento UNet A", unet_exps if unet_exps else ["—"], index=0 if unet_exps else 0)
    unet_folds = (folds_map_unet.get(unet_exp, []) if unet_exps else [])
    unet_fold = st.selectbox("Fold UNet A", unet_folds if unet_folds else ["—"], index=0)

    unet_exp_b = None
    unet_fold_b = None
    if comparison_mode == "UNet A vs UNet B":
        unet_exp_b = st.selectbox("Experimento UNet B", unet_exps if unet_exps else ["—"],
                                  index=min(1, len(unet_exps)-1) if len(unet_exps) > 1 else 0)
        unet_folds_b = (folds_map_unet.get(unet_exp_b, []) if unet_exps else [])
        unet_fold_b = st.selectbox("Fold UNet B", unet_folds_b if unet_folds_b else ["—"], index=0)

with colC:
    st.subheader("YOLO")
    yolo_exp = st.selectbox("Experimento YOLO A", yolo_exps if yolo_exps else ["—"], index=0 if yolo_exps else 0)
    yolo_folds = (folds_map_yolo.get(yolo_exp, []) if yolo_exps else [])
    yolo_fold = st.selectbox("Fold YOLO A", yolo_folds if yolo_folds else ["—"], index=0)

    yolo_exp_b = None
    yolo_fold_b = None
    if comparison_mode == "YOLO A vs YOLO B":
        yolo_exp_b = st.selectbox("Experimento YOLO B", yolo_exps if yolo_exps else ["—"],
                                  index=min(1, len(yolo_exps)-1) if len(yolo_exps) > 1 else 0)
        yolo_folds_b = (folds_map_yolo.get(yolo_exp_b, []) if yolo_exps else [])
        yolo_fold_b = st.selectbox("Fold YOLO B", yolo_folds_b if yolo_folds_b else ["—"], index=0)


st.markdown("---")

# =========================
# Load data for current selection
# =========================
# Rutas base
flair_path = get_nii_path(dataset_dir / px / ty, f"{px}_{ty}_FLAIR")
gt_path    = get_nii_path(dataset_dir / px / ty, f"{px}_{ty}_MASK")

# Siempre A
unet_pred_path, yolo_pred_path = build_pred_paths(px, ty, int(z),
                                                  unet_root, unet_exp, unet_fold,
                                                  yolo_root, yolo_exp, yolo_fold)

# Opcionales B
unet_pred_path_b = None
yolo_pred_path_b = None

if comparison_mode == "UNet A vs UNet B" and unet_exp_b and unet_fold_b:
    unet_pred_path_b, _ = build_pred_paths(px, ty, int(z),
                                           unet_root, unet_exp_b, unet_fold_b,
                                           yolo_root, yolo_exp, yolo_fold)

if comparison_mode == "YOLO A vs YOLO B" and yolo_exp_b and yolo_fold_b:
    _, yolo_pred_path_b = build_pred_paths(px, ty, int(z),
                                           unet_root, unet_exp, unet_fold,
                                           yolo_root, yolo_exp_b, yolo_fold_b)

# Carga imágenes y máscaras
errors = []
flair_img_pil = None
gt_mask_pil = None

vol_stats = None
if mode in ("minmax_volume", "percentile_volume"):
    try:
        vol = np.asanyarray(nib.load(str(flair_path)).dataobj).astype(np.float32)
        if mode == "minmax_volume":
            lo, hi = float(vol.min()), float(vol.max())
        else:
            lo, hi = np.percentile(vol, [p_low, p_high]).astype(np.float32)
        vol_stats = (lo, hi)
    except Exception as e:
        st.warning(f"No se pudieron calcular stats de volumen: {e}")

try:
    flair_slice = nii_get_slice(flair_path, int(z))
    flair_u8 = normalize_slice(flair_slice, mode=mode, p_low=p_low, p_high=p_high, vol_stats=vol_stats)
    if clahe_on:
        flair_u8 = apply_clahe(flair_u8, clip_limit=clip_limit, tile_grid_size=tile_size)
    flair_img_pil = to_pil_gray(flair_u8)
except Exception as e:
    errors.append(f"FLAIR: {e}")

try:
    gt_slice = nii_get_slice(gt_path, int(z))
    gt_mask_pil = to_pil_gray((gt_slice > 0).astype(np.uint8) * 255)
except Exception as e:
    errors.append(f"GT MASK: {e}")

unet_mask_pil = load_png_mask(unet_pred_path)
yolo_mask_pil = load_png_mask(yolo_pred_path)

unet_mask_pil_b = load_png_mask(unet_pred_path_b) if unet_pred_path_b else None
yolo_mask_pil_b = load_png_mask(yolo_pred_path_b) if yolo_pred_path_b else None

# =========================
# Mosaic controls
# =========================
st.markdown("---")
st.subheader("Comparador de experimentos (mosaico)")

col_add1, col_add2, col_add3, col_add4, col_tools = st.columns([1,1,1,1,2])

def _add_row_to_mosaic(net_label: str, exp_name: str, fold_name: str, pred_img: Optional[Image.Image], color_key: str):
    if flair_img_pil is None:
        st.warning("No hay FLAIR para generar la fila del mosaico.")
        return
    title_pred = f"Predicción {net_label} ({exp_name} / {fold_name})"
    color = MASK_COLORS.get(color_key, (0,255,0))
    alpha = ALPHA.get(color_key, 110)
    row_img = make_row_1x4(flair_img_pil, gt_mask_pil, pred_img, title_pred,
                           color_overlay=color, alpha_overlay=alpha)
    st.session_state.mosaic_rows.append({
        "net": net_label,
        "exp": exp_name,
        "fold": fold_name,
        "img": row_img
    })

with col_add1:
    if st.button("➕ Añadir UNet A"):
        _add_row_to_mosaic("UNet", unet_exp, unet_fold, unet_mask_pil, "unet")

with col_add2:
    if comparison_mode == "UNet A vs UNet B" and unet_mask_pil_b is not None:
        if st.button("➕ Añadir UNet B"):
            _add_row_to_mosaic("UNet B", unet_exp_b, unet_fold_b, unet_mask_pil_b, "unet_b")

with col_add3:
    if st.button("➕ Añadir YOLO A"):
        _add_row_to_mosaic("YOLO", yolo_exp, yolo_fold, yolo_mask_pil, "yolo")

with col_add4:
    if comparison_mode == "YOLO A vs YOLO B" and yolo_mask_pil_b is not None:
        if st.button("➕ Añadir YOLO B"):
            _add_row_to_mosaic("YOLO B", yolo_exp_b, yolo_fold_b, yolo_mask_pil_b, "yolo_b")

with col_tools:
    clear = st.button("🗑️ Vaciar mosaico")
    if clear:
        st.session_state.mosaic_rows = []

# Construir mosaico actual
rows_imgs = [r["img"] for r in st.session_state.mosaic_rows]
mosaic_img = stack_rows_vertically(rows_imgs, pad=12) if rows_imgs else None

prev_col, dl_col = st.columns([2,1])
with prev_col:
    if mosaic_img is not None:
        st.image(mosaic_img, caption=f"Filas: {len(rows_imgs)}", use_container_width=True)
    else:
        st.info("Añade experimentos para construir el mosaico (cada fila es un experimento).")

with dl_col:
    if mosaic_img is not None:
        bufm = io.BytesIO()
        mosaic_img.save(bufm, format="PNG")
        bufm.seek(0)
        mosaic_name = f"{px}_{ty}_Z{z}_mosaico_{len(rows_imgs)}filas.png"
        st.download_button("⬇️ Descargar mosaico (PNG)", data=bufm, file_name=mosaic_name, mime="image/png")
    else:
        st.button("⬇️ Descargar mosaico (PNG)", disabled=True)


# =========================
# Matrix controls
# =========================
st.markdown("---")
st.subheader("Matriz de experimentos (todos en una imagen)")

# Inicializa estado para guardar el mapeo ID -> (exp, fold)
st.session_state.setdefault("matrix_map", {})  # p.ej., {"A": {"exp": "...", "fold": "..."}}

# Qué IDs incluir (por defecto, todos)
all_ids = list(EXPERIMENTS_META.keys())
chosen_ids = st.multiselect("Elige experimentos (ID)", all_ids, default=all_ids)

# Columnas por fila para las predicciones (a la derecha)
cols_pred = st.slider("Columnas de predicción por fila", 2, 6, 4, 1)

st.caption("Asigna la carpeta de experimento y fold para cada ID (UNet/YOLO) usando los desplegables detectados en /results.")

# Construye filas de controles
for _id in chosen_ids:
    meta = EXPERIMENTS_META[_id]
    net = meta["net"]  # "unet" | "yolo"
    c1, c2, c3 = st.columns([0.5, 1.5, 1.2])
    with c1:
        st.markdown(f"**{_id}** · {meta['title'].split('·')[1].strip()}")  # muestra "U-Net" o "YOLO"
    if net == "unet":
        with c2:
            exp = st.selectbox(f"Experimento UNet ({_id})", unet_exps if unet_exps else ["—"],
                               index=0 if unet_exps else 0, key=f"mx_unet_exp_{_id}")
        with c3:
            folds = folds_map_unet.get(exp, []) if unet_exps else []
            fold = st.selectbox(f"Fold UNet ({_id})", folds if folds else ["—"], index=0, key=f"mx_unet_fold_{_id}")
    else:
        with c2:
            exp = st.selectbox(f"Experimento YOLO ({_id})", yolo_exps if yolo_exps else ["—"],
                               index=0 if yolo_exps else 0, key=f"mx_yolo_exp_{_id}")
        with c3:
            folds = folds_map_yolo.get(exp, []) if yolo_exps else []
            fold = st.selectbox(f"Fold YOLO ({_id})", folds if folds else ["—"], index=0, key=f"mx_yolo_fold_{_id}")

    st.session_state["matrix_map"][_id] = {"exp": exp, "fold": fold}

# Construir lista de predicciones (título, imagen) para los IDs elegidos
pred_cells: List[Tuple[str, Image.Image]] = []

for _id in chosen_ids:
    meta = EXPERIMENTS_META[_id]
    map_info = st.session_state["matrix_map"].get(_id)
    if not map_info:
        continue
    exp = map_info["exp"]
    fold = map_info["fold"]

    # Construir ruta de predicción según red
    if meta["net"] == "unet":
        pred_path = unet_root / exp / fold / "predictions" / f"{px}_{ty}_{int(z)}.png"
    else:
        pred_path = yolo_root / exp / fold / "predictions" / "masks" / f"{px}_{ty}_{int(z)}.png"

    pred_img = load_png_mask(pred_path)
    # Título bonito
    title = meta["title"]
    # Si quieres overlay GT + pred en la celda de matriz, cámbialo aquí:
    #   pred_img = overlay_masks(flair_img_pil, [(gt_mask_pil, MASK_COLORS["gt"], ALPHA["gt"]),
    #                                            (pred_img, (0,255,0) if meta["net"]=="unet" else (0,128,255), 110)]) if pred_img else None
    if pred_img is None:
        # placeholder negro con aviso en el título
        title += " (no encontrada)"
        pred_img = Image.new("L", flair_img_pil.size, 0)

    pred_cells.append((title, pred_img))

# Construir la matriz final
matrix_img = make_experiments_matrix(flair_img_pil, gt_mask_pil, pred_cells, cols_pred=cols_pred, pad=12, title_h=28)

# Previsualización + descarga
mcol1, mcol2 = st.columns([2,1])
with mcol1:
    if matrix_img is not None:
        st.image(matrix_img, caption=f"Matriz de {len(pred_cells)} experimentos (cols={cols_pred})", use_container_width=True)
    else:
        st.info("No se pudo generar la matriz (falta FLAIR).")

with mcol2:
    if matrix_img is not None:
        bufm = io.BytesIO()
        matrix_img.save(bufm, format="PNG")
        bufm.seek(0)
        fname = f"{px}_{ty}_Z{z}_matriz_{len(pred_cells)}exps.png"
        st.download_button("⬇️ Descargar matriz (PNG)", data=bufm, file_name=fname, mime="image/png")
    else:
        st.button("⬇️ Descargar matriz (PNG)", disabled=True)

# =========================
# Display
# =========================
info_cols = st.columns(4)
with info_cols[0]:
    st.caption(f"PX: **{px}**  •  TY: **{ty}**  •  Z: **{z}**")
with info_cols[1]:
    st.caption(f"Modo: **{comparison_mode}**")
with info_cols[2]:
    if comparison_mode.startswith("UNet"):
        st.caption(f"UNet A: **{unet_exp} / {unet_fold}**" + (f"  •  UNet B: **{unet_exp_b} / {unet_fold_b}**" if unet_pred_path_b else ""))
    else:
        st.caption(f"UNet A: **{unet_exp} / {unet_fold}**")
with info_cols[3]:
    if comparison_mode.startswith("YOLO"):
        st.caption(f"YOLO A: **{yolo_exp} / {yolo_fold}**" + (f"  •  YOLO B: **{yolo_exp_b} / {yolo_fold_b}**" if yolo_pred_path_b else ""))
    else:
        st.caption(f"YOLO A: **{yolo_exp} / {yolo_fold}**")

if errors:
    for e in errors:
        st.error(e)

# Panels principales
c1, c2, c3, c4, c5 = st.columns(5)

def show_panel(title: str, img: Image.Image | None, fallback_msg: str):
    st.markdown(f"**{title}**")
    if img is not None:
        st.image(img, use_container_width=True)
    else:
        st.warning(fallback_msg)


with c1:
    show_panel("FLAIR", flair_img_pil, "FLAIR no disponible.")
with c2:
    show_panel("GT (máscara)", gt_mask_pil, "GT no disponible.")

# c1 y c2 (FLAIR y GT) se mantienen igual

with c3:
    if comparison_mode == "UNet A vs UNet B":
        st.markdown("**UNet A (pred)**")
        if unet_mask_pil is not None:
            st.image(unet_mask_pil, use_container_width=True)
            st.caption(str(unet_pred_path))
        else:
            st.warning("Predicción UNet A no encontrada.")
            st.caption(str(unet_pred_path))

        st.markdown("**UNet B (pred)**")
        if unet_mask_pil_b is not None:
            st.image(unet_mask_pil_b, use_container_width=True)
            st.caption(str(unet_pred_path_b))
        else:
            st.info("Predicción UNet B no encontrada." if unet_pred_path_b else "UNet B no seleccionado.")
            if unet_pred_path_b: st.caption(str(unet_pred_path_b))

    else:
        st.markdown("**UNet (pred)**")
        if unet_mask_pil is not None:
            st.image(unet_mask_pil, use_container_width=True)
            st.caption(str(unet_pred_path))
        else:
            st.warning("Predicción UNet no encontrada.")
            st.caption(str(unet_pred_path))

with c4:
    if comparison_mode == "YOLO A vs YOLO B":
        st.markdown("**YOLO A (pred)**")
        if yolo_mask_pil is not None:
            st.image(yolo_mask_pil, use_container_width=True)
            st.caption(str(yolo_pred_path))
        else:
            st.warning("Predicción YOLO A no encontrada.")
            st.caption(str(yolo_pred_path))

        st.markdown("**YOLO B (pred)**")
        if yolo_mask_pil_b is not None:
            st.image(yolo_mask_pil_b, use_container_width=True)
            st.caption(str(yolo_pred_path_b))
        else:
            st.info("Predicción YOLO B no encontrada." if yolo_pred_path_b else "YOLO B no seleccionado.")
            if yolo_pred_path_b: st.caption(str(yolo_pred_path_b))
    else:
        st.markdown("**YOLO (pred)**")
        if yolo_mask_pil is not None:
            st.image(yolo_mask_pil, use_container_width=True)
            st.caption(str(yolo_pred_path))
        else:
            st.warning("Predicción YOLO no encontrada.")
            st.caption(str(yolo_pred_path))

# with c5:
#     st.markdown("**Composite (overlay)**")
#     if flair_img_pil is not None:
#         overlays = []
#         if gt_mask_pil is not None: overlays.append((gt_mask_pil, MASK_COLORS["gt"], ALPHA["gt"]))

#         if comparison_mode == "UNet A vs UNet B":
#             if unet_mask_pil is not None:   overlays.append((unet_mask_pil,   MASK_COLORS["unet"],   ALPHA["unet"]))
#             if unet_mask_pil_b is not None: overlays.append((unet_mask_pil_b, MASK_COLORS["unet_b"], ALPHA["unet_b"]))
#         elif comparison_mode == "YOLO A vs YOLO B":
#             if yolo_mask_pil is not None:   overlays.append((yolo_mask_pil,   MASK_COLORS["yolo"],   ALPHA["yolo"]))
#             if yolo_mask_pil_b is not None: overlays.append((yolo_mask_pil_b, MASK_COLORS["yolo_b"], ALPHA["yolo_b"]))
#         else:  # UNet vs YOLO
#             if unet_mask_pil is not None: overlays.append((unet_mask_pil, MASK_COLORS["unet"], ALPHA["unet"]))
#             if yolo_mask_pil is not None: overlays.append((yolo_mask_pil, MASK_COLORS["yolo"], ALPHA["yolo"]))

#         composite = overlay_masks(flair_img_pil, overlays) if overlays else flair_img_pil
#         st.image(composite, use_container_width=True)
#     else:
#         composite = None
#         st.info("Sin FLAIR no se puede generar composite.")


st.markdown("---")
st.subheader("Descargas")

# ---- Descarga composite
# dl_col1, dl_col2 = st.columns([1, 2])
# with dl_col1:
#     scale = st.slider("Escala composite", min_value=1, max_value=3, value=1, help="Multiplica el tamaño de salida.")

# with dl_col2:
#     if composite is not None:
#         out_img = composite
#         if scale != 1:
#             w, h = composite.size
#             out_img = composite.resize((w*scale, h*scale), resample=Image.NEAREST)
#         buf = io.BytesIO()
#         out_img.save(buf, format="PNG")
#         buf.seek(0)
#         fname = f"{px}_{ty}_Z{z}_{unet_exp}_{unet_fold}_{yolo_exp}_{yolo_fold}_composite.png"
#         st.download_button("⬇️ Descargar composite (PNG)", data=buf, file_name=fname, mime="image/png")
#     else:
#         st.button("⬇️ Descargar composite (PNG)", disabled=True)

# ---- Grid 1x4 con títulos y descarga
st.markdown("### Imagen 1×4 con títulos (descargable)")
def make_titled_grid(img_original: Optional[Image.Image],
                     img_gt: Optional[Image.Image],
                     img_unet: Optional[Image.Image],
                     img_yolo: Optional[Image.Image],
                     titles=("Imagen original", "Máscara real", "Predicción Unet", "Predicción Yolo"),
                     pad=12, title_h=28) -> Optional[Image.Image]:
    """Crea una cuadrícula 1x4 con títulos sobre cada panel."""
    if img_original is None:
        return None
    # Asegurar todos del mismo tamaño que la original
    base_w, base_h = img_original.size
    def ensure(img):
        if img is None:
            # placeholder vacío
            return Image.new("L", (base_w, base_h), 0)
        return img.resize((base_w, base_h), resample=Image.NEAREST)

    A = ensure(img_original.convert("L"))
    B = ensure(img_gt.convert("L") if img_gt else None)
    C = ensure(img_unet.convert("L") if img_unet else None)
    D = ensure(img_yolo.convert("L") if img_yolo else None)

    # Convertir a RGB para título
    A, B, C, D = [x.convert("RGB") for x in (A, B, C, D)]

    cell_w, cell_h = base_w, base_h + title_h
    grid_w = cell_w * 4 + pad * 5
    grid_h = cell_h + pad * 2

    canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        # font = ImageFont.load_default()
        font = ImageFont.truetype("/usr/share/fonts/TTF/Iosevka-Regular.ttc", 16)
    except Exception:
        font = None

    def paste_with_title(img, x, y, title):
        # título
        tx, ty = x + 6, y + 6
        draw.rectangle([x, y, x + cell_w, y + title_h], fill=(240, 240, 240))
        draw.text((tx, ty), title, fill=(0, 0, 0), font=font)
        # imagen
        canvas.paste(img, (x, y + title_h))

    x1, y1 = pad, pad
    x2, y2 = pad * 2 + cell_w, pad
    x3, y3 = pad * 3 + cell_w * 2, pad
    x4, y4 = pad * 4 + cell_w * 3, pad

    paste_with_title(A, x1, y1, titles[0])
    paste_with_title(B, x2, y2, titles[1])
    paste_with_title(C, x3, y3, titles[2])
    paste_with_title(D, x4, y4, titles[3])

    return canvas

# Títulos según modo
if comparison_mode == "UNet A vs UNet B":
    titles = ("Imagen original", "Máscara real", "Predicción UNet A", "Predicción UNet B")
    pred_left, pred_right = unet_mask_pil, unet_mask_pil_b
elif comparison_mode == "YOLO A vs YOLO B":
    titles = ("Imagen original", "Máscara real", "Predicción YOLO A", "Predicción YOLO B")
    pred_left, pred_right = yolo_mask_pil, yolo_mask_pil_b
else:
    titles = ("Imagen original", "Máscara real", "Predicción UNet", "Predicción YOLO")
    pred_left, pred_right = unet_mask_pil, yolo_mask_pil

grid_img = make_titled_grid(
    img_original=flair_img_pil,
    img_gt=gt_mask_pil,
    img_unet=pred_left,
    img_yolo=pred_right,
    titles=titles
)

gcol1, gcol2 = st.columns([2, 1])
with gcol1:
    if grid_img is not None:
        st.image(grid_img, use_container_width=True)
    else:
        st.info("No se puede generar la imagen 2×2 sin la imagen FLAIR.")

with gcol2:
    if grid_img is not None:
        buf2 = io.BytesIO()
        grid_img.save(buf2, format="PNG")
        buf2.seek(0)
        tag = ("unetA_unetB" if comparison_mode == "UNet A vs UNet B"
               else "yoloA_yoloB" if comparison_mode == "YOLO A vs YOLO B"
               else "unet_yolo")
        gname = f"{px}_{ty}_Z{z}_{tag}.png"
        st.download_button("⬇️ Descargar 2×2 con títulos (PNG)", data=buf2, file_name=gname, mime="image/png")
    else:
        st.button("⬇️ Descargar 2×2 con títulos (PNG)", disabled=True)

# =========================
# Footer
# =========================
with st.expander("Detalles de rutas utilizadas"):
    st.code(f"""
FLAIR: {flair_path}
GT:    {gt_path}

UNet:  {unet_pred_path}
YOLO:  {yolo_pred_path}
""", language="text")

st.caption("Overlay → GT: rojo, UNet: verde, YOLO: azul. El filtro de lesión se basa en la GT.")
