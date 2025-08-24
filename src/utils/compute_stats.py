from pathlib import Path
from PIL import Image
import numpy as np
import csv

DATASETS_ROOT = Path("datasets")
VALID_PREFIXES = ("unet", "yolo")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ZERO_TOLERANCE = 0  # si hay pequeñas variaciones, sube a 3–5

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def is_all_black(img_path: Path) -> bool:
    im = Image.open(img_path).convert("L")
    arr = np.array(im, dtype=np.uint8)
    return (arr.max() <= ZERO_TOLERANCE)

def mask_has_lesion(mask_path: Path) -> bool:
    im = Image.open(mask_path).convert("L")
    arr = np.array(im, dtype=np.uint8)
    return np.any(arr > 0)

def count_for_dataset(ds_dir: Path):
    images_train = ds_dir / "images" / "train"
    labels_train = ds_dir / "labels" / "train"

    img_files = [p for p in images_train.glob("*") if is_image_file(p)] if images_train.exists() else []
    msk_files = [p for p in labels_train.glob("*") if is_image_file(p)] if labels_train.exists() else []

    img_by_stem = {p.stem: p for p in img_files}
    msk_by_stem = {p.stem: p for p in msk_files}

    stems = sorted(set(img_by_stem) | set(msk_by_stem))

    imgs_train = len(img_files)
    imgs_negras = 0
    masks_total = len(msk_files)
    masks_lesion = 0
    masks_sin_lesion_no_negras = 0
    masks_sin_lesion_negras = 0
    masks_sin_imagen = 0
    imagenes_sin_mask = 0

    # precompute negro por imagen
    img_is_black = {}
    for stem, ipath in img_by_stem.items():
        try:
            img_is_black[stem] = is_all_black(ipath)
            if img_is_black[stem]:
                imgs_negras += 1
        except Exception:
            img_is_black[stem] = None

    for stem in stems:
        mpath = msk_by_stem.get(stem)
        ipath = img_by_stem.get(stem)

        if mpath is None:
            # imagen sin máscara
            imagenes_sin_mask += 1
            continue

        # máscara presente:
        try:
            has_lesion = mask_has_lesion(mpath)
        except Exception:
            # si la máscara está corrupta, la ignoramos del desglose
            continue

        if has_lesion:
            masks_lesion += 1
        else:
            if ipath is None:
                masks_sin_imagen += 1
            else:
                is_black = img_is_black.get(stem)
                if is_black is True:
                    masks_sin_lesion_negras += 1
                elif is_black is False:
                    masks_sin_lesion_no_negras += 1
                else:
                    # si no pudimos evaluar la imagen, cuenta como sin imagen
                    masks_sin_imagen += 1

    return {
        "experimento": ds_dir.name,
        "imgs_train": imgs_train,
        "imgs_negras": imgs_negras,
        "masks_total": masks_total,
        "masks_lesion": masks_lesion,
        "masks_sin_lesion_no_negras": masks_sin_lesion_no_negras,
        "masks_sin_lesion_negras": masks_sin_lesion_negras,
        "masks_sin_imagen": masks_sin_imagen,
        "imagenes_sin_mask": imagenes_sin_mask,
    }

def main():
    rows = []
    for ds in sorted(DATASETS_ROOT.iterdir()):
        if not ds.is_dir():
            continue
        if not ds.name.startswith(VALID_PREFIXES):
            continue
        if ds.name in ("MSLesSeg-Dataset",):
            continue

        rows.append(count_for_dataset(ds))

    if not rows:
        print("No se encontraron carpetas de experimento válidas.")
        return

    # imprimir tabla
    cols = [
        "experimento","imgs_train","imgs_negras",
        "masks_total","masks_lesion",
        "masks_sin_lesion_no_negras","masks_sin_lesion_negras",
        "masks_sin_imagen","imagenes_sin_mask"
    ]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-"*widths[c] for c in cols)
    print(header); print(sep)
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))

    # CSV
    out_csv = Path("datasets_summary_pairwise.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nGuardado: {out_csv.resolve()}")

if __name__ == "__main__":
    main()

