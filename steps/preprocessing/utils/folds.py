from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import GroupKFold

def group_by_patient(images_dir: Path) -> Dict[str, List[Path]]:
    imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    groups: Dict[str, List[Path]] = {}
    for p in imgs:
        pid = p.stem.split("_")[0]
        groups.setdefault(pid, []).append(p)
    return groups

def split_minival(images_dir: Path, frac: float, seed: int) -> Tuple[List[Path], List[Path]]:
    import random
    groups = group_by_patient(images_dir)
    patients = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(round(frac * len(patients))))
    val_patients = set(patients[:n_val])
    train_patients = set(patients[n_val:])
    train_imgs, val_imgs = [], []
    for pid, lst in groups.items():
        (val_imgs if pid in val_patients else train_imgs).extend(sorted(lst))
    return train_imgs, val_imgs

def build_kfolds(images_dir: Path, n_splits: int) -> List[Tuple[List[Path], List[Path]]]:
    """Return list of (train_imgs, val_imgs) per fold using GroupKFold by patient."""
    groups = group_by_patient(images_dir)
    patients = sorted(groups.keys())
    # Flatten items and groups
    X, grp = [], []
    for pid in patients:
        for p in sorted(groups[pid]):
            X.append(p)
            grp.append(pid)
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    import numpy as np
    idxs = np.arange(len(X))
    for tr_idx, va_idx in gkf.split(idxs, groups=grp):
        tr = [X[i] for i in tr_idx]
        va = [X[i] for i in va_idx]
        folds.append((tr, va))
    return folds
