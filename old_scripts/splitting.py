import os
from typing import NamedTuple

from sklearn.model_selection import train_test_split


class SplitDirs(NamedTuple):
    images_train_dir: str
    images_val_dir: str
    images_test_dir: str
    labels_train_dir: str
    labels_val_dir: str
    labels_test_dir: str

def create_dirs(dst_path: str) -> SplitDirs:
    """
    Create directories for images and labels for train, val, and test sets.
    Args: 
        dst_path: The base directory where the images and labels will be stored.
    Returns: 
        A tuple containing the paths to the created directories:
            images_train_dir
            images_val_dir
            images_test_dir
            labels_train_dir
            labels_val_dir
            labels_test_dir
    """
    images_train_dir = os.path.join(dst_path, "images", "train")
    images_val_dir = os.path.join(dst_path, "images", "val")
    images_test_dir = os.path.join(dst_path, "images", "test")
    labels_test_dir = os.path.join(dst_path, "labels", "test")
    labels_train_dir = os.path.join(dst_path, "labels", "train")
    labels_val_dir = os.path.join(dst_path, "labels", "val")

    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(images_test_dir, exist_ok=True)
    os.makedirs(labels_test_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    return SplitDirs(
        images_train_dir=images_train_dir,
        images_val_dir=images_val_dir,
        images_test_dir=images_test_dir,
        labels_train_dir=labels_train_dir,
        labels_val_dir=labels_val_dir,
        labels_test_dir=labels_test_dir,
    )

def split_patients_triple(
    patients: list[str], train_frac: float = 0.8, random_state: int = 42
) -> tuple[set[str], set[str], set[str]]:
    """
    Splits a list of patient IDs into three sets: train, validation, and test.
    The split is done in two steps:
    1. First, the patients are split into train and temporary sets based on the specified fraction.
    2. Then, the temporary set is split into validation and test sets, each receiving half of the temporary patients.
    Args:
        patients: List of patient IDs (strings).
        train_frac: Fraction of patients to include in the training set (default is 0.8).
        random_state: Random seed for reproducibility (default is 42).
    Returns:
        A tuple containing three sets:
        - train_pat: Set of patient IDs for the training set.
        - val_pat: Set of patient IDs for the validation set.
        - test_pat: Set of patient IDs for the test set.
    """
    # 1) train vs tmp
    train_pat, tmp_pat = train_test_split(
        patients,
        test_size=1 - train_frac,
        random_state=random_state,
    )
    # 2) val vs test (50%–50% de tmp)
    val_pat, test_pat = train_test_split(
        tmp_pat,
        test_size=0.5,
        random_state=random_state,
    )
    return set(train_pat), set(val_pat), set(test_pat)
