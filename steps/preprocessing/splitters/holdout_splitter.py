from __future__ import annotations
from typing import Iterable, Tuple, Set
import random

class HoldoutPatientSplitter:
    """
    Patient-level hold-out splitter.
    Returns (trainval_ids, empty_val_ids, test_ids).
    """
    def __init__(self, test_frac: float = 0.2, seed: int = 42):
        assert 0.0 < test_frac < 1.0
        self.test_frac = test_frac
        self.seed = seed

    def split(self, all_patients: Iterable[str]) -> Tuple[Set[str], Set[str], Set[str]]:
        all_patients = list(all_patients)
        rng = random.Random(self.seed)
        rng.shuffle(all_patients)

        n = len(all_patients)
        n_test = max(1, int(round(self.test_frac * n)))
        test_ids = set(all_patients[:n_test])
        trainval_ids = set(all_patients[n_test:])
        return trainval_ids, set(), test_ids  # (train, val, test) where val is empty
