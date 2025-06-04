from typing import List, Tuple, Set
from sklearn.model_selection import train_test_split

from interfaces.interfaces import PatientSplitter


class RandomPatientSplitter(PatientSplitter):
    def __init__(self, train_frac: float = 0.8, seed: int = 42):
        self.train_frac = train_frac
        self.seed = seed

    def split(self, all_patients: List[str]) -> Tuple[Set[str], Set[str], Set[str]]:
        # 1) train vs tmp
        train_list, tmp_list = train_test_split(
            all_patients,
            test_size=1 - self.train_frac,
            random_state=self.seed,
        )
        # 2) val/test = 50%/50% of tmp
        val_list, test_list = train_test_split(
            tmp_list,
            test_size=0.5,
            random_state=self.seed,
        )
        return set(train_list), set(val_list), set(test_list)
