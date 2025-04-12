from dataclasses import dataclass
from enum import Enum


class Dataset(Enum):
    elife = "eLife"
    plost = "PLOS"


@dataclass
class BioLaySummConfig:
    dataset: Dataset = Dataset.elife
    file: str = "finetune/biolaysumm.py"
    train_split: str = "train"
    test_split: str = "validation"
