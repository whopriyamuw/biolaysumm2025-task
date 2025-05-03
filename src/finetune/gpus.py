from collections import defaultdict
from dataclasses import dataclass


@dataclass
class GPU:
    name: str = "default"
    data_split: float = 0.2
    batch_size: int = 1


GPUS = defaultdict(lambda: GPU(), {
    "A40": GPU("A40", 0.25, 1),
    "A100": GPU("A100", 0.5, 4),
})

__all__ = ["GPUS"]
