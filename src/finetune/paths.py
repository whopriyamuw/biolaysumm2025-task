import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
CONFIG_FILE = "llama_{}_instruct{}.yaml"

__all__ = ["MODEL_DIR", "CONFIG_DIR", "CONFIG_FILE"]
