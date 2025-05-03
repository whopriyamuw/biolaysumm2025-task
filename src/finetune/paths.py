import os

from utils import PROJECT_ROOT

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
CONFIG_FILE = "llama_3.1_8b_instruct_{}.yaml"

__all__ = [MODEL_DIR, MODEL, CONFIG_DIR, CONFIG_FILE]
