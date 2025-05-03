"""Script to download datasets"""

import os.path

from datasets import load_dataset
from utils import DATA_ROOT

# Load the dataset (make sure you're logged in if it's gated)
ds1 = load_dataset("BioLaySumm/BioLaySumm2025-PLOS")
ds2 = load_dataset("BioLaySumm/BioLaySumm2025-eLife")

# Specify the local directory for export
output_dir1 = os.path.join(DATA_ROOT, "raw", "plos")
output_dir2 = os.path.join(DATA_ROOT, "raw", "elife")

os.makedirs(output_dir1, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)

for split in ds1:
    ds1[split].to_csv(f"{output_dir1}/{split}.csv", index=False)

for split in ds2:
    ds2[split].to_csv(f"{output_dir2}/{split}.csv", index=False)
