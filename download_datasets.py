"""Script to download datasets"""

from datasets import load_dataset

# Load the dataset (make sure you're logged in if it's gated)
ds1 = load_dataset("BioLaySumm/BioLaySumm2025-PLOS")
ds2 = load_dataset("BioLaySumm/BioLaySumm2025-eLife")

# Specify the local directory for export
output_dir1 = "biolaysumm_dataset/plos"
output_dir2 = "biolaysumm_dataset/elife"

for split in ds1:
    ds1[split].to_csv(f"{output_dir1}/{split}.csv", index=False)

for split in ds2:
    ds2[split].to_csv(f"{output_dir2}/{split}.csv", index=False)
