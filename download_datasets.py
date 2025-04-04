"""Script to download datasets"""

from datasets import load_dataset

# Load the dataset (make sure you're logged in if it's gated)
ds = load_dataset("BioLaySumm/BioLaySumm2025-PLOS")

# Specify the local directory for export
output_dir = "biolaysumm_dataset"

# Export each split to local files (e.g., train, test)
for split in ds:
    # You can also use .to_json or .to_parquet here
    ds[split].to_csv(f"{output_dir}/{split}.csv", index=False)
