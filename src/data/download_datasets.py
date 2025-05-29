"""Script to download the BioLaySumm2025 datasets."""

import os.path
from datasets import load_dataset
from utils import DATA_ROOT

def download_and_save_dataset(dataset_name: str, output_dir: str):
    """Download a dataset and save it as CSV files for each split."""

    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset(dataset_name)
    
    # Save each split as a CSV file
    for split_name, split_data in dataset.items():
        output_path = os.path.join(output_dir, f"{split_name}.csv")
        split_data.to_csv(output_path, index=False)
        print(f"Saved {split_name} split to {output_path}")

def main():
    """Main function to download and process both datasets."""

    datasets = {
        "BioLaySumm/BioLaySumm2025-PLOS": os.path.join(DATA_ROOT, "raw", "plos"),
        "BioLaySumm/BioLaySumm2025-eLife": os.path.join(DATA_ROOT, "raw", "elife")
    }
    
    # Download and save each dataset
    for dataset_name, output_dir in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")
        download_and_save_dataset(dataset_name, output_dir)

if __name__ == "__main__":
    main()
