"""Script to prepare model outputs for evaluation."""

import os
import pandas as pd
from datasets import load_dataset

from .dataset_configs import DATASET_CONFIGS


def process_ckpt(feather_path: str, data_files: str) -> str:
    """Process a model checkpoint file and prepare it for evaluation."""
    filename = os.path.splitext(feather_path)[0]
    output_path = f"{filename}.jsonl"

    # Load model outputs and validation data
    ckpt_df = pd.read_feather(feather_path)
    dataset_df = load_dataset(
        "whopriyam2/SUWMIT-dataset", 
        data_files=data_files, 
        split="train"
    ).to_pandas()

    # Create evaluation dataframe
    evals_df = pd.DataFrame({
        "generated_caption": ckpt_df["summary"],
        "reference": dataset_df["summary"],
        "document": dataset_df["article"],
    })
    
    # Save to JSONL format
    evals_df.to_json(output_path, orient="records", lines=True)

    return output_path


def main():
    """Process all model checkpoints and prepare them for evaluation."""
    for feather_path, config in DATASET_CONFIGS.items():
        output_path = process_ckpt(feather_path, config["data_files"])
        print(f"Processed {feather_path} -> {output_path}")


if __name__ == "__main__":
    main()
