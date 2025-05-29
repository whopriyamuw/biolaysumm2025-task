"""Script to process and convert datasets for extractive evaluation."""

import argparse
import os
from typing import Optional

import pandas as pd
from datasets import load_dataset


def process_data(data_files: str, output_column: str) -> str:
    """Process the dataset and convert it to evaluation format."""
    # Extract filename without extension for output path
    filename = os.path.splitext(os.path.basename(data_files))[0]
    output_path = f"{filename}.jsonl"
    
    # Load and validate dataset
    dataset = load_dataset(
        "whopriyam2/SUWMIT-dataset", 
        data_files=data_files, 
        split="train"
    )
    df = dataset.to_pandas()
    
    # Validate output column exists
    if output_column not in df.columns:
        raise ValueError(
            f"Column '{output_column}' not found in the dataset. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Create evaluation dataframe with required columns
    evals_df = pd.DataFrame({
        "generated_caption": df[output_column],
        "reference": df["summary"],
        "document": df["article"],
    })
    
    # Save to JSONL format
    evals_df.to_json(output_path, orient="records", lines=True)
    
    return output_path


def main():
    """Main function to process the dataset and create evaluation file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-files",
        required=True,
    )
    parser.add_argument(
        "--output-column",
        required=True,
    )
    
    args = parser.parse_args()
    
    try:
        output_path = process_data(args.data_files, args.output_column)
        print(f"Successfully processed {args.data_files} -> {output_path}")
    except Exception as e:
        print(f"Error processing file: {e}")
        raise


if __name__ == "__main__":
    main()
