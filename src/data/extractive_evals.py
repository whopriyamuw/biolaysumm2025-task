import argparse
import os

import pandas as pd
from datasets import load_dataset


def process_data(data_files, output_column):
    filename = os.path.splitext(os.path.basename(data_files))[0]
    output_path = filename + ".jsonl"

    print(filename)

    dataset = load_dataset(
        "whopriyam2/SUWMIT-dataset", data_files=data_files, split="train"
    )
    df = dataset.to_pandas()

    if output_column not in df.columns:
        raise ValueError(
            f"Column '{output_column}' not found in the dataset. Available columns: {df.columns.tolist()}"
        )

    evals_df = pd.DataFrame(
        {
            "generated_caption": df[output_column],
            "reference": df["summary"],
            "document": df["article"],
        }
    )
    evals_df.to_json(output_path, orient="records", lines=True)

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-files", required=True)
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


if __name__ == "__main__":
    main()
