import argparse

from cli import validate_args
from models import LlamaSummarizer


def main(
    output_path: str,
    checkpoint_path: str,
    dataset: str,
    dataset_split: str,
):
    summarizer = LlamaSummarizer(dataset, dataset_split, checkpoint_path)
    summarizer.generate()
    summarizer.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on the base Llama 3.1 8B Instruct model."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path where to save the generated summaries",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path where to save the summaries checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the input dataset to use",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="validation",
        help="Dataset split or data_file to use",
    )

    args = parser.parse_args()
    validated_args = validate_args(args)

    main(**validated_args)
