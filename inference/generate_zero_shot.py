import argparse

from models import LlamaSummarizer


def main(
    output_path: str,
    checkpoint_path: str,
    dataset: str,
    dataset_split: str,
    input_field: str = "article",
    batch_size: int = 1,
):
    summarizer = LlamaSummarizer(
        dataset,
        dataset_split,
        checkpoint_path,
        batch_size=batch_size,
        input_field=input_field,
    )
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
    parser.add_argument(
        "--input-field",
        type=str,
        default="article",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    main(**vars(args))
