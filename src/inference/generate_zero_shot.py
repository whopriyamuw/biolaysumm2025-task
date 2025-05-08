import argparse

from models import LlamaSummarizer


def main(*args, **kwargs):
    output_path = kwargs.pop("output_path")
    summarizer = LlamaSummarizer(*args, **kwargs)
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )
    parser.add_argument("--prompt-version", type=str, default="base")

    main(**vars(parser.parse_args()))
