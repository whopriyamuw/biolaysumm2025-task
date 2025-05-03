import os

from datasets import load_dataset

from relevance_eval import SUPPORTED_METRICS, DATASETS_DIR
from relevance_eval import main as relevance_scores


EXTRACTIVE_LENGTHS = {20, 30, 40}
EXTRACTIVE_METHODS = {
    "BioBERT",
    "MedEmbed-large",
    "S-PubMedBert-MS-MARCO",
    "medical-term-similarity",
    "pubmedbert_base",
}


def download_dataset(filename: str, split: str) -> None:
    dataset_filename = os.path.join(DATASETS_DIR, filename)
    if os.path.exists(filename):
        return

    dataset = load_dataset(
        "whopriyam2/SUWMIT-dataset", data_files=filename, split=split
    )
    os.makedirs(os.path.dirname(dataset_filename), exist_ok=True)
    dataset.to_csv(dataset_filename, index=False)


def evaluate(dataset_name: str, split: str, metrics: set[str]):
    for length in EXTRACTIVE_LENGTHS:
        for model in EXTRACTIVE_METHODS:
            filename = f"length_{length}/train_elife_{model}.csv"
            download_dataset(filename, split)

            relevance_scores(filename, dataset_name, split, metrics)


if __name__ == "__main__":
    evaluate("elife", "train", SUPPORTED_METRICS - {"gritlm"})
