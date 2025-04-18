import json
import os

from datasets import load_dataset

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

EXTRACTIVE_LENGTHS = {20, 30, 40}
EXTRACTIVE_METHODS = {
    "BioBERT",
    "MedEmbed-large",
    "S-PubMedBert-MS-MARCO",
    "medical-term-similarity",
    "pubmedbert_base",
}


def download_dataset(filename: str, split: str) -> None:
    original_filename = filename.split(".")[0]
    processed_filename = os.path.join(PROCESSED_DIR, f"{original_filename}.json")

    if os.path.exists(processed_filename):
        return

    dataset = load_dataset(
        "whopriyam2/SUWMIT-dataset", data_files=filename, split=split
    )
    sampled = dataset.shuffle(seed=1704).select(range(100))
    os.makedirs(os.path.dirname(processed_filename), exist_ok=True)

    data = [
        {
            "document": r["article"],
            "reference": r["summary"],
            "generated_caption": r["extracted_summary"],
        }
        for r in sampled
    ]

    with open(processed_filename, "w") as f:
        json.dump(data, f)


def main():
    for length in EXTRACTIVE_LENGTHS:
        for model in EXTRACTIVE_METHODS:
            filename = f"length_{length}/train_elife_{model}.csv"
            print(filename)
            download_dataset(filename, "train")


if __name__ == "__main__":
    main()
