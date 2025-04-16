import os

os.environ["TOKENIZERS_PARALLELISM"] = "True"

import argparse
import enum
import json

import evaluate
import pandas as pd
from datasets import load_dataset


class Datasets(enum.Enum):
    elife = "BioLaySumm/BioLaySumm2025-eLife"
    plos = "BioLaySumm/BioLaySumm2025-PLOS"


def relevance_eval(
    ref: list[list[str]],
    sys: list[str],
    metrics: set[str],
    metric_kwargs: dict,
) -> dict:
    results = {}

    for metric_name in metrics:
        metric = evaluate.load(metric_name)
        kwargs = {
            "references": ref,
            "predictions": sys,
            **metric_kwargs.get(metric_name, {}),
        }
        results[metric_name] = metric.compute(**kwargs)

    return results


def main(dataset_name: str, split: str, system_output_filename: str) -> None:
    dataset_id = Datasets[dataset_name].value
    dataset = load_dataset(dataset_id, split=split)
    sys_df = pd.read_csv(system_output_filename)

    metrics = {"sacrebleu", "rouge", "bertscore", "comet"}
    metric_kwargs = {
        "bertscore": {"lang": "en"},
        "comet": {
            "gpus": 1,
            "sources": dataset["article"],
            "references": sys_df["summary"],
        },
    }

    assert len(dataset["article"]) == len(sys_df["summary"])

    results = relevance_eval(
        # Most metrics expect multiple references per sample.
        [s for s in dataset["summary"]],
        sys_df["summary"],
        metrics,
        metric_kwargs,
    )

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=[d.name for d in Datasets])
    parser.add_argument("split", type=str, choices=["train", "validation"])
    parser.add_argument("system_output_filename", type=str)
    args = parser.parse_args()

    main(args.dataset, args.split, args.system_output_filename)
