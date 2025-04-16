import argparse
import enum
import json
import sys

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from torch.nn.functional import cosine_similarity

SUPPORTED_METRICS = {"sacrebleu", "rouge", "bertscore", "comet", "gritlm"}


class Datasets(enum.Enum):
    elife = "BioLaySumm/BioLaySumm2025-eLife"
    plos = "BioLaySumm/BioLaySumm2025-PLOS"


def gritlm_instruction(instruction: str):
    return (
        "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    )


def gritlm_similarity(references: list[str], predictions: list[str]):
    from gritlm import GritLM

    model = GritLM("GritLM/GritLM-7B", torch_dtype="auto", device_map="auto")

    # This configuration works on a A100 40 GB GPU
    ref_emb = model.encode(
        references, instruction=gritlm_instruction(""), batch_size=48
    )
    hyp_emb = model.encode(
        predictions, instruction=gritlm_instruction(""), batch_size=48
    )

    ref_t = torch.tensor(ref_emb, device=model.device)
    hyp_t = torch.tensor(hyp_emb, device=model.device)
    sims = cosine_similarity(ref_t, hyp_t, dim=1).cpu().numpy()

    return {
        "score": float(sims.mean()),
    }


def relevance_eval(
    ref: list[list[str]],
    hyp: list[str],
    metrics: set[str],
    metric_kwargs: dict,
) -> dict:
    results = {}
    custom_scorers = {
        "gritlm": gritlm_similarity,
    }

    for metric_name in metrics:
        kwargs = {
            "references": ref,
            "predictions": hyp,
            **metric_kwargs.get(metric_name, {}),
        }

        if metric_name in custom_scorers:
            results[metric_name] = custom_scorers[metric_name](**kwargs)
        else:
            metric = evaluate.load(metric_name)
            results[metric_name] = metric.compute(**kwargs)

    return results


def main(
    system_output_filename: str, dataset_name: str, split: str, metrics: set[str]
) -> None:
    dataset_id = Datasets[dataset_name].value
    dataset = load_dataset(dataset_id, split=split)
    predictions_df = pd.read_csv(system_output_filename)
    metric_kwargs = {
        "bertscore": {"lang": "en"},
        "comet": {
            "gpus": 1,
            "sources": dataset["article"],
            "references": dataset["summary"],
        },
        "gritlm": {
            "references": dataset["summary"],
        },
    }

    assert len(dataset["article"]) == len(predictions_df["summary"])

    results = relevance_eval(
        # Most metrics expect multiple references per sample.
        [s for s in dataset["summary"]],
        predictions_df["summary"],
        metrics,
        metric_kwargs,
    )

    json.dump(results, sys.stdout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("system_output_filename", type=str)
    parser.add_argument("dataset", type=str, choices=[d.name for d in Datasets])
    parser.add_argument("split", type=str, choices=["train", "validation"])
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        type=str,
        choices=SUPPORTED_METRICS,
        default=list(SUPPORTED_METRICS - {"gritlm"}),
    )
    args = parser.parse_args()

    main(
        args.system_output_filename, args.dataset, args.split, metrics=set(args.metrics)
    )
