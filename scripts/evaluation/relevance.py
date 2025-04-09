import argparse
import json

import evaluate


def read_lines(filename: str) -> list[str]:
    with open(filename, encoding="utf-8-sig") as f:
        return f.read().strip().split("\n")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("full_ref_filename")
    parser.add_argument("summary_ref_filename")
    parser.add_argument("summary_sys_filename")
    args = parser.parse_args()

    src = read_lines(args.full_ref_filename)
    ref = read_lines(args.summary_ref_filename)
    sys = read_lines(args.summary_sys_filename)

    # Some metrics expect multiple references per sample.
    ref_multi = [[line] for line in ref]

    metrics = {"sacrebleu", "rouge", "bertscore", "comet"}
    metric_kwargs = {
        "bertscore": {"lang": "en"},
        "comet": {"sources": src, "references": ref, "gpus": 1},
    }

    results = relevance_eval(ref_multi, sys, metrics, metric_kwargs)

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
