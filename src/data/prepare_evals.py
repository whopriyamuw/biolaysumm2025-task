import os

import pandas as pd
from datasets import load_dataset

DATASET_CONFIGS = {
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_10_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_10sent.csv",
    },
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_10_concat_abstract_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_10sent_concat_abstract.csv",
    },
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_10_exclude_abstract_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_10sent_exclude_abstract.csv",
    },
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_20_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_20sent.csv",
    },
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_20_concat_abstract_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_20sent_concat_abstract.csv",
    },
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_20_exclude_abstract_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_20sent_exclude_abstract.csv",
    },
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_30_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_30sent.csv",
    },
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_30_concat_abstract_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_30sent_concat_abstract.csv",
    },
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_30_exclude_abstract_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_30sent_exclude_abstract.csv",
    },
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_40_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_40sent.csv",
    },
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_40_concat_abstract_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_40sent_concat_abstract.csv",
    },
    "/gscratch/scrubbed/jcols/generated_summaries/finetuned_40_exclude_abstract_validation_elife_BioBERT.ckpt.feather": {
        "data_files": "new_files/validation_elife_BioBERT_40sent_exclude_abstract.csv",
    },
}


def process_ckpt(feather_path, data_files):
    filename = os.path.splitext(feather_path)[0]
    output_path = filename + ".jsonl"

    ckpt_df = pd.read_feather(feather_path)["train"]
    dataset_df = load_dataset(
        "whopriyam2/SUWMIT-dataset", data_files=data_files, split="validation"
    ).to_pandas()

    evals_df = pd.DataFrame(
        {
            "generated_caption": ckpt_df["summary"],
            "reference": dataset_df["summary"],
            "document": dataset_df["article"],
        }
    )
    evals_df.to_json(output_path, orient="records", lines=True)

    return output_path


def main():
    for feather_path, config in DATASET_CONFIGS.items():
        output_path = process_ckpt(feather_path, **config)
        print(f"Processed {feather_path} -> {output_path}")


if __name__ == "__main__":
    main()
