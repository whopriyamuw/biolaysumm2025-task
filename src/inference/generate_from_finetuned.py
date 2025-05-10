import argparse

from models import LlamaSummarizerTuned

MODEL_CONFIGS = {
    "elife_fulltext": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_BioLaySumm_BioLaySumm2025_eLife/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_{split}_elife_fulltext.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_{split}_elife_fulltext.ckpt",
        "dataset_split": "extractive/length_20/{split}_elife_BioBERT.csv",
        "input_field": "article",
        "batch_size": 1,
    },
    "elife_biobert_10_1e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_10_train_elife_BioBERT/epoch_3",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_10_epoch_1_{split}_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_10_epoch_1_{split}_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_10/{split}_elife_BioBERT.csv",
    },
    "elife_biobert_10_2e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_10_train_elife_BioBERT/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_10_epoch_2_{split}_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_10_epoch_2_{split}_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_10/{split}_elife_BioBERT.csv",
    },
    "elife_biobert_20_1e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_20_train_elife_BioBERT/epoch_3",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_20_epoch_1_{split}_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_20_epoch_1_{split}_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_20/{split}_elife_BioBERT.csv",
    },
    "elife_biobert_20_2e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_20_train_elife_BioBERT/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_20_epoch_2_{split}_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_20_epoch_2_{split}_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_20/{split}_elife_BioBERT.csv",
    },
    "elife_biobert_30_1e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_30_train_elife_BioBERT/epoch_3",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_30_epoch_1_{split}_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_30_epoch_1_{split}_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_30/{split}_elife_BioBERT.csv",
    },
    "elife_biobert_30_2e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_30_train_elife_BioBERT/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_30_epoch_2_{split}_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_30_epoch_2_{split}_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_30/{split}_elife_BioBERT.csv",
    },
    "elife_biobert_40_1e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_40_train_elife_BioBERT/epoch_3",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_epoch_1_{split}_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_epoch_1_{split}_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_40/{split}_elife_BioBERT.csv",
    },
    "elife_biobert_40_2e": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_40_train_elife_BioBERT/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_epoch_2_{split}_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_epoch_2_{split}_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_40/{split}_elife_BioBERT.csv",
    },
    "elife_biobert_40_all": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_extractive_length_40_train_elife_BioBERT_with_dev/epoch_9",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_all_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_all_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_40/{split}_elife_BioBERT.csv",
    },
    "plos_biobert_20": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_extractive_length_20_train_plos_BioBERT/epoch_9",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_20_{split}_plos_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_20_{split}_plos_BioBERT.ckpt",
        "dataset_split": "extractive/length_20/{split}_plos_BioBERT.csv",
    },
    "elife_biobert_40_70B": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/old/finetuned_whopriyam2_SUWMIT_dataset_extractive_length_40_train_elife_BioBERT/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_70B_{split}_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_70B_{split}_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_40/{split}_elife_BioBERT.csv",
        "base_model": "meta-llama/Llama-3.1-70B-Instruct",
        "batch_size": 1,
    },
}


def main(config: dict, split: str):
    kwargs = {
        "batch_size": config.get("batch_size", 8),
        "checkpoint_path": config["checkpoint_path"].format(split=split),
        "dataset": config["dataset"],
        "split": config["dataset_split"].format(split=split),
        "input_field": config.get("input_field", "extracted_summary"),
    }

    if "base_model" in config:
        kwargs["base_model"] = config["base_model"]

    summarizer = LlamaSummarizerTuned(config["adapter_path"], **kwargs)
    summarizer.generate()
    summarizer.save(config["output_path"].format(split=split))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a fine-tuned Llama 3.1 8B Instruct model."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=MODEL_CONFIGS.keys(),
    )
    parser.add_argument(
        "--split", type=str, default="validation", choices=["validation", "test"]
    )
    args = parser.parse_args()
    main({**MODEL_CONFIGS[args.model_name], "dataset": "suwmit"}, args.split)
