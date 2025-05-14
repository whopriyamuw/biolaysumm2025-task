import argparse

from models import LlamaSummarizerTuned

MODEL_CONFIGS = {
    "elife_fulltext": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_extractive_length_20_train_elife_BioBERT_llama_3.1_8b_instruct_with_dev/epoch_7",
        "output_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_{split}_elife.jsonl",
        "checkpoint_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_{split}_elife.ckpt",
        "dataset_split": "extractive/length_20/{split}_elife_BioBERT.csv",
        "input_field": "article",
        "batch_size": 1,
        "max_new_tokens": 384,
    },
    "plos_fulltext": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_extractive_length_20_train_plos_BioBERT_llama_3.1_8b_instruct_with_dev/epoch_9",
        "output_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_{split}_plos.jsonl",
        "checkpoint_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_{split}_plos.ckpt",
        "dataset_split": "extractive/length_20/{split}_plos_BioBERT.csv",
        "input_field": "article",
        "batch_size": 1,
        "max_new_tokens": 384,
    },
    "plos_fulltext_256tokens": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_extractive_length_20_train_plos_BioBERT_llama_3.1_8b_instruct_with_dev/epoch_9",
        "output_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_256tokens_{split}_plos.jsonl",
        "checkpoint_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_256tokens_{split}_plos.ckpt",
        "dataset_split": "extractive/length_20/{split}_plos_BioBERT.csv",
        "input_field": "article",
        "batch_size": 1,
        "max_new_tokens": 256,
    },
}


def main(config: dict, split: str):
    kwargs = {
        "batch_size": config.get("batch_size", 8),
        "checkpoint_path": config["checkpoint_path"].format(split=split),
        "dataset": config["dataset"],
        "input_field": config.get("input_field", "extracted_summary"),
        "max_new_tokens": config.get("max_new_tokens", 384),
        "split": config["dataset_split"].format(split=split),
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
