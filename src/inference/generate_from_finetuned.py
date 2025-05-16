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
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_extractive_length_20_train_plos_BioBERT_llama_3.1_8b_instruct_with_dev/epoch_19",
        "output_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_{split}_plos.jsonl",
        "checkpoint_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_{split}_plos.ckpt",
        "dataset_split": "extractive/length_20/{split}_plos_BioBERT.csv",
        "input_field": "article",
        "batch_size": 1,
        "max_new_tokens": 384,
    },
    "elife_fulltext_factuality": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_extractive_length_20_train_elife_BioBERT_llama_3.1_8b_instruct_with_dev/epoch_7",
        "output_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_{split}_elife_factuality.jsonl",
        "checkpoint_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_{split}_elife_factuality.ckpt",
        "dataset_split": "extractive/length_20/{split}_elife_BioBERT.csv",
        "input_field": "article",
        "batch_size": 1,
        "max_new_tokens": 384,
        "prompt_version": "factuality",
    },
    "plos_fulltext_factuality": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_extractive_length_20_train_plos_BioBERT_llama_3.1_8b_instruct_with_dev/epoch_19",
        "output_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_{split}_plos_factuality.jsonl",
        "checkpoint_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_fulltext_{split}_plos_factuality.ckpt",
        "dataset_split": "extractive/length_20/{split}_plos_BioBERT.csv",
        "input_field": "article",
        "batch_size": 1,
        "max_new_tokens": 384,
        "prompt_version": "factuality",
    },
}


def main(model_name: str, split: str):
    config = MODEL_CONFIGS[model_name]
    config = {**config, "dataset": "suwmit"}

    for key in ["checkpoint_path", "output_path", "dataset_split"]:
        if key in config:
            config[key] = config[key].format(split=split)

    output_path = config.pop("output_path")
    adapter_path = config.pop("adapter_path")
    config["split"] = config.pop("dataset_split")

    summarizer = LlamaSummarizerTuned(adapter_path, **config)
    summarizer.generate()
    summarizer.save(output_path)


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
    main(args.model_name, args.split)
