import argparse

from models import LlamaSummarizerTuned

MODEL_CONFIGS = {
    "elife_fulltext": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_BioLaySumm_BioLaySumm2025_eLife/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_{split}_elife_fulltext.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_{split}_elife_fulltext.ckpt",
        "dataset_split": "extractive/length_20/{split}_elife_BioBERT.csv",
        "input_field": "article",
        "batch_size": 1,
    },
    "elife_biobert_10": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_10sent_llama_3.1_8b_instruct/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_10_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_10_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_10sent.csv",
    },
    "elife_biobert_10_concat_abstract": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_10sent_concat_abstract_llama_3.1_8b_instruct/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_10_concat_abstract_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_10_concat_abstract_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_10sent_concat_abstract.csv",
    },
    "elife_biobert_10_exclude_abstract": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_10sent_exclude_abstract_llama_3.1_8b_instruct/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_10_exclude_abstract_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_10_exclude_abstract_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_10sent_exclude_abstract.csv",
    },
    "elife_biobert_20": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_20sent_llama_3.1_8b_instruct/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_20_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_20_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_20sent.csv",
    },
    "elife_biobert_20_concat_abstract": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_20sent_concat_abstract_llama_3.1_8b_instruct/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_20_concat_abstract_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_20_concat_abstract_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_20sent_concat_abstract.csv",
    },
    "elife_biobert_20_exclude_abstract": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_20sent_exclude_abstract_llama_3.1_8b_instruct/epoch_5",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_20_exclude_abstract_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_20_exclude_abstract_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_20sent_exclude_abstract.csv",
    },
    "elife_biobert_30": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_30sent_llama_3.1_8b_instruct/epoch_5",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_30_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_30_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_30sent.csv",
    },
    "elife_biobert_30_concat_abstract": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_30sent_concat_abstract_llama_3.1_8b_instruct/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_30_concat_abstract_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_30_concat_abstract_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_30sent_concat_abstract.csv",
    },
    "elife_biobert_30_exclude_abstract": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_30sent_exclude_abstract_llama_3.1_8b_instruct/epoch_5",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_30_exclude_abstract_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_30_exclude_abstract_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_30sent_exclude_abstract.csv",
    },
    "elife_biobert_40": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_40sent_llama_3.1_8b_instruct/epoch_5",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_40_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_40_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_40sent.csv",
    },
    "elife_biobert_40_concat_abstract": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_40sent_concat_abstract_llama_3.1_8b_instruct/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_40_concat_abstract_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_40_concat_abstract_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_40sent_concat_abstract.csv",
    },
    "elife_biobert_40_exclude_abstract": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_new_files_train_elife_BioBERT_40sent_exclude_abstract_llama_3.1_8b_instruct/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_40_exclude_abstract_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_40_exclude_abstract_{split}_elife_BioBERT.ckpt",
        "dataset_split": "new_files/{split}_elife_BioBERT_40sent_exclude_abstract.csv",
    },
    "elife_abstract_only": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_abstract_only_summaries_train_elife_abstracts_llama_3.1_8b_instruct/epoch_5",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_abstract_only_{split}_elife_BioBERT.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_abstract_only_{split}_elife_BioBERT.ckpt",
        "dataset_split": "abstract_only_summaries/{split}_elife_abstracts.csv",
    },
    "plos_abstract_only": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_abstract_only_summaries_train_plos_abstracts_llama_3.1_8b_instruct/epoch_5",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_abstract_only_{split}_plos_BioBERT.jsonl.jsonl",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_abstract_only_{split}_plos_BioBERT.jsonl.ckpt",
        "dataset_split": "abstract_only_summaries/{split}_plos_abstracts.csv",
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
