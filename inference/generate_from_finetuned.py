import argparse

from models import LlamaSummarizerTuned

MODEL_CONFIGS = {
    "biobert_10_1e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_10_train_elife_BioBERT/epoch_3",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_10_epoch_1_validation_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_10_epoch_1_validation_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_10/validation_elife_BioBERT.csv",
    },
    "biobert_10_2e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_10_train_elife_BioBERT/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_10_epoch_2_validation_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_10_epoch_2_validation_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_10/validation_elife_BioBERT.csv",
    },
    "biobert_20_1e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_20_train_elife_BioBERT/epoch_3",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_20_epoch_1_validation_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_20_epoch_1_validation_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_20/validation_elife_BioBERT.csv",
    },
    "biobert_20_2e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_20_train_elife_BioBERT/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_20_epoch_2_validation_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_20_epoch_2_validation_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_20/validation_elife_BioBERT.csv",
    },
    "biobert_30_1e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_30_train_elife_BioBERT/epoch_3",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_30_epoch_1_validation_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_30_epoch_1_validation_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_30/validation_elife_BioBERT.csv",
    },
    "biobert_30_2e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_30_train_elife_BioBERT/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_30_epoch_2_validation_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_30_epoch_2_validation_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_30/validation_elife_BioBERT.csv",
    },
    "biobert_40_1e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_40_train_elife_BioBERT/epoch_3",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_epoch_1_validation_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_epoch_1_validation_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_40/validation_elife_BioBERT.csv",
    },
    "biobert_40_2e": {
        "adapter_path": "/gscratch/scrubbed/yongsinp/biolaysumm2025-task/models/finetuned_extractive_length_40_train_elife_BioBERT/epoch_7",
        "output_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_epoch_2_validation_elife_BioBERT.json",
        "checkpoint_path": "/gscratch/scrubbed/jcols/generated_summaries/finetuned_length_40_epoch_2_validation_elife_BioBERT.ckpt",
        "dataset_split": "extractive/length_40/validation_elife_BioBERT.csv",
    },
}


def main(config: dict):
    summarizer = LlamaSummarizerTuned(
        config["adapter_path"],
        config["dataset"],
        config["dataset_split"],
        config["checkpoint_path"],
    )
    summarizer.generate()
    summarizer.save(config["output_path"])


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
    args = parser.parse_args()
    main({**MODEL_CONFIGS[args.model_name], "dataset": "suwmit"})
