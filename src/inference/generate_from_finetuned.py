import argparse

from models import LlamaSummarizerTuned

MODEL_CONFIGS = {
    "elife_abstract": {
        "adapter_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/models/finetuned_whopriyam2_SUWMIT_dataset_preprocessed_elife_train_abstract_only_preprocessed_llama_3.1_8b_instruct/epoch_7",
        "output_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_abstract_{split}_elife.jsonl",
        "checkpoint_path": "/gscratch/stf/jcols/biolaysum_summaries/finetuned_abstract_{split}_elife.ckpt",
        "dataset_split": "preprocessed/elife_validation_abstract_only_preprocessed.csv",
        "input_field": "transformed",
        "batch_size": 16,
        "max_new_tokens": 384,
    }
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
