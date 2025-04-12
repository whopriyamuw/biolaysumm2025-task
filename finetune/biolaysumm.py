import datasets

from config.data import Dataset


def load_data(split, dataset: Dataset):
    try:
        ds = datasets.load_dataset(f"BioLaySumm/BioLaySumm2025-{dataset}", split=split)
    except ValueError as e:
        if "trust_remote_code" in str(e):
            raise ValueError(
                "Loading Samsung/samsum requires you to execute the dataset script in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set HF_DATASETS_TRUST_REMOTE_CODE env variable to True.") from e
        else:
            raise e
    return ds


def get_data(dataset_config, tokenizer, split):
    CONTEXT_COLUMNS = ["title", "article"]
    PROMPT = (
        f"Summarize the following article:\n"
        f"{{article}}\n"
        f"---\n"
        f"Summary:\n"
    )

    def apply_prompt_template(sample):
        # Join title and article
        article = "\n".join(sample[col] for col in CONTEXT_COLUMNS)

        return {
            "prompt": PROMPT.format(article=article),
            "summary": sample["summary"],
        }

    def tokenize_add_label(sample):
        article = tokenizer.encode(tokenizer.bos_token + sample["article"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] + tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": article + summary,
            "attention_mask": [1] * (len(article) + len(summary)),
            "labels": [-100] * len(article) + summary,
        }

        return sample

    dataset = load_data(split, dataset_config.dataset)
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset
