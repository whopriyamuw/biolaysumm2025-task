import datasets

from config.data import Dataset


def load_data(split, dataset: Dataset):
    return datasets.load_dataset(f"BioLaySumm/BioLaySumm2025-{dataset}", split=split)


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
