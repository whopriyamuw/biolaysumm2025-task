import os
from enum import Enum

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class HfDatasets(Enum):
    elife = "BioLaySumm/BioLaySumm2025-eLife"
    plos = "BioLaySumm/BioLaySumm2025-PLOS"
    suwmit = "whopriyam2/SUWMIT-dataset"


class Prompts(Enum):
    ASSISTANT = "Lay Summary:"
    SYSTEM = (
        "You are a specialist medical communicator responsible for "
        "translating biomedical articles into a clear, accurate 10-20 sentence summary for non-experts. "
        "The summary should be at a Flesch–Kincaid grade level of 10–14 and explain any technical terms."
    )

    @staticmethod
    def chat_ml(article_text: str) -> list:
        return [
            {"role": "system", "content": Prompts.SYSTEM.value},
            {"role": "user", "content": article_text},
            {"role": "assistant", "content": Prompts.ASSISTANT.value},
        ]


class LlamaSummarizer:
    def __init__(
        self,
        dataset: str,
        split: str,
        checkpoint_path: str,
        batch_size: int = 1,
        input_field: str = "article",
    ):
        self._summaries = []
        self._dtype = torch.bfloat16
        self._model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self._model = None
        self._tokenizer = None

        self.batch_size = batch_size
        self.checkpoint_path = f"{checkpoint_path}.feather"
        self.input_field = input_field

        self.checkpoint_rate = 2  # save state every n batches.
        self.max_new_tokens = 384

        self.dataset = self._load_dataset(dataset, split)
        self._load_model()

    @classmethod
    def _load_dataset(cls, dataset: str, split: str):
        if dataset == HfDatasets.suwmit.name:
            return load_dataset(HfDatasets[dataset].value, data_files=split)["train"]

        return load_dataset(HfDatasets[dataset].value)[split]

    @classmethod
    def _get_device(cls):
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def _load_model(self):
        # https://github.com/huggingface/huggingface-llama-recipes
        torch.set_float32_matmul_precision("high")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, padding_side="left", use_fast=True
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=self._dtype,
        )
        self._model.generation_config = GenerationConfig(
            do_sample=False,
            dola_layers="low",  # https://arxiv.org/abs/2309.03883
            max_length=10000,
            max_new_tokens=self.max_new_tokens,
            cache_implementation="static",
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._model.generation_config.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        self._model.to(self._get_device())
        self._model.forward = torch.compile(
            self._model.forward, mode="reduce-overhead", fullgraph=True
        )

    def _read_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            df = pd.read_feather(self.checkpoint_path)
            self._summaries = df["summary"].tolist()
            print(f"Resuming from checkpoint (loaded {len(self._summaries)} records).")
        else:
            self._summaries = []
            print("No checkpoint found.")

        return len(self._summaries)

    def _batch_data(self, start_index: int, end_index: int) -> list[dict]:
        batch = self.dataset[self.input_field][start_index:end_index]
        prompts = [
            self._tokenizer.apply_chat_template(
                Prompts.chat_ml(sample),
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            for sample in batch
        ]

        return prompts

    def _parse_outputs(self, outputs):
        new_summaries = [s.strip() for s in outputs]
        self._summaries.extend(new_summaries)

    def _preprocess(self, start_index: int, end_index: int):
        data = self._batch_data(start_index, end_index)
        inputs = self._tokenizer(data, return_tensors="pt", padding=True).to(
            self._model.device
        )
        input_length = inputs["input_ids"].shape[1]

        return input_length, inputs

    def _postprocess(self, outputs, input_length: int, batch_number: int):
        self._parse_outputs(
            self._tokenizer.batch_decode(
                outputs[:, input_length:], skip_special_tokens=True
            )
        )

        if batch_number % self.checkpoint_rate == 0:
            pd.DataFrame({"summary": self._summaries}).to_feather(self.checkpoint_path)
            print(f"Checkpoint saved at batch {batch_number}.")

    def generate(self):
        start_index = self._read_checkpoint()
        batch_number = start_index // self.batch_size
        total_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size

        while start_index < len(self.dataset):
            batch_number += 1
            print(f"Processing batch {batch_number}/{total_batches}...")
            end_index = min(start_index + self.batch_size, len(self.dataset))

            input_length, inputs = self._preprocess(start_index, end_index)
            outputs = self._model.generate(**inputs)
            self._postprocess(outputs, input_length, batch_number)

            start_index = end_index

        return self._summaries

    def _save_json(self, output_path: str):
        df = pd.DataFrame(
            {
                "generated_caption": self._summaries,
                "reference": self.dataset["summary"],
                "document": self.dataset["article"],
            }
        )
        df.to_json(output_path, orient="records", lines=True)

    def _save_txt(self, output_path: str):
        with open(output_path, "w") as f:
            for summary in self._summaries:
                f.write(f"{summary}\n")

    def save(self, output_path: str):
        if not self._summaries:
            raise ValueError("No summaries to save. Generate summaries first.")

        file_extension = os.path.splitext(output_path)[-1].lower()

        match file_extension:
            case ".json":
                self._save_json(output_path)
            case ".txt":
                self._save_txt(output_path)


class LlamaSummarizerTuned(LlamaSummarizer):
    def __init__(self, adapter_path: str, *args, **kwargs):
        self._adapter_path = adapter_path
        super().__init__(input_field="extracted_summary", batch_size=8, *args, **kwargs)

    def _load_model(self):
        super()._load_model()

        self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
