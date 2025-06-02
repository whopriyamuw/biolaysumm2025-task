import os
from enum import Enum
from textwrap import dedent

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
    @staticmethod
    def chat_ml(article_text: str, version: str) -> list:
        return [
            {"role": "system", "content": Prompts.system(version)},
            {"role": "user", "content": article_text},
            {"role": "assistant", "content": Prompts.assistant(version)},
        ]

    @staticmethod
    def assistant(version: str) -> str:
        return "Lay Summary:"

    @staticmethod
    def system(version: str) -> str:
        if version == "base":
            return (
                "You are a specialist medical communicator responsible for "
                "translating biomedical articles into a clear, accurate 10-20 sentence summary for non-experts. "
                "The summary should be at a Flesch–Kincaid grade level of 10–14 and explain any technical terms."
            )

        if version == "cot":
            return dedent(
                """
                You are a specialist medical communicator responsible for translating biomedical articles into clear, accurate summaries for non-experts.

                Your task is to generate a LAY summary in 10–20 sentences at a Flesch–Kincaid grade level of 10–14. You must ensure the summary is factually accurate, easy to understand, and explains any technical terms used.

                Follow these three steps:

                Step 1: Based on the provided extracted summary of the article as input, write an initial LAY summary that is clear, concise, and accessible to a general audience.

                Step 2: Compare your initial summary with the extracted summary. Reflect on whether all key claims are supported by the original text. Note any inconsistencies, hallucinations, or missing information.

                Step 3: Revise your summary as needed to improve factual accuracy and clarity. Output the corrected final version enclosed within <R> and </R> tags.
                """
            )

        if version == "factuality":
            return "You are a specialist medical communicator responsible for translating biomedical articles into a clear, accurate 10 to 20 sentence summary for non‑experts. The summary should have a Flesch–Kincaid grade level of 10 to 14, explaining any technical terms in simple language. Ensure factual accuracy by using terminology from the source article, and omit all in‑text citations."

        raise NotImplementedError


class LlamaSummarizer:
    def __init__(
        self,
        dataset: str,
        split: str,
        checkpoint_path: str,
        batch_size: int = 1,
        input_field: str = "article",
        max_new_tokens: int = 384,
        prompt_version: str = "base",
        decoding: str = "dola",
        base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ):
        self._summaries = []
        self._dtype = torch.bfloat16
        self._model_id = base_model
        self._model = None
        self._tokenizer = None
        self._decoding = decoding
        self._chat_template_config = {
            "tokenize": False,
            "add_generation_prompt": False,
            "continue_final_message": True,
        }

        self.batch_size = batch_size
        self.checkpoint_path = f"{checkpoint_path}.feather"
        self.input_field = input_field
        self.prompt_version = prompt_version

        self.checkpoint_rate = 2  # save state every n batches.
        self.max_new_tokens = max_new_tokens

        self.dataset = self._load_dataset(dataset, split)
        self._load_model()

        print(f"Running inference with config: {self.__dict__}")

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

    def _get_decoding_config(self) -> dict:
        decoding_config = {
            "dola": {
                "do_sample": False,
                "dola_layers": "low",  # https://arxiv.org/abs/2309.03883
            },
            "beam": {
                "do_sample": False,
                "early_stopping": True,
                "min_new_tokens": 100,
                "no_repeat_ngram_size": 3,
                "num_beams": 5,
                "num_return_sequences": 1,
            },
            "greedy": {},
        }

        return decoding_config.get(self._decoding, {})

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
            **self._get_decoding_config(),
            max_new_tokens=self.max_new_tokens,
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._model.generation_config.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        self._model.to(self._get_device())

    def _read_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            df = pd.read_feather(self.checkpoint_path)
            self._summaries = df["summary"].tolist()
            print(f"Resuming from checkpoint (loaded {len(self._summaries)} records).")
        else:
            self._summaries = []
            print("No checkpoint found.")

        return len(self._summaries)

    def _write_checkpoint(self, batch_number: int):
        pd.DataFrame({"summary": self._summaries}).to_feather(self.checkpoint_path)
        print(f"Checkpoint saved at batch {batch_number}.")

    def _batch_data(self, start_index: int, end_index: int) -> list[dict]:
        batch = self.dataset[self.input_field][start_index:end_index]
        prompts = [
            self._tokenizer.apply_chat_template(
                Prompts.chat_ml(sample, version=self.prompt_version),
                **self._chat_template_config,
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
            self._write_checkpoint(batch_number)

    def generate(self):
        start_index = self._read_checkpoint()
        batch_number = start_index // self.batch_size
        total_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size

        while start_index < len(self.dataset):
            batch_number += 1
            print(f"Processing batch {batch_number}/{total_batches}...")
            end_index = min(start_index + self.batch_size, len(self.dataset))

            input_length, inputs = self._preprocess(start_index, end_index)
            outputs = self._model.generate(
                **inputs, generation_config=self._model.generation_config
            )
            self._postprocess(outputs, input_length, batch_number)

            start_index = end_index

        self._write_checkpoint(batch_number)

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
            case ".jsonl":
                self._save_json(output_path)
            case ".txt":
                self._save_txt(output_path)

        print(f"Done! Saved {len(self._summaries)} summaries to {output_path}.")


class LlamaSummarizerTuned(LlamaSummarizer):
    def __init__(
        self, adapter_path: str, input_field: str, batch_size: int, *args, **kwargs
    ):
        self._adapter_path = adapter_path
        super().__init__(
            input_field=input_field,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()

        self._model = PeftModel.from_pretrained(self._model, self._adapter_path)


class InteractiveLlamaSummarizer(LlamaSummarizer):
    def __init__(self, prompt_version: str = "base", *args, **kwargs):
        kwargs.update(
            {
                "dataset": "interactive",
                "split": "none",
                "checkpoint_path": "interactive",
                "prompt_version": prompt_version,
            }
        )
        super().__init__(*args, **kwargs)

    @classmethod
    def _load_dataset(cls, dataset: str, split: str):
        return None

    def generate_from_prompt(self, user_message: str):
        messages = Prompts.chat_ml(user_message, version=self.prompt_version)

        prompt = self._tokenizer.apply_chat_template(
            messages, **self._chat_template_config
        )

        inputs = self._tokenizer([prompt], return_tensors="pt", padding=True).to(
            self._model.device
        )
        input_length = inputs["input_ids"].shape[1]

        outputs = self._model.generate(
            **inputs, generation_config=self._model.generation_config
        )

        response = self._tokenizer.batch_decode(
            outputs[:, input_length:], skip_special_tokens=True
        )[0].strip()

        return response
