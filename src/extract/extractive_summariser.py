import argparse
import os.path
import re
import time
from enum import Enum

import pytextrank  # noqa
import spacy
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from utils import DATA_ROOT


class HfDatasets(Enum):
    elife = "BioLaySumm/BioLaySumm2025-eLife"
    plos = "BioLaySumm/BioLaySumm2025-PLOS"


class BioBERTSummarizer:
    """Extractive summarizer using BioBERT and TextRank."""

    def __init__(self, sentence_count: int = 20, device: str = None):
        self.sentence_count = sentence_count
        self.device = device or self._get_device()

        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("textrank")

        print("Loading BioBERT model...")
        self.model = SentenceTransformer(
            "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            device=self.device,
        )

    @staticmethod
    def _preprocess_text(text: str) -> str:
        # Remove parenthesized/braced/bracketed spans
        text = re.sub(r"\([^()]*\)", " ", text)
        text = re.sub(r"\{[^{}]*\}", " ", text)
        text = re.sub(r"\[[^\[\]]*\]", " ", text)

        # Collapse spaces around punctuation
        text = re.sub(r"\s*([^\w\s])\s*", r"\1", text)

        # Preserve decimals
        text = re.sub(r"(?<!\d)([\.!?])(?!\d)(?=\S)", r"\1 ", text)

        # Insert space after punctuation marks
        text = re.sub(r"([,;:%])(?=\S)", r"\1 ", text)

        # Clean spaces
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def extract_and_rank(self, text: str) -> str:
        """Extract and rank sentences from the input text."""
        if not isinstance(text, str) or not text.strip():
            return ""

        doc = self.nlp(text)
        textrank_sents = [
            sent.text.strip()
            for sent in doc._.textrank.summary(limit_sentences=self.sentence_count * 4)
        ]

        if not textrank_sents:
            return ""

        sentence_embeddings = self.model.encode(textrank_sents, convert_to_tensor=True)
        doc_embedding = self.model.encode(text, convert_to_tensor=True)

        similarities = (
            util.cos_sim(sentence_embeddings, doc_embedding).squeeze().cpu().numpy()
        )
        ranked_sents = [
            sent for _, sent in sorted(zip(similarities, textrank_sents), reverse=True)
        ]

        return " ".join(ranked_sents[: self.sentence_count])

    def process_dataset(
        self, dataset: str, split: str, preprocess: bool = False
    ) -> Dataset:
        print(f"Loading {dataset} dataset ({split} split)...")
        input_df = load_dataset(HfDatasets[dataset].value, split=split).to_pandas()
        total_docs = len(input_df)

        start_time = time.time()

        articles = input_df["article"]
        output_df = input_df[["article", "summary"]].copy()

        if preprocess:
            input_df["preprocessed_article"] = [
                self._preprocess_text(text)
                for text in tqdm(
                    articles,
                    desc="Preprocessing",
                    total=total_docs,
                    unit="doc",
                )
            ]
            articles = input_df["preprocessed_article"]

        output_df["extracted_summary"] = [
            self.extract_and_rank(text)
            for text in tqdm(articles, desc="Summarizing", total=total_docs, unit="doc")
        ]

        print(f"\nTime taken to extract: {time.time() - start_time:.2f} seconds")

        return Dataset.from_pandas(output_df)

    def save(
        self,
        data: Dataset | None,
        input_dataset: str,
        split: str,
        output_dataset: str,
        preprocess: bool,
    ) -> None:
        output_dir = os.path.join(DATA_ROOT, "processed", "extractive")
        os.makedirs(output_dir, exist_ok=True)

        filename_parts = [
            split,
            input_dataset,
            "BioBERT",
            f"{self.sentence_count}sent",
        ]
        if preprocess:
            filename_parts.append("preprocessed")

        local_path = os.path.join(output_dir, f"{'_'.join(filename_parts)}.csv")

        if data is None:
            if os.path.exists(local_path):
                print(f"Loading existing processed file from: {local_path}")
                data = Dataset.from_csv(local_path)
            else:
                raise ValueError("No data provided and no existing file found")
        else:
            data.to_csv(local_path)
            print(f"Results saved locally to: {local_path}")

        print(f"\nPushing dataset to HuggingFace: {output_dataset}")
        variant_config = f"{self.sentence_count}sent"
        if preprocess:
            variant_config += "_preprocessed"

        api = HfApi()
        try:
            api.create_repo(repo_id=output_dataset, repo_type="dataset", exist_ok=True)
            data.push_to_hub(
                output_dataset,
                split=split,
                config_name=variant_config,
                private=False,
                embed_external_files=False,
            )
            print(
                f"Successfully uploaded to: https://huggingface.co/datasets/{output_dataset}"
                f" (config: {variant_config}, split: {split})"
            )
        except Exception as e:
            print(f"Failed to upload to HuggingFace Hub: {str(e)}")


def main(
    sentence_count: int, dataset: str, split: str, output_dataset: str, preprocess: bool
) -> None:
    summarizer = BioBERTSummarizer(sentence_count=sentence_count)
    save_params = {
        "input_dataset": dataset,
        "split": split,
        "output_dataset": output_dataset,
        "preprocess": preprocess,
    }

    try:
        summarizer.save(data=None, **save_params)
    except ValueError:
        processed_dataset = summarizer.process_dataset(
            dataset=dataset, split=split, preprocess=preprocess
        )
        summarizer.save(data=processed_dataset, **save_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run extractive summarization using BioBERT embeddings."
    )
    parser.add_argument(
        "--sentence-count",
        type=int,
        default=20,
        help="Number of sentences in the generated summary",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[ds.name for ds in HfDatasets],
        default="elife",
        help="Dataset to process",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to process",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        default="josecols/suwmit",
        help="HuggingFace Hub dataset ID to upload output",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Whether to preprocess input texts",
    )

    args = parser.parse_args()
    main(**vars(args))
