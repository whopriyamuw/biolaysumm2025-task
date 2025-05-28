import argparse
import os
import re
from dataclasses import dataclass
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


@dataclass
class TransformOptions:
    abstract_only: bool
    concat_abstract: bool
    exclude_abstract: bool
    extract: bool
    preprocess: bool

    def __post_init__(self):
        if self.abstract_only and self.extract:
            raise ValueError("Cannot use abstract_only and extract together")

        if not self.extract and (self.concat_abstract or self.exclude_abstract):
            raise ValueError(
                "Cannot use concat_abstract or exclude_abstract without extract"
            )


class ArticleTransformer:
    """Article transformation pipeline"""

    def __init__(
        self,
        dataset: str,
        split: str,
        output_dataset: str,
        output_dataset_dir: str,
        transform_options: TransformOptions,
    ):
        self.device = self._get_device()
        self.output_dataset = output_dataset
        self.output_dataset_dir = output_dataset_dir
        self.transform_options = transform_options
        self.extractive_lengths = {10, 20, 30, 40}
        self.cache_dir = os.path.join(DATA_ROOT, "interim")
        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"Loading {dataset} dataset ({split} split)...")
        self._source_dataset, self._source_dataset_split = dataset, split
        self.dataset = load_dataset(HfDatasets[dataset].value, split=split).to_pandas()

    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

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
    def _split_abstract(text: str) -> tuple[str, str]:
        parts = text.split("\n", 1)
        if len(parts) > 1:
            return parts[0].strip(), parts[1].strip()

        return parts[0].strip(), ""

    def _get_split_articles(self):
        abstracts, bodies = [], []
        for article in tqdm(
            self.dataset["article"], desc="Processing", total=len(self.dataset)
        ):
            abstract, body = self._split_abstract(article)
            abstracts.append(abstract)
            bodies.append(body)

        return abstracts, bodies

    def _get_cached_path(self, n_sentences=None):
        filename_parts = [
            self._source_dataset,
            self._source_dataset_split,
        ]

        if self.transform_options.abstract_only:
            filename_parts.append("abstract_only")

        if self.transform_options.extract:
            if n_sentences is None:
                raise ValueError

            filename_parts.append("extractive")
            filename_parts.append(str(n_sentences))

        if self.transform_options.exclude_abstract:
            filename_parts.append("exclude_abstract")

        if self.transform_options.concat_abstract:
            filename_parts.append("concat_abstract")

        if self.transform_options.preprocess:
            filename_parts.append("preprocessed")

        filename = "_".join(filename_parts) + ".csv"

        return os.path.join(self.cache_dir, filename), filename

    def extract_and_rank(self, text: str) -> list[str]:
        """Extract and rank sentences from the input text, returning a list of ranked sentences."""
        if not isinstance(text, str) or not text.strip():
            return []

        print("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")

        print("Loading BioBERT model...")
        model = SentenceTransformer(
            "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            device=self.device,
        )

        doc = nlp(text)
        textrank_sents = [
            sent.text.strip() for sent in doc._.textrank.summary(limit_sentences=80)
        ]

        if not textrank_sents:
            return []

        sentence_embeddings = model.encode(textrank_sents, convert_to_tensor=True)
        doc_embedding = model.encode(text, convert_to_tensor=True)

        similarities = (
            util.cos_sim(sentence_embeddings, doc_embedding).squeeze().cpu().numpy()
        )
        ranked_sents = [
            sent for _, sent in sorted(zip(similarities, textrank_sents), reverse=True)
        ]

        return ranked_sents

    def extract_summaries(self, abstracts: list, bodies: list, summary_lengths: set):
        grouped_summaries = {f"extracted_summary_{n}": [] for n in summary_lengths}

        if self.transform_options.exclude_abstract:
            extractive_input = bodies
        else:
            extractive_input = self.dataset["article"]

        if self.transform_options.preprocess:
            extractive_input = list(map(self._preprocess_text, extractive_input))

        for i, text in enumerate(
            tqdm(extractive_input, desc="Summarizing", total=len(self.dataset))
        ):
            sentences = self.extract_and_rank(text)
            for n in summary_lengths:
                summary = "\n".join(sentences[:n])
                if self.transform_options.concat_abstract:
                    summary = f"{abstracts[i]}\n{summary}".strip()

                grouped_summaries[f"extracted_summary_{n}"].append(summary)

        return grouped_summaries

    def transform(self) -> Dataset:
        abstracts, bodies = self._get_split_articles()
        output_df = self.dataset[["article", "summary"]].copy()

        if self.transform_options.extract:
            grouped_summaries = self.extract_summaries(
                abstracts, bodies, self.extractive_lengths
            )

            for n in self.extractive_lengths:
                output_df[f"extracted_summary_{n}"] = grouped_summaries[
                    f"extracted_summary_{n}"
                ]
        else:
            if self.transform_options.abstract_only:
                output_df["transformed"] = abstracts
            else:
                output_df["transformed"] = output_df["article"]

            if self.transform_options.preprocess:
                output_df["transformed"] = list(
                    map(self._preprocess_text, output_df["transformed"])
                )

        return Dataset.from_pandas(output_df)

    def _save_single(self, df=None, n_sentences=None):
        cached_path, filename = self._get_cached_path(n_sentences)
        if df is None and not os.path.exists(cached_path):
            raise ValueError()

        if df is not None:
            columns = ["article", "summary", "transformed"]
            df[columns].to_csv(cached_path, index=False)
            print(f"Saved the transformed data to {cached_path}.")

        try:
            path_in_repo = os.path.join(self.output_dataset_dir, filename)
            api = HfApi()
            api.create_repo(
                repo_id=self.output_dataset, repo_type="dataset", exist_ok=True
            )
            api.upload_file(
                path_or_fileobj=cached_path,
                path_in_repo=path_in_repo,
                repo_id=self.output_dataset,
                repo_type="dataset",
            )
            print(
                f"Successfully uploaded to: https://huggingface.co/datasets/{self.output_dataset}/blob/main/{path_in_repo}"
            )
        except Exception as e:
            print(f"Failed to upload to HuggingFace Hub: {str(e)}")

    def _save_multiple(self, df=None):
        single = df.copy() if df is not None else None

        for n in self.extractive_lengths:
            if single is not None:
                single.rename(
                    columns={f"extracted_summary_{n}": "transformed"}, inplace=True
                )

            self._save_single(single, n_sentences=n)

    def save(self, data=None):
        df = data.to_pandas() if data else None

        if self.transform_options.extract:
            self._save_multiple(df)
        else:
            self._save_single(df)


def main(dataset, split, output_dataset, output_dataset_dir, **kwargs) -> None:
    # Full-text without preprocessing by default
    transform_options = TransformOptions(
        abstract_only=kwargs.get("abstract_only", False),
        concat_abstract=kwargs.get("concat_abstract", False),
        preprocess=kwargs.get("preprocess", False),
        exclude_abstract=kwargs.get("exclude_abstract", False),
        extract=kwargs.get("extract", False),
    )
    summarizer = ArticleTransformer(
        dataset, split, output_dataset, output_dataset_dir, transform_options
    )

    # Use cached data, if available
    try:
        summarizer.save()
    except ValueError:
        processed_dataset = summarizer.transform()
        summarizer.save(data=processed_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run extractive summarization using BioBERT embeddings."
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        required=True,
        help="HuggingFace Hub dataset ID to upload output",
    )
    parser.add_argument(
        "--output-dataset-dir",
        type=str,
        required=True,
        help="Output dir in the Hugging Face dataset",
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
        default="train",
        help="Dataset split to process",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Whether to preprocess input texts",
    )
    parser.add_argument(
        "--concat-abstract",
        action="store_true",
        help="Whether to prepend the abstract to the transformed article",
    )
    parser.add_argument(
        "--exclude-abstract",
        action="store_true",
        help="Whether to exclude the abstract in the text for ranking",
    )
    parser.add_argument(
        "--abstract-only",
        action="store_true",
        help="Whether to use only the article abstract",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Whether use extractive summarization",
    )
    args = parser.parse_args()
    main(**vars(args))
