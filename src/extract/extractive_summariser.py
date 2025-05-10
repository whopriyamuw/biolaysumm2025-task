import argparse
import os
import re
import time
from enum import Enum
import pandas as pd

import pytextrank  # noqa
import spacy
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


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

    def extract_and_rank(self, text: str) -> list[str]:
        """Extract and rank sentences from the input text, returning a list of ranked sentences."""
        if not isinstance(text, str) or not text.strip():
            return []
        print ("EXTRACTING SUMMARIES")
        doc = self.nlp(text)
        textrank_sents = [
            sent.text.strip()
            for sent in doc._.textrank.summary(limit_sentences=80)
        ]

        if not textrank_sents:
            return []

        sentence_embeddings = self.model.encode(textrank_sents, convert_to_tensor=True)
        doc_embedding = self.model.encode(text, convert_to_tensor=True)

        similarities = (
            util.cos_sim(sentence_embeddings, doc_embedding).squeeze().cpu().numpy()
        )
        ranked_sents = [
            sent for _, sent in sorted(zip(similarities, textrank_sents), reverse=True)
        ]

        return ranked_sents

    def process_dataset(
        self,
        dataset: str,
        split: str,
        preprocess: bool,
        exclude_abstract: bool,
        concat_abstract: bool,
    ) -> Dataset:
        print(f"Loading {dataset} dataset ({split} split)...")
        input_df = load_dataset(HfDatasets[dataset].value, split=split).to_pandas()
        total_docs = len(input_df)

        start_time = time.time()
        output_df = input_df[["article", "summary"]].copy()

        articles, abstracts = [], []
        for article in tqdm(input_df["article"], desc="Processing", total=total_docs):
            abstract, body = self._split_abstract(article)
            abstracts.append(abstract)
            text = body if exclude_abstract else article

            if preprocess:
                text = self._preprocess_text(text)

            articles.append(text)

        # Prepare columns for each summary length
        summary_lengths = [10, 20, 30, 40]
        summaries_dict = {f"extracted_summary_{n}": [] for n in summary_lengths}

        for idx, text in enumerate(tqdm(articles, desc="Summarizing", total=total_docs)):
            ranked_sents = self.extract_and_rank(text)
            for n in summary_lengths:
                summary = " ".join(ranked_sents[:n])
                if concat_abstract:
                    summary = f"{abstracts[idx]} {summary}".strip()
                summaries_dict[f"extracted_summary_{n}"].append(summary)

        for n in summary_lengths:
            output_df[f"extracted_summary_{n}"] = summaries_dict[f"extracted_summary_{n}"]

        print(f"\nTime taken to extract: {time.time() - start_time:.2f} seconds")
        return Dataset.from_pandas(output_df)

    def save(
        self,
        data: Dataset | None,
        input_dataset: str,
        split: str,
        output_dataset: str,
        preprocess: bool,
        exclude_abstract: bool,
        concat_abstract: bool,
        output_dir_in_repo: str = "",
        data_root: str = "",
    ) -> None:
        output_dir = os.path.join(data_root, "processed", "extractive")
        os.makedirs(output_dir, exist_ok=True)

        summary_lengths = [10, 20, 30, 40]
        if data is None:
            # Try to load all summary length files, else raise error
            loaded_any = False
            for n in summary_lengths:
                filename_parts = [
                    split,
                    input_dataset,
                    "BioBERT",
                    f"{n}sent",
                ]
                if exclude_abstract:
                    filename_parts.append("exclude_abstract")
                if concat_abstract:
                    filename_parts.append("concat_abstract")
                if preprocess:
                    filename_parts.append("preprocessed")
                local_path = os.path.join(output_dir, f"{'_'.join(filename_parts)}.csv")
                if os.path.exists(local_path):
                    print(f"Loading existing processed file from: {local_path}")
                    loaded_any = True
                else:
                    print(f"File not found: {local_path}")
            if not loaded_any:
                raise ValueError("No data provided and no existing file found for any summary length")
            return

        # If data is provided, save and push for each summary length
        df = data.to_pandas()
        for n in summary_lengths:
            filename_parts = [
                split,
                input_dataset,
                "BioBERT",
                f"{n}sent",
            ]
            if exclude_abstract:
                filename_parts.append("exclude_abstract")
            if concat_abstract:
                filename_parts.append("concat_abstract")
            if preprocess:
                filename_parts.append("preprocessed")
            local_path = os.path.join(output_dir, f"{'_'.join(filename_parts)}.csv")
            # Select only the relevant columns
            cols = ["article", "summary", f"extracted_summary_{n}"]
            df_out = df[cols].copy()
            df_out.rename(columns={f"extracted_summary_{n}": "extracted_summary"}, inplace=True)
            df_out.to_csv(local_path, index=False)
            print(f"Results for {n} sentences saved locally to: {local_path}")

            # Upload using HfApi.upload_file
            api = HfApi()
            repo_file_name = os.path.basename(local_path)
            if output_dir_in_repo:
                path_in_repo = os.path.join(output_dir_in_repo, repo_file_name)
            else:
                path_in_repo = repo_file_name
            try:
                api.create_repo(repo_id=output_dataset, repo_type="dataset", exist_ok=True)
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=path_in_repo,
                    repo_id=output_dataset,
                    repo_type="dataset",
                )
                print(
                    f"Successfully uploaded to: https://huggingface.co/datasets/{output_dataset}/blob/main/{path_in_repo}"
                )
            except Exception as e:
                print(f"Failed to upload to HuggingFace Hub: {str(e)}")


def main(
    dataset: str,
    split: str,
    output_dataset: str,
    preprocess: bool,
    exclude_abstract: bool,
    concat_abstract: bool,
    output_dir_in_repo: str = "",
    data_root: str = "biolaysumm_dataset/summaries",
) -> None:
    summarizer = BioBERTSummarizer()
    save_params = {
        "concat_abstract": concat_abstract,
        "exclude_abstract": exclude_abstract,
        "input_dataset": dataset,
        "output_dataset": output_dataset,
        "preprocess": preprocess,
        "split": split,
        "output_dir_in_repo": output_dir_in_repo,
        "data_root": data_root,
    }

    try:
        summarizer.save(data=None, **save_params)
    except ValueError:
        processed_dataset = summarizer.process_dataset(
            dataset=dataset,
            split=split,
            preprocess=preprocess,
            exclude_abstract=exclude_abstract,
            concat_abstract=concat_abstract,
        )
        summarizer.save(data=processed_dataset, **save_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run extractive summarization using BioBERT embeddings."
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
        "--preprocess",
        action="store_true",
        help="Whether to preprocess input texts",
    )
    parser.add_argument(
        "--exclude-abstract",
        action="store_true",
        help="Whether to exclude the abstract in the text for ranking",
    )
    parser.add_argument(
        "--concat-abstract",
        action="store_true",
        help="Whether to prepend the abstract to the extractive summary",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        default="whopriyam2/SUWMIT-dataset",
        help="HuggingFace Hub dataset ID to upload output",
    )
    parser.add_argument(
        "--output-dir-in-repo",
        type=str,
        default="new_files",
        help="Directory within the HuggingFace repo to push the files to (optional)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="biolaysumm_dataset/summaries",
        help="Local root directory to save processed files",
    )

    args = parser.parse_args()
    main(**vars(args))
