"""Script to explore and analyze the BioLaySumm2025-eLife dataset."""

import os
import nltk
import pandas as pd
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from utils import DATA_ROOT

def download_nltk_resources():
    """Download required NLTK resources."""
    nltk.download("punkt")

def count_sentences(text: str) -> int:
    """Count the number of sentences in a text using NLTK's sentence tokenizer."""
    return len(sent_tokenize(str(text))) if pd.notnull(text) else 0

def analyze_dataset(df: pd.DataFrame) -> None:
    """Analyze and print basic statistics about the dataset."""
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Rows: {len(df)} | Columns: {', '.join(df.columns)}")
    
    # Print sentence count statistics
    stats = {
        "Summary": {
            "Mean": df["Summary_sentence_count"].mean(),
            "Median": df["Summary_sentence_count"].median()
        },
        "Article": {
            "Mean": df["Article_sentence_count"].mean(),
            "Median": df["Article_sentence_count"].median()
        }
    }
    
    for text_type, metrics in stats.items():
        print(f"\n{text_type} Sentences:")
        print(f"  Mean: {metrics['Mean']:.1f}")
        print(f"  Median: {metrics['Median']:.1f}")

def main():
    """Main function to load, analyze, and save the processed dataset."""
    download_nltk_resources()
    df = load_dataset("BioLaySumm/BioLaySumm2025-eLife")["train"].to_pandas()
    
    df["Summary_sentence_count"] = df["summary"].apply(count_sentences)
    df["Article_sentence_count"] = df["article"].apply(count_sentences)
    
    analyze_dataset(df)
    
    output_dir = os.path.join(DATA_ROOT, "processed", "elife")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed dataset to: {output_path}")

if __name__ == "__main__":
    main()
