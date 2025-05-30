"""Script to normalize and analyze evaluation metrics from JSON files."""

import json
import os
from typing import Dict

import pandas as pd
from utils import REPORTS_ROOT


def load_evaluation_files(folder_path: str) -> Dict[str, Dict]:
    """Load all JSON evaluation files from the specified folder."""
    evaluations = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            model_name = os.path.splitext(filename)[0]
            filepath = os.path.join(folder_path, filename)

            with open(filepath, "r") as f:
                evaluations[model_name] = json.load(f)

    return evaluations


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize metrics in the dataframe."""
    df_normalized = df.copy()
    reverse_metrics = ["FKGL", "DCRS", "CLI"]

    for col in df.columns:
        if col in reverse_metrics:
            # For metrics where lower is better
            df_normalized[col] = (df[col].max() - df[col]) / (
                df[col].max() - df[col].min()
            )
        else:
            # For metrics where higher is better
            df_normalized[col] = (df[col] - df[col].min()) / (
                df[col].max() - df[col].min()
            )

    return df_normalized


def calculate_category_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average scores for each metric category."""
    df["relevance"] = df[["ROUGE", "BLEU", "METEOR", "BERTScore"]].mean(axis=1)
    df["readability"] = df[["FKGL", "DCRS", "CLI", "LENS"]].mean(axis=1)
    df["factuality"] = df[["AlignScore", "SummaC"]].mean(axis=1)
    df["score"] = df[["relevance", "readability", "factuality"]].mean(axis=1)

    return df


def main():
    """Main function to process and normalize evaluation metrics."""
    # Load evaluation files
    folder_path = os.path.join(REPORTS_ROOT, "evals", "elife")
    evaluations = load_evaluation_files(folder_path)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(evaluations, orient="index")

    # Normalize metrics
    df_normalized = normalize_metrics(df)

    # Calculate category averages
    df_normalized = calculate_category_averages(df_normalized)

    # Sort by total average
    df_normalized = df_normalized.sort_values(by="score", ascending=False)

    # Define and apply column order
    column_order = [
        "ROUGE",
        "BLEU",
        "METEOR",
        "BERTScore",
        "relevance",
        "FKGL",
        "DCRS",
        "CLI",
        "LENS",
        "readability",
        "AlignScore",
        "SummaC",
        "factuality",
        "score",
    ]
    df_normalized = df_normalized[column_order]

    # Save results
    output_file = "normalized_metrics.txt"
    with open(output_file, "w") as f:
        f.write(df_normalized.to_string())

    print(df_normalized)


if __name__ == "__main__":
    main()
