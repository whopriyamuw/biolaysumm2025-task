import json
import os
import pandas as pd

from utils import REPORTS_ROOT

# define path to json files
folder_path = os.path.join(REPORTS_ROOT, "evals")
all_evals = {}

# load scores
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        dict_name = os.path.splitext(filename)[0]
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "r") as f:
            all_evals[dict_name] = json.load(f)

df = pd.DataFrame.from_dict(all_evals, orient="index")

# define reversed metrics
reverse_normalize_metrics = ["FKGL", "DCRS", "CLI"]
df_normalized = df.copy()
for col in df.columns:
    if col in reverse_normalize_metrics:
        # reversed metrics
        df_normalized[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())
    else:
        # other metrics
        df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# get category averages
df_normalized["AVGFACT"] = df_normalized[["ROUGE", "BLEU", "METEOR", "BERTScore"]].mean(
    axis=1
)
df_normalized["AVGRD"] = df_normalized[["FKGL", "DCRS", "CLI"]].mean(axis=1)
df_normalized["AVGREL"] = df_normalized[["AlignScore", "SummaC"]].mean(axis=1)
df_normalized["TOTAVG"] = df_normalized[["AVGFACT", "AVGRD", "AVGREL"]].mean(axis=1)

# sort by total average
df_normalized = df_normalized.sort_values(by="TOTAVG", ascending=False)

# define column order
column_order = [
    "ROUGE",
    "BLEU",
    "METEOR",
    "BERTScore",
    "AVGFACT",
    "FKGL",
    "DCRS",
    "CLI",
    "AVGRD",
    "AlignScore",
    "SummaC",
    "AVGREL",
    "TOTAVG",
]
df_normalized = df_normalized[column_order]

# write to file
output_file_txt = "normalized_metrics.txt"
with open(output_file_txt, "w") as f:
    f.write(df_normalized.to_string())

print(df_normalized)
