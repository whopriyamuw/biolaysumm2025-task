import pandas as pd
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# Load the dataset
df = pd.read_csv("biolaysumm_dataset/elife/train.csv")

# Add Sentence_count column using nltk sentence tokenizer
df["Summary_sentence_count"] = df["summary"].apply(
    lambda x: len(sent_tokenize(str(x))) if pd.notnull(x) else 0
)
df["Article_sentence_count"] = df["article"].apply(
    lambda x: len(sent_tokenize(str(x))) if pd.notnull(x) else 0
)
# Print first few rows and columns
print(df.head())
print(df.columns)

# Print first 2 rows with sentence count
for i in range(2):
    print(f"Row {i+1}:")
    for col in df.columns:
        print(f"{col} -----------> {df.iloc[i][col]}")
        print("-" * 150)
    print("*" * 150)

print(df.loc[:, "Summary_sentence_count"].mean())
print(df.loc[:, "Article_sentence_count"].mean())

print(df.loc[:, "Summary_sentence_count"].median())
print(df.loc[:, "Article_sentence_count"].median())

df.to_csv("biolaysumm_dataset/elife/train_processed.csv")
