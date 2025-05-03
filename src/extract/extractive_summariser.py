"""Script to perform extractive summarization"""

import os.path
import time

import pandas as pd
import spacy
import torch
import pytextrank
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

from utils import DATA_ROOT


def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


device = get_device()
print("DEVICE AVAILABLE? - ", str(device))

# Load spacy + pytextrank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# Load BioBERT sentence transformer
"""
Models to be used:
1. pritamdeka/S-PubMedBert-MS-MARCO
2. pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
3. NeuML/pubmedbert-base-embeddings
4. abhinand/MedEmbed-large-v0.1
5. Manal0809/medical-term-similarity
"""
model = SentenceTransformer(
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    device=device,
)
count = 1


# Prune article at first by getting top k sentences, then get final k sentences based on contextual embedding similarity
def extract_and_rank(text, top_k, final_k):
    global count
    print(f"EXTRACTING - {count}.....")

    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    doc = nlp(text)
    textrank_sents = [
        sent.text.strip() for sent in doc._.textrank.summary(limit_sentences=top_k)
    ]

    if not textrank_sents:
        return ""

    sentence_embeddings = model.encode(textrank_sents, convert_to_tensor=True)
    doc_embedding = model.encode(text, convert_to_tensor=True)

    similarities = (
        util.cos_sim(sentence_embeddings, doc_embedding).squeeze().cpu().numpy()
    )
    ranked_sents = [
        sent for _, sent in sorted(zip(similarities, textrank_sents), reverse=True)
    ]
    count += 1

    return " ".join(ranked_sents[:final_k])


df = load_dataset("BioLaySumm/BioLaySumm2025-eLife", split="test").to_pandas()

print(len(df))

t1 = time.time()

df["extracted_summary"] = df["article"].apply(
    lambda x: extract_and_rank(x, top_k=80, final_k=20)
)

print(f"Time taken to extract: {time.time() - t1} seconds")

file_dir = os.path.join(DATA_ROOT, "processed", "extractive")
os.makedirs(file_dir, exist_ok=True)
filename = os.path.join(file_dir, "test_elife_BioBERT.csv")
df.to_csv(filename, index=False)

print(df[["article", "extracted_summary"]].head(5))
