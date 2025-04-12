"""Script to perform extractive summarization"""
import sys
import time
import pandas as pd
import spacy
import pytextrank
import torch
from sentence_transformers import SentenceTransformer, util

device_available = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE AVAILABLE? - ", str(device_available))

# Load spacy + pytextrank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# Load BioBERT sentence transformer
"""
Models to be used:
1. microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
2. pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
"""
model = SentenceTransformer(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    device=device_available,
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


df = pd.read_csv("biolaysumm_dataset/elife/train_processed.csv")

print(len(df))

t1 = time.time()

df["extracted_summary"] = df["article"].apply(
    lambda x: extract_and_rank(x, top_k=80, final_k=20)
)

print(f"Time taken to extract: {time.time() - t1} seconds")

df.to_csv(
    "biolaysumm_dataset/summaries/extracted/train_elife_BioMedNLP.csv", index=False
)

print(df[["article", "extracted_summary"]].head(5))
