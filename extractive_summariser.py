import pandas as pd
import spacy
import pytextrank
from sentence_transformers import SentenceTransformer, util

# Load spacy + pytextrank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# Load BioBERT sentence transformer
model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
count = 1

def extract_and_rank(text, top_k, final_k):
    global count
    print (f"EXTRACTING - {count}.....")
    
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    doc = nlp(text)
    textrank_sents = [sent.text.strip() for sent in doc._.textrank.summary(limit_sentences=top_k)]

    if not textrank_sents:
        return ""

    sentence_embeddings = model.encode(textrank_sents, convert_to_tensor=True)
    doc_embedding = model.encode(text, convert_to_tensor=True)

    similarities = util.cos_sim(sentence_embeddings, doc_embedding).squeeze().cpu().numpy()
    ranked_sents = [sent for _, sent in sorted(zip(similarities, textrank_sents), reverse=True)]
    count += 1

    return " ".join(ranked_sents[:final_k])

# === Load Data ===
df = pd.read_csv("biolaysumm_dataset/elife/train_processed.csv")

print (len(df))

# Apply extraction system to the 'summary' column
df['extracted_summary'] = df['article'].apply(lambda x: extract_and_rank(x, top_k=80, final_k=20))

# Save updated DataFrame (optional)
df.to_csv("biolaysumm_dataset/elife/train_extracted.csv", index=False)

# Preview
print(df[['article', 'extracted_summary']].head())
