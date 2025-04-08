"""
1. Install metric library using 'pip install git+https://github.com/yuh-zha/AlignScore'
2. Download model checkpoint - https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt
"""

from alignscore import AlignScore

model_ckpt = "AlignScore-base.ckpt"

scorer = AlignScore(
    model="roberta-base",
    batch_size=32,
    device="cpu",
    ckpt_path=model_ckpt,
    evaluation_mode="nli_sp",
)
score = scorer.score(contexts=["This is a test"], claims=["A test is being run"])

print(f"Score is - {score}")

"""
# Script to apply the score on a dataset

import pandas as pd

data = pd.read_csv("data.csv")

# Apply scorer to each row
df['score'] = df.apply(
    lambda row: scorer.score(
        contexts=[row['system summary']],
        claims=[row['gold summary']],
    )['score'],
    axis=1
)

# Compute the average score
average_score = df['score'].mean()

print("Average Score:", average_score)

"""
