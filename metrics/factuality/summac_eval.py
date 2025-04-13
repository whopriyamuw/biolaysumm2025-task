"""
Install metric library using 'pip install summac'
"""
from summac.model_summac import SummaCZS

model = SummaCZS(granularity="sentence", model_name="vitc", device="cpu")

score = model.score(["This is a test"], ["A test is being run"])["scores"][0]

print(f"Score is - {score}")

"""
# Script to apply the score on a dataset

import pandas as pd

data = pd.read_csv("data.csv")

# Apply scorer to each row
df['score'] = df.apply(
    lambda row: model.score(
        [row['system summary']],
        [row['gold summary']],
    )["scores"][0],
    axis=1
)

# Compute the average score
average_score = df['score'].mean()

print("Average Score:", average_score)

"""
