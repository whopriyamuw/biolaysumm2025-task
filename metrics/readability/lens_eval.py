## pip install lens-metric ##

from lens import download_model, LENS_SALSA
import torch

DEVICES = [0] if torch.cuda.is_available() else None

lens_salsa_path = download_model("davidheineman/lens-salsa")
lens_salsa = LENS_SALSA(lens_salsa_path)

# placeholder values
complex = [
    "They are culturally akin to the coastal peoples of Papua New Guinea."
]
simple = [
    "They are culturally similar to the people of Papua New Guinea."
]

scores, word_level_scores = lens_salsa.score(complex, simple, batch_size=8, devices=DEVICES)
print(scores) # [72.40909337997437]

# LENS-SALSA also returns an error-identification tagging, recover_output() will return the tagged output
tagged_output = lens_salsa.recover_output(word_level_scores, threshold=0.5)
print(tagged_output)
