# generate_model.py

import pandas as pd
import pickle
from ngram_model import NGramModel

data = pd.read_csv("prescription_data/prescription_clean.csv")

# ✅ Train the model
model = NGramModel(n=3)
model.train(data)

# ✅ Save the model
with open("ngram_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as ngram_model.pkl")
