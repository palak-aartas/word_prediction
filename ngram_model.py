# ngram_model.py

from collections import defaultdict
from nltk.util import ngrams
from nltk.tokenize import WordPunctTokenizer
import os
import nltk

nltk_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
print("Using NLTK path:", nltk_path)
nltk.data.path.append(nltk_path)



tokenizer = WordPunctTokenizer()

def word_tokenize_safe(text):
    return tokenizer.tokenize(text)

class NGramModel:
    def __init__(self, n=2):
        self.n = n
        self.models = defaultdict(self.create_speciality_dict)

    def create_speciality_dict(self):
        return defaultdict(self.create_prefix_dict)

    def create_prefix_dict(self):
        return defaultdict(int)

    def train(self, data):
        for _, row in data.iterrows():
            speciality = row["speciality"]
            text = row["text"].lower()
            tokens = word_tokenize_safe(text)
            n_grams = list(ngrams(tokens, self.n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"))
            for gram in n_grams:
                prefix = gram[:-1]
                next_word = gram[-1]
                self.models[speciality][prefix][next_word] += 1

    def predict(self, speciality, prefix, top_k=3):
        prefix_tuple = tuple(word_tokenize_safe(prefix.lower()))[-(self.n - 1):]
        predictions = self.models[speciality].get(prefix_tuple, {})
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_predictions[:top_k]]
