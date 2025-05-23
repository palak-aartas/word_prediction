from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os
import nltk

nltk_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
print("Using NLTK path:", nltk_path)
nltk.data.path.append(nltk_path)

app = Flask(__name__, template_folder="templates")
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:5000"])

with open("ngram_model.pkl", "rb") as f:
    ngram_model = pickle.load(f)

with open("bigram_model.pkl", "rb") as f:
    bigram_model = pickle.load(f)

with open("trigram_model.pkl", "rb") as f:
    trigram_model = pickle.load(f)

SPECIALITIES = sorted([str(k) for k in ngram_model.models.keys()])

def generate_suggestions(speciality, input_text):
    suggestions = set()

    new_model_suggestions = ngram_model.predict(speciality, input_text, top_k=5)
    if new_model_suggestions:
        return new_model_suggestions

    words = input_text.lower().split()

    if len(words) >= 2:
        trigram_key = (words[-2], words[-1])
        if trigram_key in trigram_model:
            suggestions.update(trigram_model[trigram_key])

    if len(words) >= 1:
        bigram_key = words[-1]
        if bigram_key in bigram_model:
            suggestions.update(bigram_model[bigram_key])

    return list(suggestions)[:5]

@app.route("/")
def home():
    return render_template("index.html", specialities=SPECIALITIES)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("text", "").strip()
    speciality = data.get("speciality", "").strip()

    if not input_text or not speciality:
        return jsonify({"suggestions": []})

    suggestions = generate_suggestions(speciality, input_text)
    return jsonify({"suggestions": suggestions})

if __name__ == "__main__":
    app.run(debug=True)
