from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__, template_folder="templates")  # Ensure templates folder is set
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:5000"])
# Load pre-trained models (saved as dictionaries)
with open("bigram_model.pkl", "rb") as f:
    bigram_model = pickle.load(f)

with open("trigram_model.pkl", "rb") as f:
    trigram_model = pickle.load(f)


def generate_suggestions(input_text):
    suggestions = set()
    
    words = input_text.split()
    
    if len(words) >= 2:
        trigram_key = (words[-2], words[-1])
        if trigram_key in trigram_model:
            suggestions.update(trigram_model[trigram_key])

    if len(words) >= 1:
        bigram_key = words[-1]
        if bigram_key in bigram_model:
            suggestions.update(bigram_model[bigram_key])

    return list(suggestions)[:5]  # Return top 5 suggestions


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("text", "").strip()

    if not input_text:
        return jsonify({"suggestions": []})

    suggestions = generate_suggestions(input_text)
    return jsonify({"suggestions": suggestions})


if __name__ == "__main__":
    app.run(debug=True)
