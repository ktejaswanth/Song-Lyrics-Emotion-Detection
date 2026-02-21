from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import pandas as pd
import re
import random

app = Flask(__name__)
CORS(app)

# ==============================
# Load Emotion Model
# ==============================
classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)

# ==============================
# Load Dataset
# ==============================
songs_df = pd.read_csv("songs_with_emotion.csv")

# ==============================
# Text Preprocessing
# ==============================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text[:300]

# ==============================
# Detect Emotion
# ==============================
def detect_emotion(text):
    cleaned = preprocess_text(text)
    result = classifier(cleaned)
    return result[0]["label"]

# ==============================
# Recommend Songs
# ==============================
def recommend_songs(emotion):
    matching = songs_df[songs_df["emotion"] == emotion]

    if matching.empty:
        return []

    sample = matching.sample(min(5, len(matching)))

    recommendations = []
    for _, row in sample.iterrows():
        recommendations.append({
            "artist": row["artist"],
            "song": row["song"]
        })

    return recommendations

# ==============================
# MAIN API ROUTE
# ==============================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        lyrics = data.get("lyrics", "")

        if not lyrics:
            return jsonify({"error": "No lyrics provided"}), 400

        emotion = detect_emotion(lyrics)
        recommendations = recommend_songs(emotion)

        return jsonify({
            "emotion": emotion,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    app.run(debug=True, port=5000)