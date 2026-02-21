import os
import pandas as pd
from transformers import pipeline
import re

DATASET_FOLDER = "csv"

all_songs = []

print("Reading files...")

for filename in os.listdir(DATASET_FOLDER):
    if filename.endswith(".csv") or filename.endswith(".xls") or filename.endswith(".xlsx"):
        filepath = os.path.join(DATASET_FOLDER, filename)

        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        print(f"Processing: {filename}")

        for _, row in df.iterrows():
            all_songs.append({
                "artist": row.get("Artist", ""),
                "song": row.get("Title", ""),
                "lyrics": row.get("Lyric", "")
            })

df_all = pd.DataFrame(all_songs)

# Remove empty lyrics
df_all = df_all.dropna(subset=["lyrics"])
df_all = df_all[df_all["lyrics"].astype(str).str.strip() != ""]

print("Total songs loaded:", len(df_all))

# Optional: limit for faster testing
df_all = df_all.head(1000)

# ✅ Load classifier (NO top_k, NO return_all_scores)
classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text[:300]

def detect_emotion(text):
    text = preprocess(text)
    result = classifier(text)
    return result[0]["label"]

print("Detecting emotions...")

df_all["emotion"] = df_all["lyrics"].apply(detect_emotion)

df_all.to_csv("songs_with_emotion.csv", index=False)

print("✅ SUCCESS! songs_with_emotion.csv created")