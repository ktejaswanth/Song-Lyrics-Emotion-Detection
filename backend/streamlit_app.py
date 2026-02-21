"""
🎵 Song Lyrics Emotion Detection
A Data-Driven Approach to Understanding Human Emotions
Built with Streamlit, HuggingFace Transformers, and Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import requests
import os
import re
import io
import time
import glob
import speech_recognition as sr
import streamlit.components.v1 as components

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🎵 Song Lyrics Emotion Detection",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    /* ========================================= */
    /* 1. GLOBAL VARIABLES & BASE STYLES         */
    /* ========================================= */
    .main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ========================================= */
    /* 2. LAYOUT COMPONENTS                      */
    /* ========================================= */
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1.1rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ========================================= */
    /* 3. METRICS & BADGES                       */
    /* ========================================= */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-top: auto;
    }

    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin: 0.5rem;
    }
    
    .emotion-joy { background: linear-gradient(135deg, #f6d365, #fda085); color: #333; }
    .emotion-sadness { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    .emotion-anger { background: linear-gradient(135deg, #f5576c, #ff6b6b); color: white; }
    .emotion-fear { background: linear-gradient(135deg, #4facfe, #00f2fe); color: #333; }
    .emotion-surprise { background: linear-gradient(135deg, #43e97b, #38f9d7); color: #333; }
    .emotion-love { background: linear-gradient(135deg, #f093fb, #f5576c); color: white; }

    /* ========================================= */
    /* 4. STREAMLIT ELEMENTS OVERRIDES           */
    /* ========================================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
    }
    
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        flex-wrap: wrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #a0aec0;
        padding: 10px 20px;
        white-space: nowrap;
        margin-bottom: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* ========================================= */
    /* 5. RESPONSIVE MEDIA QUERIES               */
    /* ========================================= */
    @media screen and (max-width: 768px) {
        .main-header {
            padding: 1.5rem 1rem;
        }
        .main-header h1 {
            font-size: 1.8rem;
        }
        .main-header p {
            font-size: 0.95rem;
        }
        .metric-value {
            font-size: 1.5rem;
        }
        .glass-card {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .emotion-badge {
            font-size: 1rem;
            padding: 0.4rem 1.2rem;
        }
        .stButton > button {
            padding: 0.6rem 1rem !important;
            font-size: 0.9rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 10px;
            font-size: 0.8rem;
        }
        div[data-testid="column"] {
            min-width: 100% !important;
            margin-bottom: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PROGRESSIVE WEB APP (PWA) REGISTRATION
# ============================================================
pwa_script = '''
<script>
    if (window.parent.navigator.serviceWorker) {
        const manifest = {
            "name": "🎵 Song Lyrics Emotion Detection",
            "short_name": "MoodMate AI",
            "description": "Analyze the emotional fingerprint of any song.",
            "start_url": "./",
            "display": "standalone",
            "background_color": "#0f0c29",
            "theme_color": "#667eea",
            "icons": [
                {
                    "src": "https://cdn-icons-png.flaticon.com/512/3048/3048122.png",
                    "sizes": "192x192",
                    "type": "image/png"
                },
                {
                    "src": "https://cdn-icons-png.flaticon.com/512/3048/3048122.png",
                    "sizes": "512x512",
                    "type": "image/png"
                }
            ]
        };
        const manifestBlob = new Blob([JSON.stringify(manifest)], {type: 'application/json'});
        const manifestUrl = URL.createObjectURL(manifestBlob);
        
        let link = window.parent.document.querySelector('link[rel="manifest"]');
        if (!link) {
            link = window.parent.document.createElement('link');
            link.rel = 'manifest';
            window.parent.document.head.appendChild(link);
        }
        link.href = manifestUrl;

        const sw = `
            self.addEventListener('install', e => {
                console.log('PWA Service Worker Install');
            });
            self.addEventListener('fetch', e => {
                const url = new URL(e.request.url);
                if (url.pathname.includes('/_stcore/')) {
                    // Do not cache websockets or live Streamlit data
                    return fetch(e.request);
                }
                
                // Network-first strategy for frontend assets
                e.respondWith(
                    fetch(e.request).catch(function() {
                        return caches.match(e.request);
                    })
                );
            });
        `;
        const blob = new Blob([sw], {type: 'application/javascript'});
        const url = URL.createObjectURL(blob);
        
        window.parent.navigator.serviceWorker.register(url)
            .then(function(r) { console.log('PWA SW registered!'); })
            .catch(function(e) { console.log('SW error!', e); });
    }
</script>
'''
components.html(pwa_script, height=0, width=0)



# ============================================================
# CONSTANTS
# ============================================================
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲"
}

EMOTION_COLORS = {
    "sadness": "#667eea",
    "joy": "#f6d365",
    "love": "#f093fb",
    "anger": "#f5576c",
    "fear": "#4facfe",
    "surprise": "#43e97b"
}

EMOTION_GRADIENT_COLORS = {
    "sadness": ["#667eea", "#764ba2"],
    "joy": ["#f6d365", "#fda085"],
    "love": ["#f093fb", "#f5576c"],
    "anger": ["#f5576c", "#ff6b6b"],
    "fear": ["#4facfe", "#00f2fe"],
    "surprise": ["#43e97b", "#38f9d7"]
}

# ============================================================
# LOAD MODEL (CACHED)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the DistilBERT emotion classification model."""
    classifier = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        top_k=None
    )
    return classifier

# ============================================================
# ML MODEL TRAINING (CACHED)
# ============================================================
ML_MODEL_INFO = {
    "Logistic Regression": {
        "icon": "📐",
        "desc": "Linear model using log-odds for multi-class classification",
        "color": "#667eea"
    },
    "Random Forest": {
        "icon": "🌲",
        "desc": "Ensemble of decision trees with bagging",
        "color": "#43e97b"
    },
    "SVM (LinearSVC)": {
        "icon": "📊",
        "desc": "Support Vector Machine with linear kernel",
        "color": "#f5576c"
    },
    "Naive Bayes": {
        "icon": "📈",
        "desc": "Probabilistic classifier based on Bayes' theorem",
        "color": "#f6d365"
    }
}

@st.cache_resource(show_spinner=False)
def train_ml_models():
    """Train ML models on labeled data, then use fast ML to label all CSV songs."""
    
    # ── PHASE 1: Train on already-labeled songs_with_emotion.csv ──
    labeled_df = pd.read_csv("songs_with_emotion.csv")
    labeled_df["cleaned"] = labeled_df["lyrics"].apply(
        lambda x: re.sub(r"[^\w\s']", " ", str(x).lower()).strip()[:512]
    )
    labeled_df = labeled_df[labeled_df["cleaned"].str.len() > 10]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(labeled_df["emotion"])
    
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english")
    X_tfidf = tfidf.fit_transform(labeled_df["cleaned"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "SVM (LinearSVC)": LinearSVC(max_iter=2000, random_state=42, C=1.0),
        "Naive Bayes": MultinomialNB(alpha=1.0)
    }
    
    trained_models = {}
    metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        
        trained_models[name] = model
        metrics[name] = {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
            "y_test": y_test,
            "y_pred": y_pred
        }
    
    # ── PHASE 2: Load ALL CSVs & label with fast Logistic Regression ──
    all_song_rows = []
    
    # Add already-labeled songs
    for _, row in labeled_df.iterrows():
        all_song_rows.append({
            "Artist": row.get("artist", "Unknown"),
            "Title": row.get("song", "Unknown"),
            "Lyric": str(row.get("lyrics", "")),
            "emotion": row["emotion"],
            "cleaned": row["cleaned"]
        })
    
    # Load all artist CSVs and label with fast ML
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
    lr_model = trained_models["Logistic Regression"]
    
    if os.path.isdir(csv_dir):
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                lyric_col = next((c for c in df.columns if c.lower() in ["lyric", "lyrics"]), None)
                if lyric_col is None:
                    continue
                artist_col = next((c for c in df.columns if c.lower() == "artist"), None)
                title_col = next((c for c in df.columns if c.lower() == "title"), None)
                
                # Batch process entire CSV at once for speed
                lyrics_list = df[lyric_col].dropna().astype(str).tolist()
                cleaned_list = [re.sub(r"[^\w\s']", " ", t.lower()).strip()[:512] for t in lyrics_list]
                valid_mask = [len(c) > 10 for c in cleaned_list]
                valid_cleaned = [c for c, v in zip(cleaned_list, valid_mask) if v]
                valid_rows = [r for (_, r), v in zip(df.iterrows(), valid_mask) if v]
                
                if valid_cleaned:
                    X_vec = tfidf.transform(valid_cleaned)
                    preds = lr_model.predict(X_vec)
                    pred_labels = le.inverse_transform(preds)
                    
                    for row, cleaned, emotion in zip(valid_rows, valid_cleaned, pred_labels):
                        all_song_rows.append({
                            "Artist": str(row[artist_col]) if artist_col and pd.notna(row.get(artist_col)) else os.path.basename(csv_file).replace(".csv", ""),
                            "Title": str(row[title_col]) if title_col and pd.notna(row.get(title_col)) else "Unknown",
                            "Lyric": str(row[lyric_col]),
                            "emotion": emotion,
                            "cleaned": cleaned
                        })
            except Exception:
                continue
    
    song_db = pd.DataFrame(all_song_rows)
    
    dataset_info = {
        "total_songs": len(song_db),
        "emotion_distribution": song_db["emotion"].value_counts().to_dict(),
        "artists": song_db["Artist"].nunique()
    }
    
    return trained_models, metrics, tfidf, le, X_test, y_test, song_db, dataset_info

def ml_predict(text, model, tfidf, le):
    """Predict emotion using a trained ML model."""
    cleaned = re.sub(r"[^\w\s']", " ", str(text).lower()).strip()[:512]
    if not cleaned.strip():
        return "unknown", {}
    X = tfidf.transform([cleaned])
    pred = model.predict(X)[0]
    label = le.inverse_transform([pred])[0]
    
    # Get probability scores if available
    probas = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        for idx, cls in enumerate(le.classes_):
            probas[cls] = round(float(proba[idx]) * 100, 2)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)[0]
        exp_d = np.exp(decision - np.max(decision))
        softmax = exp_d / exp_d.sum()
        for idx, cls in enumerate(le.classes_):
            probas[cls] = round(float(softmax[idx]) * 100, 2)
    else:
        for cls in le.classes_:
            probas[cls] = 100.0 if cls == label else 0.0
    
    return label, probas

# ============================================================
# YOUTUBE API INTEGRATION
# ============================================================
YOUTUBE_API_KEY = "AIzaSyCZ6QLP_f8vw66d-0WEy6MB2-7NRySXf4M"

def search_youtube(query, max_results=5):
    """Search YouTube for songs using the YouTube Data API v3."""
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "videoCategoryId": "10",  # Music category
            "maxResults": max_results,
            "key": YOUTUBE_API_KEY,
            "order": "relevance"
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data.get("items", []):
                video_id = item["id"].get("videoId", "")
                snippet = item["snippet"]
                results.append({
                    "title": snippet.get("title", "Unknown"),
                    "channel": snippet.get("channelTitle", "Unknown"),
                    "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                })
            return results
        else:
            return []
    except Exception:
        return []

def get_emotion_song_query(emotion):
    """Generate a YouTube search query based on emotion."""
    queries = {
        "joy": "happy upbeat feel good songs",
        "sadness": "sad emotional songs",
        "anger": "angry intense rock songs",
        "love": "romantic love songs",
        "fear": "dark atmospheric emotional songs",
        "surprise": "unexpected plot twist surprise songs"
    }
    return queries.get(emotion, f"{emotion} mood songs")

def search_songs_in_db(song_db, query, max_results=20):
    """Search for songs in the local database by title or artist."""
    query_lower = query.lower().strip()
    if not query_lower:
        return pd.DataFrame()
    
    mask = (
        song_db["Title"].str.lower().str.contains(query_lower, na=False) |
        song_db["Artist"].str.lower().str.contains(query_lower, na=False)
    )
    results = song_db[mask].head(max_results).copy()
    return results

# ============================================================
# HELPER FUNCTIONS
# ============================================================

# Filler words/sounds commonly found in lyrics that don't carry emotion
FILLER_PATTERNS = [
    r'\b(oh+|ooh+|ah+|uh+|eh+|mm+|hmm+|yeah+|ya+|hey+|whoa+|wo+ah)\b',
    r'\b(la la|na na|da da|sha la|do do|ba ba|tra la)\b',
    r'\b(verse|chorus|bridge|outro|intro|hook|repeat|x\d+|\d+x)\b',
    r'\[.*?\]',          # Remove [Verse 1], [Chorus], etc.
    r'\(.*?\)',           # Remove (repeat), (x2), etc.
]

def preprocess_text(text):
    """Clean lyrics text for model input — removes fillers and noise."""
    text = str(text).lower()
    # Remove filler patterns
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    # Remove special characters but keep apostrophes
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def deduplicate_lines(text):
    """Remove repeated/duplicate lines from lyrics (choruses, hooks)."""
    lines = text.split("\n")
    seen = set()
    unique_lines = []
    for line in lines:
        cleaned_line = line.strip().lower()
        if cleaned_line and len(cleaned_line) > 5 and cleaned_line not in seen:
            seen.add(cleaned_line)
            unique_lines.append(line.strip())
    return " ".join(unique_lines)

def split_into_chunks(text, max_chunk_size=450):
    """Split text into meaningful chunks that fit the model's token limit."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0
    
    for word in words:
        if current_len + len(word) + 1 > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(word)
        current_len += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [text[:max_chunk_size]]

def predict_emotions(classifier, text):
    """Predict emotion probabilities using multi-chunk analysis for accuracy."""
    # Step 1: Deduplicate lines (remove repeated choruses)
    deduped = deduplicate_lines(text)
    
    # Step 2: Clean the text
    cleaned = preprocess_text(deduped)
    if not cleaned.strip() or len(cleaned) < 5:
        return {e: 0.0 for e in EMOTION_EMOJIS}
    
    # Step 3: Split into chunks for long lyrics
    chunks = split_into_chunks(cleaned, max_chunk_size=450)
    
    # Step 4: Analyze each chunk
    all_scores = {e: [] for e in EMOTION_EMOJIS}
    
    for chunk in chunks[:5]:  # Max 5 chunks to keep it fast
        if len(chunk.strip()) < 10:
            continue
        try:
            results = classifier(chunk, truncation=True, max_length=512)[0]
            for item in results:
                label = item["label"].lower()
                if label in all_scores:
                    all_scores[label].append(item["score"] * 100)
        except Exception:
            continue
    
    # Step 5: Weighted average — give more weight to higher-scoring chunks
    emotions = {}
    for emotion, scores in all_scores.items():
        if scores:
            # Weighted: higher scores contribute more
            weights = [s ** 1.5 for s in scores]
            total_weight = sum(weights)
            if total_weight > 0:
                emotions[emotion] = round(sum(s * w for s, w in zip(scores, weights)) / total_weight, 2)
            else:
                emotions[emotion] = round(np.mean(scores), 2)
        else:
            emotions[emotion] = 0.0
    
    return emotions

def get_dominant_emotion(emotions):
    """Get the dominant emotion from predictions."""
    return max(emotions, key=emotions.get)

def compute_intensity_score(emotions):
    """Compute emotion intensity score (how concentrated the emotion is)."""
    values = list(emotions.values())
    max_val = max(values)
    mean_val = np.mean(values)
    intensity = (max_val - mean_val) / max_val if max_val > 0 else 0
    return round(intensity * 100, 1)

def transcribe_audio(audio_bytes):
    """Transcribe audio bytes to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_bytes) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def predict_emotion_line_by_line(text, model, tfidf, le):
    """Analyze emotions line-by-line using the fast ML model for the Journey Map & Line Highlighter."""
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
    if not lines:
        return []
    
    line_results = []
    for idx, line in enumerate(lines):
        label, probas = ml_predict(line, model, tfidf, le)
        line_results.append({
            "line_num": idx + 1,
            "text": line,
            "emotion": label,
            "score": probas.get(label, 0)
        })
    return line_results

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def create_journey_map_chart(line_results):
    """Create a timeline chart showing emotion journey throughout the song."""
    if not line_results:
        return go.Figure()

    df = pd.DataFrame(line_results)
    
    # Map emotions to a numeric value for y-axis so they spread out
    emotion_order = {"joy": 6, "surprise": 5, "love": 4, "fear": 3, "sadness": 2, "anger": 1}
    df["y_val"] = df["emotion"].map(emotion_order)
    
    # Generate colors
    colors = [EMOTION_COLORS.get(e, "#ffffff") for e in df["emotion"]]
    emojis = [EMOTION_EMOJIS.get(e, "🎭") for e in df["emotion"]]
    
    fig = go.Figure()

    # Add line tracing the journey
    fig.add_trace(go.Scatter(
        x=df["line_num"],
        y=df["y_val"],
        mode="lines+markers+text",
        line=dict(color="rgba(255,255,255,0.2)", width=2),
        marker=dict(
            color=colors,
            size=14,
            line=dict(color="white", width=1.5)
        ),
        text=emojis,
        textposition="top center",
        customdata=df[["text", "emotion"]],
        hovertemplate="<b>Line %{x}</b><br><i>%{customdata[0]}</i><br>Emotion: %{customdata[1]}<extra></extra>"
    ))
    
    # Customize axis
    fig.update_layout(
        title=dict(text="🎢 Emotion Journey Map", font=dict(color="white", size=18)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Song Timeline (Line #)",
            title_font=dict(color="#a0aec0"),
            tickfont=dict(color="white"),
            showgrid=False
        ),
        yaxis=dict(
            title="Emotion",
            title_font=dict(color="#a0aec0"),
            tickvals=list(emotion_order.values()),
            ticktext=[e.capitalize() for e in emotion_order.keys()],
            tickfont=dict(color="white"),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            range=[0.5, 6.5]
        ),
        margin=dict(t=50, b=20, l=60, r=20),
        height=400,
        showlegend=False
    )
    return fig

def create_bar_chart(emotions, title="Emotion Distribution"):
    """Create a beautiful bar chart for emotion distribution."""
    labels = [f"{EMOTION_EMOJIS.get(e, '')} {e.capitalize()}" for e in emotions.keys()]
    values = list(emotions.values())
    colors = [EMOTION_COLORS.get(e, "#667eea") for e in emotions.keys()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.3)", width=1),
                opacity=0.9
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="white", size=13, family="Arial Black"),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}%<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=18)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickfont=dict(color="white", size=12),
            showgrid=False
        ),
        yaxis=dict(
            title="Probability (%)",
            title_font=dict(color="#a0aec0"),
            tickfont=dict(color="#a0aec0"),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            range=[0, max(values) * 1.3] if values else [0, 100]
        ),
        margin=dict(t=50, b=20, l=40, r=20),
        height=400
    )
    return fig

def create_radar_chart(emotions, title="Emotion Radar"):
    """Create a radar/spider chart for emotion distribution."""
    categories = [f"{EMOTION_EMOJIS.get(e, '')} {e.capitalize()}" for e in emotions.keys()]
    values = list(emotions.values())
    
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(102, 126, 234, 0.3)",
        line=dict(color="#667eea", width=2),
        marker=dict(size=8, color="#764ba2"),
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2] if values else [0, 100],
                showticklabels=True,
                tickfont=dict(color="#a0aec0", size=10),
                gridcolor="rgba(255,255,255,0.1)"
            ),
            angularaxis=dict(
                tickfont=dict(color="white", size=12),
                gridcolor="rgba(255,255,255,0.1)"
            )
        ),
        title=dict(text=title, font=dict(color="white", size=18)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=30, l=60, r=60),
        height=420,
        showlegend=False
    )
    return fig

def create_comparison_chart(songs_data):
    """Create a grouped bar chart to compare emotions across songs."""
    fig = go.Figure()
    
    emotion_list = list(EMOTION_EMOJIS.keys())
    
    for emotion in emotion_list:
        emoji = EMOTION_EMOJIS[emotion]
        values = [song["emotions"].get(emotion, 0) for song in songs_data]
        labels = [song["name"] for song in songs_data]
        
        fig.add_trace(go.Bar(
            name=f"{emoji} {emotion.capitalize()}",
            x=labels,
            y=values,
            marker_color=EMOTION_COLORS[emotion],
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(size=9, color="white")
        ))
    
    fig.update_layout(
        barmode="group",
        title=dict(text="🎵 Song Emotion Comparison", font=dict(color="white", size=18)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="white", size=11), showgrid=False),
        yaxis=dict(
            title="Probability (%)",
            title_font=dict(color="#a0aec0"),
            tickfont=dict(color="#a0aec0"),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)"
        ),
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)"
        ),
        margin=dict(t=50, b=20, l=40, r=20),
        height=500
    )
    return fig

def create_emotion_ranking_chart(emotions):
    """Create a horizontal bar chart showing emotions ranked."""
    sorted_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
    labels = [f"{EMOTION_EMOJIS.get(e, '')} {e.capitalize()}" for e in sorted_emotions.keys()]
    values = list(sorted_emotions.values())
    colors = [EMOTION_COLORS.get(e, "#667eea") for e in sorted_emotions.keys()]
    
    fig = go.Figure(data=[
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=colors, opacity=0.9),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="white", size=12),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.2f}%<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=dict(text="🏆 Emotion Ranking", font=dict(color="white", size=18)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Probability (%)",
            title_font=dict(color="#a0aec0"),
            tickfont=dict(color="#a0aec0"),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            range=[0, max(values) * 1.3] if values else [0, 100]
        ),
        yaxis=dict(tickfont=dict(color="white", size=13), autorange="reversed"),
        margin=dict(t=50, b=20, l=100, r=40),
        height=350
    )
    return fig

def create_dataset_heatmap(df, emotion_cols):
    """Create a heatmap of emotions across songs in dataset."""
    # Use first N songs for readability
    display_df = df.head(20)
    
    # Build a label for each row
    if "Title" in display_df.columns:
        labels = display_df["Title"].astype(str).tolist()
    elif "song" in display_df.columns:
        labels = display_df["song"].astype(str).tolist()
    else:
        labels = [f"Song {i+1}" for i in range(len(display_df))]
    
    # Truncate long labels
    labels = [l[:30] + "..." if len(l) > 30 else l for l in labels]
    
    z_data = display_df[emotion_cols].values
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[col.replace(" %", "") for col in emotion_cols],
        y=labels,
        colorscale=[
            [0, "#0f0c29"],
            [0.25, "#302b63"],
            [0.5, "#667eea"],
            [0.75, "#764ba2"],
            [1, "#f093fb"]
        ],
        text=[[f"{val:.1f}%" for val in row] for row in z_data],
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text="🎭 Emotion Heatmap (Top 20 Songs)", font=dict(color="white", size=18)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="white", size=12)),
        yaxis=dict(tickfont=dict(color="white", size=10), autorange="reversed"),
        margin=dict(t=50, b=20, l=150, r=20),
        height=max(400, len(labels) * 30)
    )
    return fig

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎵 Song Lyrics Emotion Detection</h1>
    <p>A Data-Driven Approach to Understanding Human Emotions</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("Emotion Intelligence Dashboard")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    mode = st.radio(
        "🔄 Select Mode",
        [
            "🎤 Single Song Analysis", 
            "📊 CSV Bulk Analysis", 
            "🔀 Song Comparison", 
            "🤖 ML Model Comparison", 
            "🔍 Song Search",
            "👤 Artist Emotion DNA",
            "🎵 Mood Playlist Generator"
        ],
        index=0
    )
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### 📖 How to Use")
    if "Single" in mode:
        st.info("1. Enter lyrics (text/voice)\n2. Analyze Emotion\n3. View emotion breakdown charts\n4. See line-by-line Highlighter & Journey Map")
    elif "CSV" in mode:
        st.info("1. Upload a CSV with a **lyrics** or **Lyric** column\n2. Wait for batch processing\n3. Download results with emotion scores")
    elif "ML" in mode:
        st.info("1. View trained ML model metrics\n2. Compare Accuracy, F1, Precision, Recall\n3. Test lyrics across all models")
    elif "Search" in mode:
        st.info("1. Search songs by title or artist\n2. View detected emotion for each song\n3. Get YouTube suggestions")
    elif "Artist Emotion DNA" in mode:
        st.info("1. Select an artist from the dataset\n2. View their overall emotional fingerprint across all their songs")
    elif "Playlist Simulator" in mode or "Mood Playlist Generator" in mode:
        st.info("1. Pick a mood\n2. Get a curated AI playlist from the database + YouTube videos")
    else:
        st.info("1. Enter lyrics for 2+ songs\n2. Compare their emotional profiles\n3. View side-by-side analysis")
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### 🎭 Emotion Labels")
    for emotion, emoji in EMOTION_EMOJIS.items():
        st.markdown(f"{emoji} **{emotion.capitalize()}**")
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#718096; font-size:0.8rem;'>"
        "Built with ❤️ using Streamlit & Transformers</p>",
        unsafe_allow_html=True
    )

# ============================================================
# LOAD MODEL
# ============================================================
with st.spinner("🧠 Loading AI Model... (first time may take a minute)"):
    classifier = load_model()

with st.spinner("⚙️ Training ML Models on ALL CSV datasets... (first time may take a few minutes)"):
    trained_models, ml_metrics, tfidf_vectorizer, label_encoder, X_test_data, y_test_data, song_database, dataset_info = train_ml_models()

# ============================================================
# MODE 1: SINGLE SONG ANALYSIS
# ============================================================
if "Single" in mode:
    st.markdown("## 🎤 Single Song Analysis")
    st.markdown("Paste song lyrics below to detect the emotional fingerprint.")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    input_method = st.radio("Choose Input Method", ["📝 Text", "🎤 Voice (Sing/Speak)"], horizontal=True)
    
    lyrics_input = ""
    if input_method == "📝 Text":
        lyrics_input = st.text_area(
            "📝 Enter Song Lyrics",
            height=200,
            placeholder="Paste your song lyrics here...\n\nExample: 'I'm walking on sunshine, whoa...'",
            key="single_lyrics"
        )
    else:
        st.info("Record yourself singing or speaking the lyrics. The AI will transcribe and analyze it!")
        audio_val = st.audio_input("Record Audio")
        if audio_val:
            with st.spinner("🎙️ Transcribing audio..."):
                transcribed = transcribe_audio(audio_val)
                if transcribed:
                    st.success("📝 Transcribed text:")
                    lyrics_input = st.text_area("Edit transcribed text if needed:", value=transcribed, height=150)
                else:
                    st.error("❌ Could not transcribe audio. Please try again or use text input.")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        analyze_btn = st.button("🔍 Analyze Emotion", use_container_width=True)
    
    if analyze_btn and lyrics_input.strip():
        with st.spinner("🎯 Analyzing emotions..."):
            progress = st.progress(0.0)
            for i in range(50):
                time.sleep(0.01)
                progress.progress((i + 1) / 100.0)
            
            emotions = predict_emotions(classifier, lyrics_input)
            
            for i in range(50, 100):
                time.sleep(0.01)
                progress.progress((i + 1) / 100.0)
            progress.empty()
        
        dominant = get_dominant_emotion(emotions)
        intensity = compute_intensity_score(emotions)
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # ---- Dominant Emotion Display ----
        st.markdown("### 🎯 Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2.5rem;">{EMOTION_EMOJIS.get(dominant, '🎭')}</div>
                <div class="metric-value">{dominant.upper()}</div>
                <div class="metric-label">Dominant Emotion</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{emotions[dominant]:.1f}%</div>
                <div class="metric-label">Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{intensity}%</div>
                <div class="metric-label">Emotion Intensity</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # ---- Emotion Percentages ----
        st.markdown("### 📊 Emotion Breakdown")
        
        emotion_cols = st.columns(6)
        sorted_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
        for idx, (emotion, score) in enumerate(sorted_emotions.items()):
            with emotion_cols[idx]:
                emoji = EMOTION_EMOJIS.get(emotion, "🎭")
                is_dominant = emotion == dominant
                border_color = EMOTION_COLORS.get(emotion, "#667eea")
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.05);
                    border-radius: 12px;
                    padding: 1rem;
                    text-align: center;
                    border: 2px solid {'#f093fb' if is_dominant else 'rgba(255,255,255,0.1)'};
                    {'box-shadow: 0 0 20px rgba(240,147,251,0.3);' if is_dominant else ''}
                ">
                    <div style="font-size: 1.8rem;">{emoji}</div>
                    <div style="color: white; font-weight: 600; font-size: 0.85rem;">{emotion.capitalize()}</div>
                    <div style="color: {border_color}; font-size: 1.4rem; font-weight: 800;">{score:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # ---- 🎨 Color Palette Generation ----
        st.markdown("### 🎨 Emotion Color Palette")
        st.markdown("A unique aesthetic color gradient based on the song's emotional makeup:")
        palette_colors = []
        for em, sc in sorted_emotions.items():
            if sc > 5.0:  # Include emotions with > 5% score
                palette_colors.append(EMOTION_COLORS.get(em, "#ffffff"))
        if not palette_colors:
            palette_colors = ["#cccccc", "#999999"]
        
        # Render a CSS gradient block
        gradient_css = f"linear-gradient(90deg, {', '.join(palette_colors)})"
        st.markdown(f"""
        <div style="
            width: 100%;
            height: 60px;
            border-radius: 12px;
            background: {gradient_css};
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            margin-bottom: 1rem;
        "></div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # ---- Charts & Line Analysis ----
        st.markdown("### 📈 Visualizations & Journey")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Bar Chart", "🕸️ Radar Chart", "🏆 Ranking", "🎢 Journey Map", "📝 Line Highlighter"])
        
        with tab1:
            fig_bar = create_bar_chart(emotions, "Emotion Probability Distribution")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            fig_radar = create_radar_chart(emotions, "Emotion Radar Profile")
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with tab3:
            fig_rank = create_emotion_ranking_chart(emotions)
            st.plotly_chart(fig_rank, use_container_width=True)
            
        # Get line-by-line predictions using the fast LR model
        lr_model = trained_models["Logistic Regression"]
        line_results = predict_emotion_line_by_line(lyrics_input, lr_model, tfidf_vectorizer, label_encoder)
        
        with tab4:
            st.markdown("Analyze how the emotion fluctuates line by line through 's journey.")
            fig_journey = create_journey_map_chart(line_results)
            st.plotly_chart(fig_journey, use_container_width=True)
            
        with tab5:
            st.markdown("#### Lyrics Line Highlighter")
            st.markdown("Every line is color-coded by its dominant emotion.")
            highlight_html = "<div style='background: rgba(0,0,0,0.2); padding: 20px; border-radius: 12px; line-height: 2.0; font-size: 1.1rem;'>"
            for res in line_results:
                line_color = EMOTION_COLORS.get(res["emotion"], "#ffffff")
                emoji = EMOTION_EMOJIS.get(res["emotion"], "🎭")
                # Create a slight background highlight
                highlight_html += f"""
                <span style="background-color: {line_color}33; border-left: 4px solid {line_color}; padding: 4px 10px; margin-bottom: 6px; display: inline-block; border-radius: 0 4px 4px 0; width: 100%;">
                    <span style="font-size: 0.9em; margin-right: 8px;" title="{res['emotion'].capitalize()}">{emoji}</span>
                    {res["text"]}
                </span>
                """
            highlight_html += "</div>"
            st.markdown(highlight_html, unsafe_allow_html=True)
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # ---- ML Model Predictions Comparison ----
        st.markdown("### 🤖 ML Model Predictions Comparison")
        st.markdown("See how traditional ML algorithms compare with the DistilBERT transformer.")
        
        ml_cols = st.columns(5)
        
        # DistilBERT result first
        with ml_cols[0]:
            st.markdown(f"""
            <div style="
                background: rgba(102,126,234,0.15);
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
                border: 2px solid #667eea;
                min-height: 180px;
            ">
                <div style="font-size: 1.2rem;">🧠</div>
                <div style="color: #667eea; font-weight: 700; font-size: 0.75rem;">DistilBERT</div>
                <div style="font-size: 1.8rem; margin: 0.3rem 0;">{EMOTION_EMOJIS.get(dominant, '🎭')}</div>
                <div style="color: white; font-weight: 800; font-size: 1rem;">{dominant.upper()}</div>
                <div style="color: #a0aec0; font-size: 0.75rem;">{emotions[dominant]:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ML model results
        for idx, (model_name, model) in enumerate(trained_models.items()):
            ml_label, ml_probas = ml_predict(lyrics_input, model, tfidf_vectorizer, label_encoder)
            ml_emoji = EMOTION_EMOJIS.get(ml_label, '🎭')
            ml_conf = ml_probas.get(ml_label, 0)
            info = ML_MODEL_INFO[model_name]
            with ml_cols[idx + 1]:
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.05);
                    border-radius: 12px;
                    padding: 1rem;
                    text-align: center;
                    border: 1px solid rgba(255,255,255,0.15);
                    min-height: 180px;
                ">
                    <div style="font-size: 1.2rem;">{info['icon']}</div>
                    <div style="color: {info['color']}; font-weight: 700; font-size: 0.7rem;">{model_name}</div>
                    <div style="font-size: 1.8rem; margin: 0.3rem 0;">{ml_emoji}</div>
                    <div style="color: white; font-weight: 800; font-size: 1rem;">{ml_label.upper()}</div>
                    <div style="color: #a0aec0; font-size: 0.75rem;">{ml_conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # ---- YouTube Song Suggestions ----
        st.markdown("### 🎬 YouTube Song Suggestions")
        st.markdown(f"Songs matching the **{dominant}** {EMOTION_EMOJIS.get(dominant, '')} mood:")
        
        yt_query = get_emotion_song_query(dominant)
        yt_results = search_youtube(yt_query, max_results=6)
        
        if yt_results:
            yt_cols = st.columns(3)
            for idx, video in enumerate(yt_results):
                with yt_cols[idx % 3]:
                    st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.05);
                        border-radius: 12px;
                        padding: 0.5rem;
                        margin-bottom: 0.8rem;
                        border: 1px solid rgba(255,255,255,0.1);
                    ">
                        <iframe width="100%" height="180" 
                            src="https://www.youtube.com/embed/{video['video_id']}" 
                            frameborder="0" 
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                            allowfullscreen
                            style="border-radius: 8px;">
                        </iframe>
                        <div style="padding: 0.4rem 0.2rem 0;">
                            <div style="color: white; font-weight: 600; font-size: 0.8rem; line-height: 1.3;">{video['title'][:55]}</div>
                            <div style="color: #a0aec0; font-size: 0.7rem; margin-top: 0.2rem;">🎵 {video['channel']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("YouTube suggestions unavailable. Check your API key.")
    
    elif analyze_btn:
        st.warning("⚠️ Please enter some lyrics to analyze.")

# ============================================================
# MODE 2: CSV BULK ANALYSIS
# ============================================================
elif "CSV" in mode:
    st.markdown("## 📊 CSV Bulk Analysis")
    st.markdown("Upload a CSV file containing song lyrics for batch emotion analysis.")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    st.info("📋 **CSV Requirements:** Your file must contain a column named `lyrics` or `Lyric` with song text.")
    
    uploaded_file = st.file_uploader(
        "📁 Upload CSV File",
        type=["csv"],
        help="Upload a CSV file with a 'lyrics' or 'Lyric' column"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded! Found **{len(df)} rows** and **{len(df.columns)} columns**.")
            
            # Detect lyrics column
            lyrics_col = None
            for col in df.columns:
                if col.lower() in ["lyrics", "lyric"]:
                    lyrics_col = col
                    break
            
            if lyrics_col is None:
                st.error("❌ Could not find a column named 'lyrics' or 'Lyric'. Please check your CSV.")
            else:
                st.markdown(f"📌 Using column: **`{lyrics_col}`**")
                
                # Show preview
                with st.expander("👀 Preview Uploaded Data", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
                with col_btn2:
                    process_btn = st.button("🚀 Process All Songs", use_container_width=True)
                
                if process_btn:
                    emotion_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total = len(df)
                    for idx, row in df.iterrows():
                        lyrics_text = str(row[lyrics_col])
                        emotions = predict_emotions(classifier, lyrics_text)
                        emotion_results.append(emotions)
                        
                        progress = (idx + 1) / total
                        progress_bar.progress(progress)
                        status_text.markdown(
                            f"⏳ Processing song **{idx + 1}/{total}**... "
                            f"({progress*100:.0f}%)"
                        )
                    
                    progress_bar.empty()
                    status_text.success(f"✅ All **{total}** songs analyzed!")
                    
                    # Create emotion columns
                    emotion_df = pd.DataFrame(emotion_results)
                    emotion_cols = []
                    for emotion in EMOTION_EMOJIS.keys():
                        col_name = f"{emotion} %"
                        df[col_name] = emotion_df[emotion]
                        emotion_cols.append(col_name)
                    
                    # Add dominant emotion column
                    df["dominant_emotion"] = emotion_df.apply(
                        lambda row: max(row.to_dict(), key=row.to_dict().get),
                        axis=1
                    )
                    
                    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
                    
                    # ---- Dataset Metrics ----
                    st.markdown("### 📊 Dataset Emotion Summary")
                    
                    avg_emotions = {
                        e: df[f"{e} %"].mean() for e in EMOTION_EMOJIS.keys()
                    }
                    dataset_dominant = get_dominant_emotion(avg_emotions)
                    
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    with mcol1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{total}</div>
                            <div class="metric-label">Songs Analyzed</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with mcol2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.8rem;">{EMOTION_EMOJIS.get(dataset_dominant, '🎭')}</div>
                            <div class="metric-value">{dataset_dominant.upper()}</div>
                            <div class="metric-label">Dataset Dominant Emotion</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with mcol3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{avg_emotions[dataset_dominant]:.1f}%</div>
                            <div class="metric-label">Avg Dominant Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with mcol4:
                        most_common = df["dominant_emotion"].value_counts().index[0]
                        count = df["dominant_emotion"].value_counts().values[0]
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{EMOTION_EMOJIS.get(most_common, '')} {count}</div>
                            <div class="metric-label">Most Common: {most_common.capitalize()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
                    
                    # ---- Visualizations ----
                    st.markdown("### 📈 Dataset Visualizations")
                    
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Average Emotions", "🕸️ Radar", "🎭 Heatmap", "📈 Distribution"
                    ])
                    
                    with tab1:
                        fig_avg = create_bar_chart(avg_emotions, "Average Emotion Distribution Across Dataset")
                        st.plotly_chart(fig_avg, use_container_width=True)
                    
                    with tab2:
                        fig_radar = create_radar_chart(avg_emotions, "Dataset Emotion Radar")
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    with tab3:
                        fig_heatmap = create_dataset_heatmap(df, emotion_cols)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    with tab4:
                        # Emotion distribution (pie chart)
                        dominant_counts = df["dominant_emotion"].value_counts()
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=[f"{EMOTION_EMOJIS.get(e, '')} {e.capitalize()}" for e in dominant_counts.index],
                            values=dominant_counts.values,
                            marker=dict(colors=[EMOTION_COLORS.get(e, "#667eea") for e in dominant_counts.index]),
                            hole=0.4,
                            textinfo="label+percent",
                            textfont=dict(color="white", size=12),
                            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
                        )])
                        fig_pie.update_layout(
                            title=dict(text="🎯 Dominant Emotion Distribution", font=dict(color="white", size=18)),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            legend=dict(font=dict(color="white")),
                            margin=dict(t=50, b=20),
                            height=450
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
                    
                    # ---- Updated DataFrame ----
                    st.markdown("### 📋 Updated Dataset with Emotion Scores")
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # ---- Download Button ----
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
                    
                    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1])
                    with dl_col2:
                        st.download_button(
                            label="📥 Download Processed CSV",
                            data=csv_data,
                            file_name="songs_with_emotions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ============================================================
# MODE 3: SONG COMPARISON
# ============================================================
elif "Comparison" in mode:
    st.markdown("## 🔀 Song Emotion Comparison")
    st.markdown("Compare the emotional profiles of multiple songs side by side.")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    num_songs = st.slider("🎵 Number of songs to compare", min_value=2, max_value=5, value=2)
    
    songs_data = []
    
    cols = st.columns(num_songs)
    for i in range(num_songs):
        with cols[i]:
            st.markdown(f"### 🎵 Song {i + 1}")
            name = st.text_input(
                f"Song Name",
                value=f"Song {i + 1}",
                key=f"song_name_{i}"
            )
            lyrics = st.text_area(
                f"Lyrics",
                height=150,
                placeholder="Paste lyrics here...",
                key=f"song_lyrics_{i}"
            )
            songs_data.append({"name": name, "lyrics": lyrics})
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        compare_btn = st.button("⚡ Compare Emotions", use_container_width=True)
    
    if compare_btn:
        valid_songs = [s for s in songs_data if s["lyrics"].strip()]
        
        if len(valid_songs) < 2:
            st.warning("⚠️ Please enter lyrics for at least 2 songs.")
        else:
            with st.spinner("🎯 Analyzing all songs..."):
                for song in valid_songs:
                    song["emotions"] = predict_emotions(classifier, song["lyrics"])
                    song["dominant"] = get_dominant_emotion(song["emotions"])
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # Results for each song
            st.markdown("### 🎯 Individual Results")
            result_cols = st.columns(len(valid_songs))
            for idx, song in enumerate(valid_songs):
                with result_cols[idx]:
                    dominant = song["dominant"]
                    emoji = EMOTION_EMOJIS.get(dominant, "🎭")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.5rem; color: white; font-weight: 700;">{song['name']}</div>
                        <div style="font-size: 2rem; margin: 0.5rem 0;">{emoji}</div>
                        <div class="metric-value">{dominant.upper()}</div>
                        <div class="metric-label">{song['emotions'][dominant]:.1f}% confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # Comparison chart
            st.markdown("### 📊 Comparison Chart")
            fig_compare = create_comparison_chart(valid_songs)
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Individual radar charts
            st.markdown("### 🕸️ Radar Comparison")
            radar_cols = st.columns(len(valid_songs))
            for idx, song in enumerate(valid_songs):
                with radar_cols[idx]:
                    fig = create_radar_chart(
                        song["emotions"],
                        f"{song['name']}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODE 4: ML MODEL COMPARISON
# ============================================================
elif "ML" in mode:
    st.markdown("## 🤖 Machine Learning Model Comparison")
    st.markdown("Compare traditional ML algorithms trained on the song lyrics dataset.")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ---- Model Accuracy Overview ----
    st.markdown("### 📊 Model Accuracy Overview")
    
    acc_cols = st.columns(4)
    for idx, (model_name, data) in enumerate(ml_metrics.items()):
        info = ML_MODEL_INFO[model_name]
        acc = data["accuracy"] * 100
        with acc_cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">{info['icon']}</div>
                <div style="color: {info['color']}; font-weight: 700; font-size: 0.9rem;">{model_name}</div>
                <div class="metric-value">{acc:.1f}%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ---- Accuracy Comparison Bar Chart ----
    st.markdown("### 📈 Accuracy Comparison")
    
    model_names = list(ml_metrics.keys())
    accuracies = [ml_metrics[m]["accuracy"] * 100 for m in model_names]
    colors = [ML_MODEL_INFO[m]["color"] for m in model_names]
    icons = [ML_MODEL_INFO[m]["icon"] for m in model_names]
    
    fig_acc = go.Figure(data=[
        go.Bar(
            x=[f"{icons[i]} {model_names[i]}" for i in range(len(model_names))],
            y=accuracies,
            marker=dict(color=colors, opacity=0.9,
                        line=dict(color="rgba(255,255,255,0.3)", width=1)),
            text=[f"{a:.1f}%" for a in accuracies],
            textposition="outside",
            textfont=dict(color="white", size=14, family="Arial Black"),
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>"
        )
    ])
    fig_acc.update_layout(
        title=dict(text="Model Accuracy Comparison", font=dict(color="white", size=18)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="white", size=12), showgrid=False),
        yaxis=dict(
            title="Accuracy (%)",
            title_font=dict(color="#a0aec0"),
            tickfont=dict(color="#a0aec0"),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            range=[0, max(accuracies) * 1.2]
        ),
        margin=dict(t=50, b=20, l=40, r=20),
        height=400
    )
    st.plotly_chart(fig_acc, use_container_width=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ---- Detailed Metrics Per Model ----
    st.markdown("### 📋 Detailed Model Metrics")
    
    selected_model = st.selectbox(
        "Select a model to view detailed metrics:",
        model_names,
        index=0
    )
    
    if selected_model:
        data = ml_metrics[selected_model]
        info = ML_MODEL_INFO[selected_model]
        
        detail_col1, detail_col2 = st.columns([1, 1])
        
        with detail_col1:
            st.markdown(f"#### {info['icon']} {selected_model}")
            st.markdown(f"*{info['desc']}*")
            
            # Classification Report Table
            report = data["report"]
            report_data = []
            for emotion in label_encoder.classes_:
                if emotion in report:
                    r = report[emotion]
                    report_data.append({
                        "Emotion": f"{EMOTION_EMOJIS.get(emotion, '')} {emotion.capitalize()}",
                        "Precision": f"{r['precision']*100:.1f}%",
                        "Recall": f"{r['recall']*100:.1f}%",
                        "F1-Score": f"{r['f1-score']*100:.1f}%",
                        "Support": int(r['support'])
                    })
            
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True, hide_index=True)
            
            # Overall metrics
            st.markdown(f"""
            <div class="metric-card" style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-around;">
                    <div>
                        <div class="metric-value">{data['accuracy']*100:.1f}%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div>
                        <div class="metric-value">{report['weighted avg']['precision']*100:.1f}%</div>
                        <div class="metric-label">Wt. Precision</div>
                    </div>
                    <div>
                        <div class="metric-value">{report['weighted avg']['recall']*100:.1f}%</div>
                        <div class="metric-label">Wt. Recall</div>
                    </div>
                    <div>
                        <div class="metric-value">{report['weighted avg']['f1-score']*100:.1f}%</div>
                        <div class="metric-label">Wt. F1-Score</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with detail_col2:
            # Confusion Matrix Heatmap
            st.markdown("#### 🔥 Confusion Matrix")
            cm = data["confusion_matrix"]
            emotion_labels = [f"{EMOTION_EMOJIS.get(e, '')} {e[:3].upper()}" for e in label_encoder.classes_]
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=emotion_labels,
                y=emotion_labels,
                colorscale=[
                    [0, "#0f0c29"],
                    [0.3, "#302b63"],
                    [0.6, "#667eea"],
                    [0.8, "#764ba2"],
                    [1, "#f093fb"]
                ],
                text=[[str(val) for val in row] for row in cm],
                texttemplate="%{text}",
                textfont=dict(size=12, color="white"),
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            ))
            fig_cm.update_layout(
                xaxis=dict(title="Predicted", title_font=dict(color="#a0aec0"),
                          tickfont=dict(color="white", size=10)),
                yaxis=dict(title="Actual", title_font=dict(color="#a0aec0"),
                          tickfont=dict(color="white", size=10), autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=40, l=60, r=20),
                height=380
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ---- F1-Score Comparison Across All Models ----
    st.markdown("### 🏆 F1-Score Comparison Across All Models")
    
    f1_fig = go.Figure()
    for model_name in model_names:
        report = ml_metrics[model_name]["report"]
        emotions_list = [e for e in label_encoder.classes_ if e in report]
        f1_scores = [report[e]["f1-score"] * 100 for e in emotions_list]
        emotion_labels_f1 = [f"{EMOTION_EMOJIS.get(e, '')} {e.capitalize()}" for e in emotions_list]
        
        f1_fig.add_trace(go.Bar(
            name=f"{ML_MODEL_INFO[model_name]['icon']} {model_name}",
            x=emotion_labels_f1,
            y=f1_scores,
            marker_color=ML_MODEL_INFO[model_name]["color"],
            text=[f"{s:.0f}%" for s in f1_scores],
            textposition="outside",
            textfont=dict(size=9, color="white")
        ))
    
    f1_fig.update_layout(
        barmode="group",
        title=dict(text="F1-Score by Emotion & Model", font=dict(color="white", size=18)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="white", size=11), showgrid=False),
        yaxis=dict(
            title="F1-Score (%)",
            title_font=dict(color="#a0aec0"),
            tickfont=dict(color="#a0aec0"),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)"
        ),
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)"
        ),
        margin=dict(t=50, b=20, l=40, r=20),
        height=450
    )
    st.plotly_chart(f1_fig, use_container_width=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ---- Live Prediction with All Models ----
    st.markdown("### 🎯 Live Prediction — All Models")
    st.markdown("Enter lyrics below to see how each model predicts the emotion.")
    
    ml_lyrics_input = st.text_area(
        "📝 Enter Song Lyrics for ML Comparison",
        height=150,
        placeholder="Paste lyrics to compare across all models...",
        key="ml_lyrics"
    )
    
    ml_btn_col1, ml_btn_col2, ml_btn_col3 = st.columns([1, 1, 1])
    with ml_btn_col2:
        ml_predict_btn = st.button("⚡ Predict with All Models", use_container_width=True)
    
    if ml_predict_btn and ml_lyrics_input.strip():
        with st.spinner("🔮 Running predictions..."):
            # DistilBERT prediction
            bert_emotions = predict_emotions(classifier, ml_lyrics_input)
            bert_dominant = get_dominant_emotion(bert_emotions)
            
            # ML predictions
            ml_results = {}
            for model_name, model in trained_models.items():
                label, probas = ml_predict(ml_lyrics_input, model, tfidf_vectorizer, label_encoder)
                ml_results[model_name] = {"label": label, "probas": probas}
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # Show all predictions side by side
        st.markdown("#### 🏅 Prediction Results")
        pred_cols = st.columns(5)
        
        # DistilBERT
        with pred_cols[0]:
            st.markdown(f"""
            <div style="
                background: rgba(102,126,234,0.15);
                border-radius: 12px;
                padding: 1.2rem;
                text-align: center;
                border: 2px solid #667eea;
                min-height: 200px;
            ">
                <div style="font-size: 1.5rem;">🧠</div>
                <div style="color: #667eea; font-weight: 700; font-size: 0.85rem;">DistilBERT</div>
                <div style="font-size: 2.5rem; margin: 0.5rem 0;">{EMOTION_EMOJIS.get(bert_dominant, '🎭')}</div>
                <div style="color: white; font-weight: 800; font-size: 1.2rem;">{bert_dominant.upper()}</div>
                <div style="color: #a0aec0; font-size: 0.8rem;">{bert_emotions[bert_dominant]:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        for idx, (model_name, result) in enumerate(ml_results.items()):
            info = ML_MODEL_INFO[model_name]
            ml_label = result["label"]
            ml_conf = result["probas"].get(ml_label, 0)
            with pred_cols[idx + 1]:
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.05);
                    border-radius: 12px;
                    padding: 1.2rem;
                    text-align: center;
                    border: 1px solid rgba(255,255,255,0.15);
                    min-height: 200px;
                ">
                    <div style="font-size: 1.5rem;">{info['icon']}</div>
                    <div style="color: {info['color']}; font-weight: 700; font-size: 0.8rem;">{model_name}</div>
                    <div style="font-size: 2.5rem; margin: 0.5rem 0;">{EMOTION_EMOJIS.get(ml_label, '🎭')}</div>
                    <div style="color: white; font-weight: 800; font-size: 1.2rem;">{ml_label.upper()}</div>
                    <div style="color: #a0aec0; font-size: 0.8rem;">{ml_conf:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # Probability comparison chart
        st.markdown("#### 📊 Probability Distribution Comparison")
        
        all_models_fig = go.Figure()
        
        # DistilBERT bars
        emotion_list = list(EMOTION_EMOJIS.keys())
        bert_vals = [bert_emotions.get(e, 0) for e in emotion_list]
        all_models_fig.add_trace(go.Bar(
            name="🧠 DistilBERT",
            x=[f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in emotion_list],
            y=bert_vals,
            marker_color="#667eea"
        ))
        
        # ML model bars
        for model_name, result in ml_results.items():
            info = ML_MODEL_INFO[model_name]
            vals = [result["probas"].get(e, 0) for e in emotion_list]
            all_models_fig.add_trace(go.Bar(
                name=f"{info['icon']} {model_name}",
                x=[f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in emotion_list],
                y=vals,
                marker_color=info["color"]
            ))
        
        all_models_fig.update_layout(
            barmode="group",
            title=dict(text="All Models — Emotion Probabilities", font=dict(color="white", size=18)),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickfont=dict(color="white", size=11), showgrid=False),
            yaxis=dict(
                title="Probability (%)",
                title_font=dict(color="#a0aec0"),
                tickfont=dict(color="#a0aec0"),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)"
            ),
            legend=dict(
                font=dict(color="white"),
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="rgba(255,255,255,0.1)"
            ),
            margin=dict(t=50, b=20, l=40, r=20),
            height=450
        )
        st.plotly_chart(all_models_fig, use_container_width=True)
    
    elif ml_predict_btn:
        st.warning("⚠️ Please enter some lyrics to predict.")

# ============================================================
# MODE 5: SONG SEARCH
# ============================================================
elif "Search" in mode:
    st.markdown("## 🔍 Song Search & Emotion Discovery")
    st.markdown(f"Search through **{dataset_info['total_songs']:,}** songs from **{dataset_info['artists']}** artists.")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # Dataset Stats
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{dataset_info['total_songs']:,}</div>
            <div class="metric-label">Total Songs</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{dataset_info['artists']}</div>
            <div class="metric-label">Artists</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_cols[2]:
        top_emotion = max(dataset_info['emotion_distribution'], key=dataset_info['emotion_distribution'].get)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem;">{EMOTION_EMOJIS.get(top_emotion, '')}</div>
            <div class="metric-value">{top_emotion.upper()}</div>
            <div class="metric-label">Most Common Emotion</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">21</div>
            <div class="metric-label">CSV Files Loaded</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # Search Bar
    search_query = st.text_input(
        "🔍 Search by Song Title or Artist",
        placeholder="e.g. Taylor Swift, Love Story, Eminem...",
        key="song_search"
    )
    
    # Emotion filter
    filter_col1, filter_col2 = st.columns([1, 1])
    with filter_col1:
        emotion_filter = st.multiselect(
            "🎭 Filter by Emotion",
            options=list(EMOTION_EMOJIS.keys()),
            format_func=lambda x: f"{EMOTION_EMOJIS[x]} {x.capitalize()}",
            default=[]
        )
    with filter_col2:
        max_results = st.slider("📊 Max Results", min_value=5, max_value=50, value=20)
    
    if search_query.strip():
        results = search_songs_in_db(song_database, search_query, max_results=max_results)
        
        # Apply emotion filter
        if emotion_filter:
            results = results[results["emotion"].isin(emotion_filter)]
        
        if len(results) > 0:
            st.success(f"🎵 Found **{len(results)}** songs matching \"{search_query}\"")
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # Display results as cards
            for idx, (_, row) in enumerate(results.iterrows()):
                emotion = row["emotion"]
                emoji = EMOTION_EMOJIS.get(emotion, "🎭")
                color = EMOTION_COLORS.get(emotion, "#667eea")
                
                with st.expander(f"{emoji} {row['Title']} — {row['Artist']} ({emotion.capitalize()})", expanded=(idx < 3)):
                    info_col1, info_col2 = st.columns([2, 1])
                    
                    with info_col1:
                        st.markdown(f"""
                        <div style="
                            background: rgba(255,255,255,0.05);
                            border-radius: 12px;
                            padding: 1rem;
                            border-left: 4px solid {color};
                        ">
                            <div style="color: white; font-weight: 700; font-size: 1.1rem;">🎵 {row['Title']}</div>
                            <div style="color: #a0aec0; font-size: 0.9rem;">by {row['Artist']}</div>
                            <div style="margin-top: 0.5rem;">
                                <span style="
                                    background: {color}20;
                                    color: {color};
                                    padding: 0.25rem 0.75rem;
                                    border-radius: 20px;
                                    font-size: 0.8rem;
                                    font-weight: 600;
                                ">{emoji} {emotion.capitalize()}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show lyrics preview
                        lyric_preview = str(row["Lyric"])[:300]
                        st.markdown(f"**Lyrics Preview:**")
                        st.text(lyric_preview + "...")
                    
                    with info_col2:
                        # Run emotion analysis on this song
                        song_emotions = predict_emotions(classifier, str(row["Lyric"]))
                        fig = create_bar_chart(song_emotions, f"Emotion Profile")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # YouTube embedded player for this song
                    yt_search = f"{row['Title']} {row['Artist']} official"
                    yt_results = search_youtube(yt_search, max_results=2)
                    if yt_results:
                        st.markdown("**🎬 Listen on YouTube:**")
                        yt_link_cols = st.columns(2)
                        for vidx, video in enumerate(yt_results):
                            with yt_link_cols[vidx]:
                                st.markdown(f"""
                                <div style="
                                    background: rgba(255,255,255,0.05);
                                    border-radius: 10px;
                                    padding: 0.4rem;
                                    border: 1px solid rgba(255,255,255,0.1);
                                ">
                                    <iframe width="100%" height="160"
                                        src="https://www.youtube.com/embed/{video['video_id']}"
                                        frameborder="0"
                                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                        allowfullscreen
                                        style="border-radius: 8px;">
                                    </iframe>
                                    <div style="padding: 0.3rem 0.2rem 0;">
                                        <div style="color: white; font-weight: 600; font-size: 0.75rem;">{video['title'][:45]}</div>
                                        <div style="color: #a0aec0; font-size: 0.65rem;">{video['channel']}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # Emotion distribution of search results
            st.markdown("### 📊 Emotion Distribution of Search Results")
            emotion_dist = results["emotion"].value_counts()
            dist_fig = go.Figure(data=[go.Pie(
                labels=[f"{EMOTION_EMOJIS.get(e, '')} {e.capitalize()}" for e in emotion_dist.index],
                values=emotion_dist.values,
                marker=dict(colors=[EMOTION_COLORS.get(e, "#667eea") for e in emotion_dist.index]),
                hole=0.4,
                textinfo="label+percent",
                textfont=dict(color="white", size=12),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
            )])
            dist_fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="white")),
                margin=dict(t=20, b=20),
                height=350
            )
            st.plotly_chart(dist_fig, use_container_width=True)
        
        else:
            st.warning(f"No songs found matching \"{search_query}\". Try a different search term.")
    
    else:
        # Show browsing by emotion when no search
        st.markdown("### 🎭 Browse by Emotion")
        st.markdown("Click an emotion to see songs from the database:")
        
        browse_emotion = st.selectbox(
            "Select Emotion",
            options=list(EMOTION_EMOJIS.keys()),
            format_func=lambda x: f"{EMOTION_EMOJIS[x]} {x.capitalize()}",
            key="browse_emotion"
        )
        
        if browse_emotion:
            emotion_songs = song_database[song_database["emotion"] == browse_emotion].head(15)
            st.markdown(f"#### {EMOTION_EMOJIS[browse_emotion]} Top {len(emotion_songs)} {browse_emotion.capitalize()} Songs")
            
            for _, row in emotion_songs.iterrows():
                color = EMOTION_COLORS.get(browse_emotion, "#667eea")
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.05);
                    border-radius: 10px;
                    padding: 0.8rem 1rem;
                    margin-bottom: 0.5rem;
                    border-left: 3px solid {color};
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <span style="color: white; font-weight: 600;">🎵 {row['Title']}</span>
                        <span style="color: #a0aec0; font-size: 0.85rem;"> — {row['Artist']}</span>
                    </div>
                    <span style="
                        background: {color}20;
                        color: {color};
                        padding: 0.2rem 0.6rem;
                        border-radius: 20px;
                        font-size: 0.75rem;
                        font-weight: 600;
                    ">{EMOTION_EMOJIS[browse_emotion]} {browse_emotion.capitalize()}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # YouTube suggestions for this emotion
            st.markdown(f"### 🎬 YouTube: {browse_emotion.capitalize()} Songs")
            yt_query = get_emotion_song_query(browse_emotion)
            yt_results = search_youtube(yt_query, max_results=6)
            
            if yt_results:
                yt_cols = st.columns(3)
                for vidx, video in enumerate(yt_results):
                    with yt_cols[vidx % 3]:
                        st.markdown(f"""
                        <div style="
                            background: rgba(255,255,255,0.05);
                            border-radius: 12px;
                            padding: 0.5rem;
                            margin-bottom: 0.8rem;
                            border: 1px solid rgba(255,255,255,0.1);
                        ">
                            <iframe width="100%" height="180"
                                src="https://www.youtube.com/embed/{video['video_id']}"
                                frameborder="0"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowfullscreen
                                style="border-radius: 8px;">
                            </iframe>
                            <div style="padding: 0.4rem 0.2rem 0;">
                                <div style="color: white; font-weight: 600; font-size: 0.8rem; line-height: 1.3;">{video['title'][:55]}</div>
                                <div style="color: #a0aec0; font-size: 0.7rem; margin-top: 0.2rem;">🎵 {video['channel']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# ============================================================
# MODE 6: ARTIST EMOTION DNA
# ============================================================
elif "Artist Emotion DNA" in mode:
    st.markdown("## 👤 Artist Emotion DNA")
    st.markdown("Analyze ALL songs by an artist to reveal their unique emotional fingerprint.")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### 🏆 Global Emotion Leaderboard")
    st.markdown("Which artists represent each emotion the most? *(Based on artists with 5+ songs in DB)*")
    
    # Group by artist and calculate statistics
    artist_counts = song_database.groupby(['Artist', 'emotion']).size().unstack(fill_value=0)
    artist_pct = artist_counts.div(artist_counts.sum(axis=1), axis=0) * 100
    artist_totals = song_database['Artist'].value_counts()
    valid_artists = artist_totals[artist_totals >= 5].index
    
    if len(valid_artists) > 0:
        valid_artist_pct = artist_pct.loc[artist_pct.index.isin(valid_artists)]
        
        # Display superlaties in 2 rows of 3
        em_list = list(EMOTION_EMOJIS.keys())
        rows = [st.columns(3), st.columns(3)]
        
        for idx, em in enumerate(em_list):
            if idx < 6:
                current_col = rows[idx // 3][idx % 3]
                with current_col:
                    if em in valid_artist_pct.columns and not valid_artist_pct[em].empty:
                        top_artist = valid_artist_pct[em].idxmax()
                        top_score = valid_artist_pct[em].max()
                        emoji = EMOTION_EMOJIS.get(em, "🎭")
                        color = EMOTION_COLORS.get(em, "#667eea")
                        
                        superlative = em.capitalize()
                        if em == 'joy': superlative = 'Happiest'
                        elif em == 'sadness': superlative = 'Saddest'
                        elif em == 'anger': superlative = 'Angriest'
                        elif em == 'love': superlative = 'Most Loving'
                        elif em == 'fear': superlative = 'Most Fearful'
                        elif em == 'surprise': superlative = 'Most Surprising'
                        
                        st.markdown(f"""
                        <div style="
                            background: rgba(255,255,255,0.05);
                            border-radius: 12px;
                            padding: 1.2rem;
                            text-align: center;
                            border: 1px solid rgba(255,255,255,0.1);
                            margin-bottom: 1rem;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        ">
                            <div style="color: #a0aec0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">{superlative} Artist</div>
                            <div style="font-size: 2.2rem; margin:0.4rem 0;">{emoji}</div>
                            <div style="font-size: 1.2rem; font-weight:700; color:white;">{top_artist}</div>
                            <div style="color: {color}; font-weight:800; font-size:1rem; margin-top:0.3rem;">{top_score:.1f}% {em.capitalize()}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🧬 Individual Artist DNA Profiler")
    
    unique_artists = sorted([str(a) for a in song_database['Artist'].unique() if pd.notna(a) and str(a) != "Unknown" and str(a) != "nan"])
    selected_artist = st.selectbox("Select an Artist to view their full emotional profile:", unique_artists)
    
    if selected_artist:
        artist_songs = song_database[song_database['Artist'] == selected_artist]
        total_artist_songs = len(artist_songs)
        
        if total_artist_songs > 0:
            st.markdown(f"**Analyzing {total_artist_songs} songs from {selected_artist}...**")
            
            # Calculate artist emotion distribution
            emotion_counts = artist_songs['emotion'].value_counts()
            artist_emotions = {e: 0.0 for e in EMOTION_EMOJIS.keys()}
            for em, count in emotion_counts.items():
                if em in artist_emotions:
                    artist_emotions[em] = round((count / total_artist_songs) * 100, 1)
                    
            # Top emotion
            top_emotion = max(artist_emotions, key=artist_emotions.get)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="margin-top:20px; text-align:center; padding: 2rem;">
                    <div style="font-size: 4rem;">{EMOTION_EMOJIS.get(top_emotion, "🎭")}</div>
                    <div style="font-size: 1.5rem; font-weight:700; color:white; margin-top:10px;">{top_emotion.upper()}</div>
                    <div class="metric-label">{artist_emotions[top_emotion]:.1f}% Dominant</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show top 5 songs for this artist for the dominant emotion
                top_songs = artist_songs[artist_songs['emotion'] == top_emotion].head(5)
                if not top_songs.empty:
                    st.markdown(f"<br>**Top {top_emotion.capitalize()} Songs:**", unsafe_allow_html=True)
                    for _, ts in top_songs.iterrows():
                        st.markdown(f"- 🎵 {ts['Title']}")
                        
            with col2:
                fig_artist = create_radar_chart(artist_emotions, f"{selected_artist}'s Emotion Profile")
                st.plotly_chart(fig_artist, use_container_width=True)


# ============================================================
# MODE 7: MOOD PLAYLIST GENERATOR
# ============================================================
elif "Playlist" in mode:
    st.markdown("## 🎵 AI Mood Playlist Generator")
    st.markdown("Select an emotion, and the AI will curate a tailored playlist from the 6,000+ database and YouTube.")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    selected_mood = st.selectbox("How are you feeling?", options=list(EMOTION_EMOJIS.keys()), format_func=lambda x: f"{EMOTION_EMOJIS[x]} {x.capitalize()}")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        generate_btn = st.button("✨ Generate My Playlist", use_container_width=True)
        
    if generate_btn:
        with st.spinner("🎧 Curating your playlist..."):
            time.sleep(1.5)
            mood_songs = song_database[song_database['emotion'] == selected_mood]
            # Shuffle and pick top 10
            if not mood_songs.empty:
                playlist_songs = mood_songs.sample(n=min(10, len(mood_songs)))
                
                st.success(f"Here is your {selected_mood.capitalize()} {EMOTION_EMOJIS[selected_mood]} Playlist!")
                
                # Render Playlist Timeline
                for idx, row in playlist_songs.reset_index().iterrows():
                    color = EMOTION_COLORS.get(selected_mood, "#667eea")
                    st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.05);
                        border-radius: 10px;
                        padding: 1rem;
                        margin-bottom: 0.8rem;
                        border-left: 4px solid {color};
                    ">
                        <span style="font-size:1.2rem; font-weight:700; color:white;">{idx+1}. {row['Title']}</span>
                        <span style="color:#a0aec0; margin-left: 10px;">by {row['Artist']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
                
                st.markdown("### 🎬 Watch/Listen on YouTube")
                yt_query = get_emotion_song_query(selected_mood)
                yt_results = search_youtube(yt_query, max_results=3)
                if yt_results:
                    yt_cols = st.columns(3)
                    for vidx, video in enumerate(yt_results):
                        with yt_cols[vidx]:
                            st.markdown(f"""
                            <div style="
                                background: rgba(255,255,255,0.05);
                                border-radius: 12px;
                                padding: 0.5rem;
                                border: 1px solid rgba(255,255,255,0.1);
                            ">
                                <iframe width="100%" height="180"
                                    src="https://www.youtube.com/embed/{video['video_id']}"
                                    frameborder="0"
                                    allowfullscreen
                                    style="border-radius: 8px;">
                                </iframe>
                                <div style="padding: 0.4rem 0.2rem 0;">
                                    <div style="color: white; font-weight: 600; font-size: 0.8rem;">{video['title'][:55]}</div>
                                    <div style="color: #a0aec0; font-size: 0.7rem;">🎵 {video['channel']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("No songs found for this mood.")

# ============================================================
# FOOTER
# ============================================================
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #718096; font-size: 0.85rem;">
    <p>🎵 Song Lyrics Emotion Detection • A Data-Driven Approach to Understanding Human Emotions</p>
    <p>Powered by <strong>DistilBERT</strong> • <strong>Logistic Regression</strong> • <strong>Random Forest</strong> • <strong>SVM</strong> • <strong>Naive Bayes</strong></p>
    <p>Built with <strong>Streamlit</strong> • <strong>HuggingFace Transformers</strong> • <strong>Scikit-Learn</strong> • <strong>YouTube API</strong></p>
</div>
""", unsafe_allow_html=True)
