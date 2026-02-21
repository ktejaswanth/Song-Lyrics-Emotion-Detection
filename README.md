# 🎵 Song Lyrics Emotion Detection AI

A comprehensive, data-driven full-stack web application that predicts emotions from song lyrics using **DistilBERT Transformer** and **Traditional ML Algorithms**. Features an interactive Streamlit dashboard with YouTube integration, song search across 6,000+ songs, and real-time model comparison.

---

## ✨ Key Features

### 🎤 Single Song Analysis
- Paste any song lyrics and get an instant **emotion breakdown** (Joy, Sadness, Anger, Love, Fear, Surprise)
- **Bar chart**, **radar chart**, and **emotion ranking** visualizations
- **ML model predictions comparison** — see how DistilBERT, Logistic Regression, Random Forest, SVM, and Naive Bayes classify the same lyrics
- **YouTube song suggestions** — embedded YouTube players recommending songs matching the detected mood

### 📊 CSV Bulk Analysis
- Upload a CSV file with a `lyrics` or `Lyric` column for batch emotion detection
- Process hundreds of songs at once
- Download results with added emotion scores (percentage per emotion)
- View dataset-level summaries: heatmaps, average emotion charts

### 🔀 Song Comparison
- Input lyrics for 2+ songs side-by-side
- Compare emotional profiles with grouped bar charts and individual radar charts

### 🤖 ML Model Comparison
- **4 trained ML algorithms** compared against the DistilBERT transformer:
  - 📐 Logistic Regression
  - 🌲 Random Forest
  - 📊 SVM (LinearSVC)
  - 📈 Naive Bayes
- **Accuracy comparison** bar chart
- **Detailed metrics**: Precision, Recall, F1-Score per emotion class
- **Confusion matrices** for each model
- **Live prediction** — input lyrics and see all 5 models predict simultaneously

### 🔍 Song Search & Emotion Discovery
- Search through **6,000+ songs** from **21 popular artists** (Taylor Swift, Eminem, Drake, BTS, Billie Eilish, etc.)
- Filter by emotion, artist, or song title
- View **emotion profile bar charts** for each search result
- **Embedded YouTube players** — listen to songs right inside the app
- **Browse by Emotion** — discover songs grouped by mood

### 🎬 YouTube API Integration
- Real-time YouTube search powered by **YouTube Data API v3**
- Embedded video players — songs play directly in the app
- Mood-based song recommendations after every analysis

---

## 🏗️ Project Structure

```
skill palavar 2/
├── backend/                    # Python Backend
│   ├── app.py                  # Flask REST API (HuggingFace Transformers)
│   ├── streamlit_app.py        # Streamlit Dashboard (main app)
│   ├── prepare_dataset.py      # Dataset preparation script
│   ├── songs_with_emotion.csv  # Pre-labeled emotion dataset (~1,000 songs)
│   ├── requirements.txt        # Python dependencies
│   └── csv/                    # 21 Artist Lyric Datasets (~5,200 songs)
│       ├── TaylorSwift.csv
│       ├── Eminem.csv
│       ├── Drake.csv
│       ├── BTS.csv
│       ├── BillieEilish.csv
│       ├── ArianaGrande.csv
│       ├── EdSheeran.csv
│       ├── Beyonce.csv
│       ├── Rihanna.csv
│       ├── Coldplay.csv
│       └── ... (11 more artists)
├── frontend/                   # React Frontend (Vite)
│   ├── src/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── model/                      # Model Training & Evaluation
│   ├── train.py                # Fine-tune BERT
│   └── evaluate.py             # Generate metrics & confusion matrix
└── README.md
```

---

## 🧠 AI Models & Architecture

### Transformer Model
- **DistilBERT** (`bhadresh-savani/distilbert-base-uncased-emotion`) via HuggingFace Transformers
- Pre-trained on emotion classification — detects 6 emotions from text
- Used for high-accuracy emotion prediction on user input

### Traditional ML Pipeline
- **Training Data**: `songs_with_emotion.csv` (pre-labeled dataset)
- **Feature Extraction**: TF-IDF Vectorizer (8,000 features, unigrams + bigrams)
- **Models Trained**:
  | Model | Description |
  |-------|-------------|
  | Logistic Regression | Linear model using log-odds for multi-class classification |
  | Random Forest | Ensemble of 150 decision trees with bagging |
  | SVM (LinearSVC) | Support Vector Machine with linear kernel |
  | Naive Bayes | Probabilistic classifier based on Bayes' theorem |
- **Labeling Strategy**: The Logistic Regression model (fastest) is used to label 5,200+ songs from artist CSV files for the Song Search database

### Emotion Labels
| Emotion | Emoji | Color |
|---------|-------|-------|
| Joy | 😊 | #FFD700 |
| Sadness | 😢 | #4169E1 |
| Anger | 😡 | #FF4500 |
| Love | ❤️ | #FF69B4 |
| Fear | 😨 | #8B008B |
| Surprise | 😲 | #00CED1 |

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend only)
- YouTube Data API key (for song suggestions)

### 1. Backend Setup (Flask API)
```bash
cd backend
pip install -r requirements.txt
python app.py
```
The Flask server starts at `http://127.0.0.1:5000`.

### 2. Streamlit Dashboard (Main App)
```bash
cd backend
pip install -r requirements.txt
streamlit run streamlit_app.py
```
The Streamlit app opens at `http://localhost:8501`.

> **Note**: First launch takes ~15-30 seconds to load the DistilBERT model and train ML models. Results are cached for subsequent runs.

### 3. Frontend Setup (React UI — Optional)
```bash
cd frontend
npm install
npm run dev
```
Opens at `http://localhost:5173`.

---

## 📦 Dependencies

### Python (Backend & Streamlit)
| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive dashboard framework |
| `transformers` | DistilBERT emotion classifier (HuggingFace) |
| `scikit-learn` | ML models, TF-IDF, metrics |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `plotly` | Interactive visualizations |
| `flask` | REST API |
| `flask-cors` | Cross-origin requests |
| `requests` | YouTube API integration |

### Frontend
| Package | Purpose |
|---------|---------|
| `react` | UI framework |
| `vite` | Build tool & dev server |
| `chart.js` | Visualizations |

---

## 🔑 API Keys

### YouTube Data API v3
The app uses the YouTube Data API for song suggestions. The API key is configured in `streamlit_app.py`:
```python
YOUTUBE_API_KEY = "your-api-key-here"
```
To get your own key:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable **YouTube Data API v3**
3. Create an API key under Credentials

---

## 📊 Dataset Information

| Source | Songs | Description |
|--------|-------|-------------|
| `songs_with_emotion.csv` | ~1,000 | Pre-labeled with emotions (training data) |
| `csv/*.csv` (21 files) | ~5,200 | Artist lyrics labeled by ML model at runtime |
| **Total** | **~6,200** | Full searchable song database |

### Artists in Database
Ariana Grande, Beyoncé, Billie Eilish, BTS, Cardi B, Charlie Puth, Coldplay, Drake, Dua Lipa, Ed Sheeran, Eminem, Justin Bieber, Katy Perry, Khalid, Lady Gaga, Maroon 5, Nicki Minaj, Post Malone, Rihanna, Selena Gomez, Taylor Swift

---

## 🧪 Model Training (Optional)

To fine-tune your own BERT model:
```bash
cd model
python train.py        # Fine-tune BERT (GPU recommended)
python evaluate.py     # Generate metrics & confusion matrix
```
The fine-tuned model is saved to `./emotion_model`.

---

## ☁️ Azure Deployment (Optional)

### Backend (Azure App Service)
1. Create an **App Service** (Python 3.10) in Azure Portal
2. Deploy with VS Code Azure Tools or Git:
   ```bash
   az webapp up --sku F1 --name <your-app-name>
   ```
3. Set startup command:
   ```bash
   gunicorn --bind=0.0.0.0:8000 app:app
   ```

### Streamlit (Azure App Service)
1. Create an **App Service** (Python 3.10)
2. Set startup command:
   ```bash
   streamlit run streamlit_app.py --server.port 8000 --server.address 0.0.0.0
   ```

### Frontend (Azure Static Web Apps)
1. Create **Static Web App** in Azure Portal
2. Link GitHub repository
3. Build settings:
   - **App location**: `frontend`
   - **Output location**: `dist`
4. Update `fetch` URL in `App.jsx` to deployed backend URL

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI/ML** | DistilBERT, Scikit-Learn, TF-IDF |
| **Dashboard** | Streamlit |
| **Backend API** | Flask, Flask-CORS |
| **Frontend** | React, Vite |
| **Visualizations** | Plotly |
| **Data** | Pandas, NumPy |
| **External API** | YouTube Data API v3 |
| **Deployment** | Azure App Service, Azure Static Web Apps |

---

## 📸 App Modes Overview

| Mode | Description |
|------|-------------|
| 🎤 Single Song | Analyze one song's emotions + YouTube suggestions |
| 📊 CSV Bulk | Upload CSV for batch analysis + download results |
| 🔀 Comparison | Compare 2+ songs' emotional profiles |
| 🤖 ML Models | Train & compare 4 ML algorithms vs DistilBERT |
| 🔍 Song Search | Search 6,000+ songs by artist/title with embedded YouTube |

---

Built with ❤️ using **Streamlit** • **HuggingFace Transformers** • **Scikit-Learn** • **YouTube API** • **React** • **Flask**
