import { useState } from "react";
import "./index.css";

function App() {
  const [lyrics, setLyrics] = useState("");
  const [emotionInput, setEmotionInput] = useState("");
  const [result, setResult] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ===============================
  // Analyze Lyrics Emotion
  // ===============================
  const analyzeEmotion = async () => {
    if (!lyrics.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);
    setRecommendations([]);

    try {
      const response = await fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ lyrics })
      });

      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "Server error");

      setResult(data.emotion);
      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ===============================
  // Search Songs By Emotion
  // ===============================
  const searchByEmotion = async () => {
    if (!emotionInput.trim()) return;

    setLoading(true);
    setError("");
    setResult(emotionInput);
    setRecommendations([]);

    try {
      const response = await fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ lyrics: emotionInput })
      });

      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "Server error");

      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>🎵 Emotion Based Song Recommender</h1>
        <p>Detect emotion from lyrics or search by mood</p>
      </header>

      <main className="main-content">

        {/* ================= LYRICS INPUT ================= */}
        <section className="glass-card input-section">
          <h2>Analyze Lyrics</h2>
          <textarea
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
            placeholder="Paste song lyrics here..."
            rows="6"
          />
          <button onClick={analyzeEmotion} disabled={loading}>
            {loading ? "Analyzing..." : "Detect Emotion"}
          </button>
        </section>

        {/* ================= EMOTION SEARCH ================= */}
        <section className="glass-card input-section">
          <h2>Search By Emotion</h2>
          <input
            type="text"
            value={emotionInput}
            onChange={(e) => setEmotionInput(e.target.value)}
            placeholder="Enter emotion (joy, sadness, anger, love...)"
          />
          <button onClick={searchByEmotion} disabled={loading}>
            {loading ? "Searching..." : "Find Songs"}
          </button>
        </section>

        {/* ================= RESULT SECTION ================= */}
        <section className="result-section">
          {error && <p className="error">{error}</p>}

          {result && (
            <div className="glass-card result-card">
              <h3>Detected Emotion</h3>
              <h2 className="emotion-text">{result}</h2>
            </div>
          )}

          {recommendations.length > 0 && (
            <div className="glass-card recommendation-card">
              <h3>Recommended Songs</h3>
              <ul>
                {recommendations.map((rec, index) => (
                  <li key={index}>
                    🎶 <strong>{rec.song}</strong> – {rec.artist}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {!loading && !result && (
            <div className="glass-card placeholder">
              <p>Your recommendations will appear here ✨</p>
            </div>
          )}
        </section>

      </main>
    </div>
  );
}

export default App;