# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Skripsi/Tesis — Chatbot Kesehatan Mental Berbahasa Indonesia dengan Pendekatan Neuro-Fuzzy (TFNN)**

A mental health chatbot for Indonesian language using a **Takagi-Sugeno-Kang-inspired neuro-fuzzy neural network (TFNN)**. The model learns to rank response relevance given a user's question context using fuzzy membership layers + dense layers, trained on pairwise relevance data.

## Project Structure

| File | Purpose |
|------|---------|
| `fix_chatbot.py` | Original Colab notebook (training + eval + inference) |
| `chatbot_model.py` | TFNN model class, fuzzy layer, feature engineering, inference function |
| `prepare_model.py` | One-time training script → saves artifacts |
| `app.py` | Flask web server (chat UI + Firebase Firestore integration) |
| `templates/index.html` | Chat UI with Tailwind CSS + history support |
| `terjemah.csv` | Dataset (9,061 mental health Q&A pairs) |
| `serviceAccountKey.json` | **Firebase credentials (you add this)** |

## How to Run

```bash
# 1. Install dependencies
pip install sentence-transformers scikit-learn pandas numpy torch firebase-admin flask

# 2. Train the model (first time only)
python prepare_model.py

# 3. Start the web app
python app.py
# → http://localhost:5000
```

## Firebase Integration (Optional)

The app saves chat history to **Firebase Firestore** automatically when configured:

1. Go to [Firebase Console](https://console.firebase.google.com) → Create project
2. Firestore → Create database (start in **test mode**)
3. Project Settings → **Service Accounts** → Generate new private key
4. Save the JSON file as `serviceAccountKey.json` in the project root
5. Restart the Flask server

Without `serviceAccountKey.json`, the chatbot still works — chat history is not persisted between page refreshes.

**Firestore data structure:**
```
sessions/{session_id}/
  - messages: [
      { role: "user", content: "...", timestamp: Timestamp },
      { role: "bot", content: "...", timestamp: Timestamp, results: [...] }
    ]
  - created_at: Timestamp
  - updated_at: Timestamp
```

## File Details

### `chatbot_model.py`
- `TriangularFuzzyLayer` — adaptive fuzzy membership functions (3 sets per feature)
- `TFNN` — FuzzyLayer → Dense(128) → ReLU → Dropout → Dense(64) → ReLU → Dropout → Sigmoid
- `make_features()` — concatenates `[ctx, resp, |ctx-resp|, ctx*resp]` (1536-dim)
- `jawab_chatbot()` — inference: encode question, score all responses, return top-K

### `prepare_model.py`
- Loads CSV, creates embeddings with `paraphrase-multilingual-MiniLM-L12-v2`
- Builds positive/negative pairs, trains TFNN for 30 epochs
- Saves: `model_weights.pth`, `embeddings.npz`, `model_data.pkl`

### `app.py`
- Loads artifacts on startup
- Routes: `/` (chat UI), `/chat` (POST inference), `/history` (GET messages), `/new-session` (POST)
- Firestore save/load with session cookies (no login required)

### `templates/index.html`
- Tailwind CSS via CDN, responsive design
- Welcome message + suggestion chips on first visit
- Loads previous chat history from Firestore on page load
- "New Chat" button to start fresh session (history preserved in Firestore)
- Typing animation, error handling, XSS protection

## Dataset

- **`terjemah.csv`** — 9,061 rows of mental health Q&A pairs
- Columns: `Context`, `Response`, `Context_ID`, `Response_ID`, `Context_clean`, `Response_clean`
- `Context_clean` / `Response_clean` = preprocessed English text (for embeddings)
- `Response_ID` = Indonesian translation (displayed to user)
