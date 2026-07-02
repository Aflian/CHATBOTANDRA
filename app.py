"""
app.py — Flask Web Chatbot (TFNN) + Firebase Firestore
========================================================
Menyimpan riwayat chat ke Firestore (tanpa login).

Local:
    python app.py
    lalu buka http://localhost:5000

Firebase:
    1. Buka https://console.firebase.google.com
    2. Buat project → Firestore → Buat database (mode test)
    3. Project Settings → Service Accounts → Generate key
    4. Di Render: set FIREBASE_CREDENTIALS (base64) sebagai env var
       Atau lokal: simpan sebagai serviceAccountKey.json
"""

import base64
import json
import os
import pickle
import uuid
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, session
from sentence_transformers import SentenceTransformer

from chatbot_model import TFNN, jawab_chatbot

# ── Firebase ─────────────────────────────────────
FIREBASE_AVAILABLE = False
db = None

try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    # Coba dari env var FIREBASE_CREDENTIALS (Render/production)
    env_creds = os.environ.get("FIREBASE_CREDENTIALS")
    if env_creds:
        try:
            cred_dict = json.loads(base64.b64decode(env_creds))
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            FIREBASE_AVAILABLE = True
            print("[OK] Firebase connected (from env)")
        except Exception as e:
            print(f"[!] Firebase env creds failed: {e}")

    # Fallback: serviceAccountKey.json (local dev)
    if not FIREBASE_AVAILABLE:
        SA_PATH = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
        if os.path.exists(SA_PATH):
            cred = credentials.Certificate(SA_PATH)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            FIREBASE_AVAILABLE = True
            print("[OK] Firebase Firestore connected (from file)")
        else:
            print("[!] serviceAccountKey.json tidak ditemukan.")
            print("  Chat tetap jalan tanpa penyimpanan riwayat.")
except Exception as e:
    print(f"[!] Firebase init failed: {e}")
    print("  Chat tetap jalan tanpa penyimpanan riwayat.")

# ── Konfigurasi Flask ────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())
app.config["SESSION_COOKIE_NAME"] = "mindcare_session"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

# ── Artifacts ────────────────────────────────────
ARTIFACTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_weights.pth")
EMB_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.npz")
DATA_PATH = os.path.join(ARTIFACTS_DIR, "model_data.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global state ─────────────────────────────────
model = None
embedder = None
resp_emb = None
responses_display = None
is_ready = False


# ── Session helpers ──────────────────────────────

def get_session_id():
    """Dapatkan session ID dari cookie, buat baru jika belum ada."""
    session.permanent = True
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


# ── Firebase helpers ─────────────────────────────

def save_to_firebase(session_id, role, content, results=None):
    """Simpan satu pesan ke Firestore."""
    if not FIREBASE_AVAILABLE or db is None:
        return

    try:
        doc_ref = db.collection("sessions").document(session_id)
        now = datetime.utcnow().isoformat()
        message = {
            "role": role,
            "content": content,
            "timestamp": now,
        }
        if results is not None:
            message["results"] = results

        doc = doc_ref.get()
        if doc.exists:
            doc_ref.update({
                "messages": firestore.ArrayUnion([message]),
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
        else:
            # Pertama: buat document dulu dengan field kosong
            doc_ref.set({
                "messages": [],
                "created_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            # Kedua: tambah pesan via update (ArrayUnion)
            doc_ref.update({
                "messages": firestore.ArrayUnion([message]),
            })
    except Exception as e:
        print(f"[!] Firebase save error: {e}")


def load_from_firebase(session_id):
    """Muat riwayat pesan dari Firestore."""
    if not FIREBASE_AVAILABLE or db is None:
        return []

    try:
        doc_ref = db.collection("sessions").document(session_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return data.get("messages", [])
    except Exception as e:
        print(f"[!] Firebase load error: {e}")
    return []


# ── Load artifacts ───────────────────────────────

def load_artifacts():
    """Load semua artifacts. Return True jika sukses."""
    global model, embedder, resp_emb, responses_display, is_ready

    if not all(os.path.exists(p) for p in [MODEL_PATH, EMB_PATH, DATA_PATH]):
        return False

    try:
        with open(DATA_PATH, "rb") as f:
            data = pickle.load(f)
        responses_display = data["responses_display"]
        embedder_name = data.get("embedder_name", "paraphrase-multilingual-MiniLM-L12-v2")

        emb_data = np.load(EMB_PATH)
        resp_emb = emb_data["resp_emb"]

        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model = TFNN(
            in_dim=checkpoint["in_dim"],
            n_fuzzy=checkpoint.get("n_fuzzy", 3),
            hidden=checkpoint.get("hidden", 128),
            dropout=checkpoint.get("dropout", 0.3),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        embedder = SentenceTransformer(embedder_name, device=str(device))

        is_ready = True
        print(f"[OK] Model loaded   ({sum(p.numel() for p in model.parameters()):,} params)")
        print(f"[OK] Embedder       ({embedder_name})")
        print(f"[OK] Embeddings     ({len(resp_emb)} x {resp_emb.shape[1]} dim)")
        print(f"[OK] Device         ({device})")
        return True
    except Exception as e:
        print(f"[ERR] Gagal load artifacts: {e}")
        return False


# ── Routes ───────────────────────────────────────

@app.route("/")
def index():
    if not is_ready:
        return render_template("index.html", not_ready=True)
    return render_template("index.html", not_ready=False,
                           firebase_ready=FIREBASE_AVAILABLE)


@app.route("/chat", methods=["POST"])
def chat():
    if not is_ready:
        return jsonify({"error": "Model belum siap. Jalankan prepare_model.py dulu."}), 503

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Pesan tidak boleh kosong."}), 400

    pesan = data["message"].strip()
    if not pesan:
        return jsonify({"error": "Pesan tidak boleh kosong."}), 400

    try:
        hasil = jawab_chatbot(
            pertanyaan=pesan,
            embedder=embedder,
            resp_emb=resp_emb,
            responses_display=responses_display,
            model=model,
            device=device,
            top_k=3,
        )

        # Simpan ke Firestore
        sid = get_session_id()
        save_to_firebase(sid, "user", pesan)
        save_to_firebase(sid, "bot",
                         f"Menampilkan 3 jawaban teratas",
                         results=[{"rank": h["rank"], "skor": h["skor"],
                                    "jawaban": h["jawaban"]} for h in hasil])

        return jsonify({"success": True, "hasil": hasil,
                        "firebase": FIREBASE_AVAILABLE})

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500


@app.route("/history", methods=["GET"])
def history():
    """Muat riwayat chat session ini."""
    if not is_ready:
        return jsonify({"messages": []})

    sid = get_session_id()
    msgs = load_from_firebase(sid)
    return jsonify({"messages": msgs, "session_id": sid,
                    "firebase": FIREBASE_AVAILABLE})


@app.route("/new-session", methods=["POST"])
def new_session():
    """Hapus session ID → mulai session baru."""
    session.pop("session_id", None)
    return jsonify({"success": True})


# ── Startup (module-level — jalan juga di gunicorn) ──

print("=" * 50)
print("[TFNN Chatbot] Flask + Firebase")
print("=" * 50)
print("\nLoading artifacts...")
load_artifacts()
print()

if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
