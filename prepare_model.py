"""
prepare_model.py — Training TFNN & Menyimpan Artifacts
=======================================================
Jalankan SEKALI untuk melatih model dan menyimpan:
  - model_weights.pth    (bobot TFNN)
  - embeddings.npz       (embedding response)
  - model_data.pkl       (metadata, response text dll)

Usage:
    python prepare_model.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from chatbot_model import TFNN, make_features

# ── Konfigurasi ──────────────────────────────────
CSV_PATH = "terjemah.csv"
SEED = 42
TEST_SIZE = 0.2
N_NEG = 1
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ── 1. Load Dataset ──────────────────────────────
print("=" * 50)
print("1. Loading dataset...")
print("=" * 50)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["Context_clean", "Response_clean"]).reset_index(drop=True)
print(f"Jumlah pasangan (after dropna): {len(df)}")

# Untuk embedding → gunakan Context_clean & Response_clean (English, preprocessed)
# Untuk display  → Response_ID (Bahasa Indonesia)
contexts = df["Context_clean"].astype(str).tolist()
responses = df["Response_clean"].astype(str).tolist()
responses_display = df["Response_ID"].astype(str).tolist()

# ── 2. Buat Embedding ────────────────────────────
print("\n" + "=" * 50)
print("2. Creating embeddings... (bisa makan waktu ~5-10 menit)")
print("=" * 50)

embedder = SentenceTransformer(EMBEDDER_NAME, device=str(device))

ctx_emb = embedder.encode(contexts, convert_to_numpy=True,
                           show_progress_bar=True, normalize_embeddings=True)
resp_emb = embedder.encode(responses, convert_to_numpy=True,
                            show_progress_bar=True, normalize_embeddings=True)
EMB_DIM = ctx_emb.shape[1]
print(f"Dimensi embedding: {EMB_DIM}")

# ── 3. Bangun Pair (Positif + Negatif) ───────────
print("\n" + "=" * 50)
print("3. Building training pairs...")
print("=" * 50)

def build_pairs(ctx_emb, resp_emb, n_neg=1, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(ctx_emb)
    X_ctx, X_resp, y = [], [], []
    for i in range(n):
        X_ctx.append(ctx_emb[i]); X_resp.append(resp_emb[i]); y.append(1.0)
        for _ in range(n_neg):
            j = rng.integers(0, n)
            while j == i:
                j = rng.integers(0, n)
            X_ctx.append(ctx_emb[i]); X_resp.append(resp_emb[j]); y.append(0.0)
    return (np.array(X_ctx, dtype=np.float32),
            np.array(X_resp, dtype=np.float32),
            np.array(y, dtype=np.float32))

Xc, Xr, y = build_pairs(ctx_emb, resp_emb, n_neg=N_NEG)
X = make_features(Xc, Xr)
print(f"Total sampel: {len(y)} | Positif: {int(y.sum())} | Negatif: {int((y == 0).sum())}")
print(f"Dimensi fitur input: {X.shape}")

# ── 4. Train/Test Split ──────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

# ── 5. Inisialisasi Model ────────────────────────
print("\n" + "=" * 50)
print("4. Initializing TFNN model...")
print("=" * 50)

model = TFNN(in_dim=X.shape[1], n_fuzzy=3).to(device)
print(model)

# ── 6. Training ──────────────────────────────────
print("\n" + "=" * 50)
print("5. Training...")
print("=" * 50)

class PairDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_loader = DataLoader(PairDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(PairDataset(X_test, y_test), batch_size=128)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def eval_loss(loader):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total += criterion(model(xb), yb).item() * xb.size(0)
    return total / len(loader.dataset)

history = {"train_loss": [], "val_loss": []}
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_loader.dataset)
    val_loss = eval_loss(test_loader)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ── 7. Evaluasi ──────────────────────────────────
print("\n" + "=" * 50)
print("6. Evaluation...")
print("=" * 50)

model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        all_pred.append(model(xb).cpu().numpy())
        all_true.append(yb.numpy())

y_score = np.concatenate(all_pred)
y_true  = np.concatenate(all_true)
y_pred  = (y_score >= 0.5).astype(int)

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, r2_score)
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_true, y_pred):.4f}")
print(f"R2 Score : {r2_score(y_true, y_score):.4f}")

# ── 8. Simpan Artifacts ──────────────────────────
print("\n" + "=" * 50)
print("7. Saving artifacts...")
print("=" * 50)

# Model weights + config
torch.save({
    'model_state_dict': model.state_dict(),
    'in_dim': X.shape[1],
    'n_fuzzy': 3,
    'hidden': 128,
    'dropout': 0.3,
    'history': history,
}, 'model_weights.pth')
print("  [OK] model_weights.pth")

# Embeddings (hanya resp_emb yang diperlukan untuk inference)
np.savez('embeddings.npz',
         resp_emb=resp_emb,
         ctx_emb=ctx_emb)
print("  [OK] embeddings.npz")

# Metadata
data = {
    'responses_display': responses_display,
    'embedder_name': EMBEDDER_NAME,
    'emb_dim': EMB_DIM,
    'n_samples': len(df),
}
with open('model_data.pkl', 'wb') as f:
    pickle.dump(data, f)
print("  [OK] model_data.pkl")

print("\n" + "=" * 50)
print("[SELESAI] Semua artifacts tersimpan.")
print("Jalankan 'python app.py' untuk memulai web chatbot.")
print("=" * 50)
