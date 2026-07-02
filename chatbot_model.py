"""
chatbot_model.py — TFNN Model & Inference Utilities
====================================================
Model definition, feature engineering, and inference logic
extracted from the original fix_chatbot.py Colab notebook.
"""

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
#  Triangular Fuzzy Layer (Neuro-Fuzzy)
# ──────────────────────────────────────────────

class TriangularFuzzyLayer(nn.Module):
    """
    Fuzzifikasi tiap fitur menjadi derajat keanggotaan terhadap K himpunan
    fuzzy bermembership segitiga. Center & width dilatih adaptif (neuro-fuzzy).
    """
    def __init__(self, in_dim, n_fuzzy=3):
        super().__init__()
        self.in_dim = in_dim
        self.n_fuzzy = n_fuzzy
        centers = torch.linspace(-1.0, 1.0, n_fuzzy).repeat(in_dim, 1)
        self.centers = nn.Parameter(centers)                   # (in_dim, n_fuzzy)
        self.widths  = nn.Parameter(torch.ones(in_dim, n_fuzzy) * 0.5)

    def forward(self, x):
        x = x.unsqueeze(-1)                                    # (batch, in_dim, 1)
        c = self.centers.unsqueeze(0)                          # (1, in_dim, n_fuzzy)
        w = torch.clamp(self.widths.unsqueeze(0), min=1e-4)
        membership = torch.clamp(1.0 - torch.abs(x - c) / w, min=0.0)
        return membership.reshape(x.size(0), -1)               # (batch, in_dim*n_fuzzy)


class TFNN(nn.Module):
    """Takagi-Sugeno-Kang inspired Neuro-Fuzzy Network."""
    def __init__(self, in_dim, n_fuzzy=3, hidden=128, dropout=0.3):
        super().__init__()
        self.fuzzy = TriangularFuzzyLayer(in_dim, n_fuzzy)
        fuzzy_out = in_dim * n_fuzzy
        self.net = nn.Sequential(
            nn.Linear(fuzzy_out, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        f = self.fuzzy(x)
        score = self.net(f)                       # logit
        return torch.sigmoid(score).squeeze(-1)   # defuzzifikasi → skor [0,1]


# ──────────────────────────────────────────────
#  Feature Engineering
# ──────────────────────────────────────────────

def make_features(Xc, Xr):
    """
    Gabungkan embedding context & response beserta informasi interaksi.

    Returns: [ctx, resp, |ctx-resp|, ctx*resp]  →  (N, 4 * emb_dim)
    """
    return np.concatenate([Xc, Xr, np.abs(Xc - Xr), Xc * Xr], axis=1)


# ──────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────

def jawab_chatbot(pertanyaan, embedder, resp_emb, responses_display,
                  model, device, top_k=3):
    """
    Cari top-K response paling relevan untuk pertanyaan user.

    Parameters
    ----------
    pertanyaan : str — input user
    embedder   : SentenceTransformer — model embedding
    resp_emb   : np.ndarray (N, emb_dim) — precomputed response embeddings
    responses_display : list[str] — response text untuk ditampilkan (Bahasa Indonesia)
    model      : TFNN — trained model
    device     : torch.device
    top_k      : int — jumlah jawaban yang dikembalikan

    Returns
    -------
    list[dict] — [{"skor": float, "jawaban": str}, ...] sorted descending
    """
    q_emb = embedder.encode([pertanyaan], convert_to_numpy=True,
                            normalize_embeddings=True)
    q_rep = np.repeat(q_emb, len(resp_emb), axis=0)
    feats = make_features(q_rep, resp_emb)

    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(feats, dtype=torch.float32).to(device))
        scores = scores.cpu().numpy()

    idx = scores.argsort()[::-1][:top_k]

    hasil = []
    for rank, i in enumerate(idx, 1):
        hasil.append({
            "rank": rank,
            "skor": float(scores[i]),
            "jawaban": responses_display[i],
        })
    return hasil
