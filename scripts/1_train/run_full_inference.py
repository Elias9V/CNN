#!/usr/bin/env python
"""
Genera la predicción completa sobre una imagen (10, H, W) usando
EncoderCNN + DecoderLSTM y guarda los resultados .npy y .png.
"""

import os, torch
import numpy as np
import matplotlib.pyplot as plt
from app.models.encoder_cnn import EncoderCNN
from app.models.decoder_lstm import DecoderLSTM

# ───── Rutas ─────────────────────────────────────────────
INPUT_PATH  = "data/inputs/full_image.pt"   # (10, H, W)
MODEL_PATH  = "data/models/modelo_encoder_decoder_full_20250623.pth"  # ← cambia fecha si es necesario
OUT_NPY     = "data/outputs/full_pred.npy"
OUT_PNG     = "data/outputs/full_pred.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───── Cargar datos ──────────────────────────────────────
X = torch.load(INPUT_PATH).to(DEVICE)  # (10, H, W)
T, H, W = X.shape

# ───── Cargar modelos ────────────────────────────────────
encoder = EncoderCNN().to(DEVICE)

# Obtener feature_dim
with torch.no_grad():
    ft_test, _ = encoder(X[0].unsqueeze(0).unsqueeze(0))
    feature_dim = ft_test.view(1, -1).shape[-1]

decoder = DecoderLSTM(feature_dim=feature_dim).to(DEVICE)

# Cargar pesos
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])
encoder.eval(); decoder.eval()

# ───── Inference ─────────────────────────────────────────
with torch.no_grad():
    feat_seq = []
    for t in range(T):
        xt = X[t].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        ft, _ = encoder(xt)
        ft = ft.view(1, -1)
        feat_seq.append(ft)

    input_seq = torch.stack(feat_seq, dim=1)  # (1,T,F)
    logits = decoder(input_seq, output_size=(H, W))
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()

# ───── Guardar salida ────────────────────────────────────
np.save(OUT_NPY, prob)
plt.imsave(OUT_PNG, prob, cmap="turbo")
print(f"✅ Predicción completa guardada en:\n   → {OUT_NPY}\n   → {OUT_PNG}")
