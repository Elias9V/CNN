#!/usr/bin/env python
"""
Entrena EncoderCNN + DecoderLSTM sobre una imagen completa (10, H, W)
y guarda el modelo y su predicción.
"""

import os, datetime, torch
import torch.nn as nn
import torch.optim as optim
from app.models.encoder_cnn import EncoderCNN
from app.models.decoder_lstm import DecoderLSTM
import numpy as np
import matplotlib.pyplot as plt

# ───── Rutas y Configuración ────────────────────────────
INPUT_PATH = "data/inputs/full_image.pt"  # tensor de shape (10, H, W)
MASK_PATH  = "data/inputs/full_mask.pt"   # tensor de shape (1, H, W)
DATE       = datetime.datetime.now().strftime("%Y%m%d")
MODEL_OUT  = f"data/models/modelo_encoder_decoder_full_{DATE}.pth"
PNG_OUT    = f"data/outputs/full_train_pred.png"
NPY_OUT    = f"data/outputs/full_train_pred.npy"

LR       = 1e-4
EPOCHS   = 50
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───── Cargar datos ──────────────────────────────
X = torch.load(INPUT_PATH).to(DEVICE)     # (10, H, W)
Y = torch.load(MASK_PATH).unsqueeze(0).to(DEVICE).float()  # (1, H, W)
T, H, W = X.shape

# ───── Modelos ───────────────────────────────────────
encoder = EncoderCNN().to(DEVICE)

# Calcular feature_dim automáticamente
with torch.no_grad():
    ft_test, _ = encoder(X[0].unsqueeze(0).unsqueeze(0))  # (1, C, h, w)
    feature_dim = ft_test.view(1, -1).shape[-1]

decoder = DecoderLSTM(feature_dim=feature_dim).to(DEVICE)

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=LR)
criterion = nn.BCEWithLogitsLoss()

# ───── Entrenamiento ──────────────────────────────
print(f"🚀 Entrenando imagen completa ({H}x{W}) en {DEVICE}")
for epoch in range(1, EPOCHS + 1):
    encoder.train(); decoder.train()
    optimizer.zero_grad()

    # Construir secuencia de características
    feat_seq = []
    for t in range(T):
        xt = X[t].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        ft, _ = encoder(xt)
        ft = ft.view(1, -1)                  # (1,F)
        feat_seq.append(ft)

    feat_tensor = torch.stack(feat_seq, dim=1)  # (1,T,F)
    logits = decoder(feat_tensor, output_size=(H, W))
    loss = criterion(logits, Y)
    loss.backward(); optimizer.step()

    print(f"Epoch {epoch:03d}/{EPOCHS}  Loss = {loss.item():.6f}")

# ───── Guardar modelo ─────────────────────────────
torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict()
}, MODEL_OUT)
print("✅ Modelo guardado →", MODEL_OUT)

# ───── Evaluación y Visualización ────────────────────────
encoder.eval(); decoder.eval()
with torch.no_grad():
    sequence = []
    for t in range(T):
        xt = X[t].unsqueeze(0).unsqueeze(0)
        ft, _ = encoder(xt)
        ft = ft.view(1, -1)
        sequence.append(ft)

    input_seq = torch.stack(sequence, dim=1)
    logits = decoder(input_seq, output_size=(H, W))
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()

np.save(NPY_OUT, prob)
plt.imsave(PNG_OUT, prob, cmap="turbo")
print(f"🖼️  Predicción guardada en:\n   → {NPY_OUT}\n   → {PNG_OUT}")
