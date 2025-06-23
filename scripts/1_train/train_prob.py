#!/usr/bin/env python
"""
Entrena EncoderCNN + DecoderLSTM y guarda la probabilidad por píxel.

 ▸ Lee:  data/tensors/patches_input.pt   (B, 10, 128, 128)
         data/tensors/masks.pt          (B, 1, 128, 128)
 ▸ Guarda:
         data/models/modelo_encoder_decoder.pth
         data/outputs/prob_patches/prob_patch_000.pt …
"""

import os, datetime, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from app.models.encoder_cnn import EncoderCNN
from app.models.decoder_lstm import DecoderLSTM
from app.metrics import compute_iou, compute_f1_score, compute_accuracy

# ───── Paths ──────────────────────────────────────────────
PATCHES_PATH   = "data/tensors/patches_input.pt"
MASKS_PATH     = "data/tensors/masks.pt"

DATE           = datetime.datetime.now().strftime("%Y%m%d")
MODEL_OUT      = f"data/models/modelo_encoder_decoder_{DATE}.pth"
PROB_DIR       = "data/outputs/prob_patches"

os.makedirs("data/models",     exist_ok=True)
os.makedirs(PROB_DIR,          exist_ok=True)

# ───── Hiperparámetros ───────────────────────────────────
BATCH_SIZE = 8
EPOCHS     = 30
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───── Dataset ───────────────────────────────────────────
X = torch.load(PATCHES_PATH)                         # (B, 10, 128, 128)
Y = torch.load(MASKS_PATH).unsqueeze(1).float()      # (B, 1, 128, 128)

loader = DataLoader(TensorDataset(X, Y),
                    batch_size=BATCH_SIZE,
                    shuffle=True)

# ───── Modelos y optimizador ─────────────────────────────
encoder = EncoderCNN().to(DEVICE)
decoder = DecoderLSTM(feature_dim=131072).to(DEVICE)

params = list(encoder.parameters()) + list(decoder.parameters())
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params, lr=LR)

# ───── Entrenamiento ─────────────────────────────────────
print(f"🚀 Training on {DEVICE} …")
for epoch in range(1, EPOCHS + 1):
    encoder.train(); decoder.train()
    epoch_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)   # xb: (B,10,128,128)

        optimizer.zero_grad()
        feature_seq = []

        # Paso por encoder en cada paso temporal
        for t in range(xb.shape[1]):  # loop sobre los 10 pasos (meses)
            xt = xb[:, t].unsqueeze(1)           # (B,1,128,128)
            ft = encoder(xt)                     # (B, 128, 32, 32)
            ft = ft.view(ft.size(0), -1)         # (B, 131072)
            feature_seq.append(ft)

        feature_seq_tensor = torch.stack(feature_seq, dim=1)  # (B, T, Features)

        logits = decoder(feature_seq_tensor)  # (B, 1, 128, 128)
        loss   = criterion(logits, yb)
        loss.backward(); optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch:02d}/{EPOCHS}  Loss = {epoch_loss/len(loader):.4f}")

# ───── Guardar pesos ─────────────────────────────────────
torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict()
}, MODEL_OUT)
print("✅ Modelos guardados →", MODEL_OUT)

# ───── Inferencia global y métricas ──────────────────────
print("🧪 Evaluando …")
encoder.eval(); decoder.eval()
with torch.no_grad():
    features_all = []

    for t in range(X.shape[1]):  # loop sobre los 10 pasos temporales
        xt = X[:, t].unsqueeze(1).to(DEVICE)     # (B,1,128,128)
        ft = encoder(xt)
        ft = ft.view(ft.size(0), -1)             # (B, Features)
        features_all.append(ft)

    features_seq_tensor = torch.stack(features_all, dim=1)  # (B, T, Features)
    logits_all = decoder(features_seq_tensor).cpu()         # (B,1,128,128)
    prob_all   = torch.sigmoid(logits_all)

    bin_all = (prob_all > 0.5).int()
    iou = compute_iou(bin_all, Y.int())
    f1  = compute_f1_score(bin_all, Y.int())
    acc = compute_accuracy(bin_all, Y.int())
    print(f"IoU={iou:.4f}  F1={f1:.4f}  Acc={acc:.4f}")

# ───── Guardar probabilidades por parche ─────────────────
print("💾 Guardando prob_patch_###.pt …")
for i, prob in enumerate(prob_all.squeeze(1)):     # (B,H,W)
    torch.save(prob, os.path.join(PROB_DIR, f"prob_patch_{i:03d}.pt"))

print("🏁 Fin del entrenamiento.")
