import torch
import numpy as np
from app.models.encoder_cnn import EncoderCNN
from app.models.decoder_lstm import DecoderLSTM
import matplotlib.pyplot as plt
import os

# ───── CONFIGURACIÓN ─────────────────────────────────────
INPUT_PATH = "data/inputs/full_image.pt"       # (10, H, W)
MODEL_PATH = "data/models/modelo_encoder_decoder_full_20250623.pth"
OUTPUT_NPY = "data/outputs/full_image_pred.npy"
OUTPUT_PNG = "data/outputs/full_image_pred.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───── CARGAR MODELO ─────────────────────────────────────
encoder = EncoderCNN().to(DEVICE)
decoder = DecoderLSTM(feature_dim=131072).to(DEVICE)  # Ajusta si usaste otro tamaño

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval(); decoder.eval()

# ───── CARGAR IMAGEN COMPLETA ────────────────────────────
input_tensor = torch.load(INPUT_PATH)  # (10, H, W)
T, H, W = input_tensor.shape
assert T == 10, "La imagen debe tener 10 pasos temporales"

# ───── PASO POR MODELO ───────────────────────────────────
with torch.no_grad():
    feature_seq = []

    for t in range(T):
        xt = input_tensor[t].unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)
        ft = encoder(xt)                # (1, 128, 32, 32)
        ft = ft.view(1, -1)             # (1, 131072)
        feature_seq.append(ft)

    feature_seq_tensor = torch.stack(feature_seq, dim=1)  # (1, T, F)
    logits = decoder(feature_seq_tensor)                  # (1,1,H,W)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H,W)

# ───── GUARDAR RESULTADOS ────────────────────────────────
np.save(OUTPUT_NPY, prob)

# Heatmap visual
plt.imsave(OUTPUT_PNG, prob, cmap="turbo")
print("✅ Predicción guardada como:")
print("   →", OUTPUT_NPY)
print("   →", OUTPUT_PNG)
