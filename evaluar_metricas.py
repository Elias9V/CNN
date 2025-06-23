import os
import torch
from app.metrics import compute_accuracy, compute_iou, compute_f1_score

# Rutas
PRED_DIR = "data/outputs/predicciones"
MASKS_PATH = "data/tensors/masks.pt"

# 1. Cargar ground truth
print("📦 Cargando máscaras verdaderas...")
masks = torch.load(MASKS_PATH).int()  # (B, 128, 128)

# 2. Cargar predicciones
print("📦 Cargando predicciones...")
archivos = sorted([f for f in os.listdir(PRED_DIR) if f.endswith(".pt")])
preds = []

for i, f in enumerate(archivos):
    pred = torch.load(os.path.join(PRED_DIR, f)).squeeze()

    # Asegurar que esté binarizado
    if pred.max() > 1.0:
        pred = (torch.sigmoid(pred) > 0.5).int()
    else:
        pred = pred.int()

    preds.append(pred)

# 3. Convertir a tensor
preds_tensor = torch.stack(preds)  # (B, 128, 128)

# 4. Evaluación
print("\n📊 Resultados de Evaluación:")
print(f"- Accuracy : {compute_accuracy(preds_tensor.unsqueeze(1), masks.unsqueeze(1)):.4f}")
print(f"- IoU      : {compute_iou(preds_tensor.unsqueeze(1), masks.unsqueeze(1)):.4f}")
print(f"- F1-score : {compute_f1_score(preds_tensor.unsqueeze(1), masks.unsqueeze(1)):.4f}")
