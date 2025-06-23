import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from app.models.segmentador_cnn import SegmentadorCNN
from app.metrics import compute_iou, compute_f1_score, compute_accuracy

# Configuración
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas
PATCHES_PATH = "data/tensors/patches_input.pt"
MASKS_PATH = "data/tensors/masks.pt"
MODEL_OUTPUT = "data/outputs/modelo_entrenado.pth"
PREDICTIONS_DIR = "data/outputs/predicciones"

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# 1. Cargar datos
X = torch.load(PATCHES_PATH)
Y = torch.load(MASKS_PATH).unsqueeze(1).float()  # (B, 1, 128, 128)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2. Modelo
model = SegmentadorCNN().to(DEVICE)

# 3. Optimización
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 4. Entrenamiento
print("🚀 Iniciando entrenamiento...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        out = model(xb)  # (B, 1, 128, 128)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"📚 Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# 5. Guardar modelo
torch.save(model.state_dict(), MODEL_OUTPUT)
print(f"✅ Modelo guardado en {MODEL_OUTPUT}")

# 6. Evaluación general
print("🧪 Evaluando modelo completo...")
model.eval()
with torch.no_grad():
    preds = model(X.to(DEVICE)).cpu()
    preds_bin = (torch.sigmoid(preds) > 0.5).int()

    iou = compute_iou(preds, Y.int())
    f1 = compute_f1_score(preds, Y.int())
    acc = compute_accuracy(preds, Y.int())

    print(f"📊 Métricas -> IoU: {iou:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}")

# 7. Guardar predicciones binarizadas por parche
print("💾 Guardando predicciones binarizadas...")
for i, pred in enumerate(preds_bin.squeeze(1)):
    torch.save(pred, os.path.join(PREDICTIONS_DIR, f"pred_patch_{i:03d}.pt"))

print("🏁 Entrenamiento y predicción finalizados.")
