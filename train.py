import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from app.dataset import LandslideDataset
from app.models.encoder_cnn import EncoderCNN
from app.models.decoder_lstm import DecoderLSTM
from app.metrics import compute_accuracy, compute_iou, compute_f1

# 🔧 Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
MODEL_PATH = "data/models/best_model.pth"

# 📦 Dataset de validación
val_dataset = LandslideDataset("data/val_inputs.pt", "data/val_masks.pt")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 🧠 Modelos
encoder = EncoderCNN().to(device)
decoder = DecoderLSTM().to(device)

# 💾 Cargar pesos entrenados
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    print("📦 Modelo anterior cargado desde:", MODEL_PATH)
else:
    print("🆕 No se encontró modelo previo. Entrenamiento desde cero.")


encoder.eval()
decoder.eval()

# 📊 Métricas acumuladas
all_acc, all_iou, all_f1 = [], [], []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        features = encoder(inputs)                      # (B,128,32,32)
        seq = features.view(features.size(0), 1024, -1) # (B,1024,128)
        outputs = decoder(seq)                          # (B,2,128,128)

        all_acc.append(compute_accuracy(outputs, targets))
        all_iou.append(compute_iou(outputs, targets))
        all_f1.append(compute_f1(outputs, targets))

# 📢 Reporte final
print("✅ EVALUACIÓN FINAL")
print(f"Accuracy promedio: {np.mean(all_acc):.4f}")
print(f"IoU promedio: {np.mean(all_iou):.4f}")
print(f"F1-score promedio: {np.mean(all_f1):.4f}")

# 🎨 Visualizar primera predicción
preds = torch.argmax(outputs, dim=1)
pred_img = preds[0].cpu().numpy()

plt.imshow(pred_img, cmap='inferno')
plt.title("Predicción ejemplo (clase)")
plt.colorbar()
plt.savefig("plots/prediccion_ejemplo.png")
plt.close()
