import os
import torch

# Ruta donde están las predicciones
PRED_DIR = "data/outputs/predicciones"

# Leer archivos .pt
archivos = sorted([f for f in os.listdir(PRED_DIR) if f.endswith(".pt")])
if not archivos:
    print("❌ No se encontraron predicciones")
    exit()

total_pixeles = 0
total_riesgo = 0

print("📋 Conteo de píxeles de riesgo por parche:\n")

for nombre in archivos:
    path = os.path.join(PRED_DIR, nombre)
    pred = torch.load(path)

    # Si son logits, aplicar sigmoide y binarizar
    if pred.max() > 1.0:
        pred = (torch.sigmoid(pred) > 0.5).int()
    else:
        pred = pred.int()

    num_pixeles = pred.numel()
    num_riesgo = (pred == 1).sum().item()
    porcentaje = 100 * num_riesgo / num_pixeles

    total_pixeles += num_pixeles
    total_riesgo += num_riesgo

    print(f"{nombre}: {num_riesgo} / {num_pixeles} píxeles ({porcentaje:.2f}%)")

print("\n📊 Resultado total:")
print(f"Total riesgo: {total_riesgo} / {total_pixeles} ({100 * total_riesgo / total_pixeles:.2f}%)")
