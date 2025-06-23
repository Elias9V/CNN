import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Ruta donde están las predicciones
PRED_DIR = "data/outputs/predicciones"

# Obtener todos los archivos pred_patch_*.pt ordenados
archivos = sorted([f for f in os.listdir(PRED_DIR) if f.endswith(".pt")])

# Mostrar hasta 16 por cuadrícula (ajustable)
MAX = 16
cols = 4
rows = int(np.ceil(min(MAX, len(archivos)) / cols))

plt.figure(figsize=(12, 3 * rows))
for i, archivo in enumerate(archivos[:MAX]):
    path = os.path.join(PRED_DIR, archivo)
    pred = torch.load(path).numpy()

    plt.subplot(rows, cols, i + 1)
    plt.imshow(pred, cmap="gray")
    plt.title(archivo)
    plt.axis("off")

plt.tight_layout()
plt.suptitle("Predicciones Binarias (Riesgo vs No riesgo)", fontsize=16, y=1.02)
plt.show()