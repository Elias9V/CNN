import torch
import os
import random

# Configuración
INPUT_X_PATH = "data/uploads/patches_input.pt"
INPUT_Y_PATH = "data/masks.pt"

TRAIN_X_PATH = "data/train_inputs.pt"
TRAIN_Y_PATH = "data/train_masks.pt"
VAL_X_PATH = "data/val_inputs.pt"
VAL_Y_PATH = "data/val_masks.pt"

SPLIT_RATIO = 0.8  # 80% entrenamiento, 20% validación
SEED = 42

# Cargar tensores
X = torch.load(INPUT_X_PATH)  # (N, 10, 128, 128)
Y = torch.load(INPUT_Y_PATH)  # (N, 128, 128)

assert X.shape[0] == Y.shape[0], "X e Y deben tener mismo número de muestras"
total = X.shape[0]
indices = list(range(total))

# Mezclar de forma reproducible
random.seed(SEED)
random.shuffle(indices)

split = int(SPLIT_RATIO * total)
train_idx = indices[:split]
val_idx = indices[split:]

# Separar
train_X = X[train_idx]
train_Y = Y[train_idx]
val_X = X[val_idx]
val_Y = Y[val_idx]

# Guardar
torch.save(train_X, TRAIN_X_PATH)
torch.save(train_Y, TRAIN_Y_PATH)
torch.save(val_X, VAL_X_PATH)
torch.save(val_Y, VAL_Y_PATH)

print(f"✅ Dataset dividido: {len(train_X)} train, {len(val_X)} val")
