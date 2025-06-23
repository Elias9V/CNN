#!/usr/bin/env python
"""
Reconstruye una imagen completa (10, H, W) desde los parches de entrada
y la guarda como full_image.pt para entrenamientos completos.
NO usa el modelo aún.
"""

import os, math, torch

PATCHES_PATH = "data/tensors/patches_input.pt"      # (B, 10, 128, 128)
OUT_PATH     = "data/inputs/full_image.pt"
COLS         = 4                                    # ← AJUSTA si usaste otro
PATCH_SIZE   = 128

def main():
    patches = torch.load(PATCHES_PATH)  # (B, 10, 128, 128)
    B, T, H, W = patches.shape
    rows = math.ceil(B / COLS)

    # Crear tensor completo: (10, H_total, W_total)
    full_tensor = torch.zeros((T, rows * PATCH_SIZE, COLS * PATCH_SIZE))

    for idx in range(B):
        r, c = divmod(idx, COLS)
        for t in range(T):
            full_tensor[t,
                        r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                        c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = patches[idx, t]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(full_tensor, OUT_PATH)
    print("✅ Imagen completa guardada en:", OUT_PATH)
    print("   Shape:", tuple(full_tensor.shape))  # (10, H, W)

if __name__ == "__main__":
    main()
