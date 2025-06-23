#!/usr/bin/env python
"""
Reconstruye una máscara completa (1, H, W) desde masks.pt (B, 128, 128)
y la guarda como full_mask.pt para entrenamiento global.
"""

import os, math, torch

MASKS_PATH = "data/tensors/masks.pt"
COLS       = 4  # ← AJUSTA según tu mosaico original
PATCH_SIZE = 128
OUT_PATH   = "data/inputs/full_mask.pt"

def main():
    masks = torch.load(MASKS_PATH)  # (B, 128, 128)
    B, H, W = masks.shape
    rows = math.ceil(B / COLS)

    full_mask = torch.zeros((1, rows * PATCH_SIZE, COLS * PATCH_SIZE))  # (1, H, W)

    for idx in range(B):
        r, c = divmod(idx, COLS)
        full_mask[0,
                  r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                  c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = masks[idx]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(full_mask, OUT_PATH)
    print("✅ Máscara completa guardada en:", OUT_PATH)
    print("   Shape:", tuple(full_mask.shape))

if __name__ == "__main__":
    main()
