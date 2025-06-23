#!/usr/bin/env python
"""
Une todos los prob_patch_###.pt en un único mosaico y lo guarda
como NumPy (float32 0-1).

Lee  : data/outputs/prob_patches/prob_patch_000.pt ...
Guarda: data/outputs/mosaico_prob.npy
"""

import os, math, numpy as np, torch

PATCH_DIR   = "data/outputs/prob_patches"
PATCH_SIZE  = 128
COLS        = 4                                   # nº parches por fila
OUT_NPY     = "data/outputs/mosaico_prob.npy"

def main():
    files = sorted(f for f in os.listdir(PATCH_DIR) if f.endswith(".pt"))
    if not files:
        raise RuntimeError(f"No hay *.pt en {PATCH_DIR}")

    rows = math.ceil(len(files) / COLS)
    canvas = np.zeros((rows*PATCH_SIZE, COLS*PATCH_SIZE), dtype=np.float32)

    for idx, fname in enumerate(files):
        prob = torch.load(os.path.join(PATCH_DIR, fname)).squeeze().numpy()
        r, c = divmod(idx, COLS)
        canvas[r*PATCH_SIZE:(r+1)*PATCH_SIZE, c*PATCH_SIZE:(c+1)*PATCH_SIZE] = prob

    os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)
    np.save(OUT_NPY, canvas)
    print("✅ Mosaico de probabilidades guardado →", OUT_NPY)

if __name__ == "__main__":
    main()
