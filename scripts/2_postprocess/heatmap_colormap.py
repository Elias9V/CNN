#!/usr/bin/env python
"""
Genera un PNG a color con degradado de probabilidad de riesgo.
Lee:
    data/outputs/mosaico_prob.npy      # float32 (H,W) en [0,1]

Guarda:
    data/outputs/heatmap_riesgo.png
"""

import numpy as np, cv2, os
from PIL import Image

PROB_NPY  = "data/outputs/mosaico_prob.npy"
PNG_OUT   = "data/outputs/heatmap_riesgo.png"

COLORMAP  = cv2.COLORMAP_TURBO        # azul-rojo gradiente
BORDER_CLR_BGR = (0, 255, 255)        # amarillo en BGR (cv2)

# ── 1. Cargar probas ────────────────────────────────────
if not os.path.exists(PROB_NPY):
    raise FileNotFoundError(f"No existe {PROB_NPY}. Ejecuta reconstruir_prob_mosaico.py primero.")

prob = np.load(PROB_NPY)              # (H,W) float32 [0-1]

# ── 2. Convertir a 0-255 uint8 para cv2 ─────────────────
prob_8u = (prob * 255).clip(0, 255).astype(np.uint8)

# ── 3. Aplicar colormap TURBO (BGR) y pasar a RGB ───────
heat_bgr = cv2.applyColorMap(prob_8u, COLORMAP)
heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

# ── 4. Dibujar bordes amarillo sobre prob > 0.5 —────────
mask_bin = (prob >= 0.5).astype(np.uint8) * 255
edges    = cv2.Canny(mask_bin, 80, 160)                # or ajusta
heat_rgb[edges > 0] = BORDER_CLR_BGR[::-1]             # BGR → RGB

# ── 5. Guardar PNG ──────────────────────────────────────
os.makedirs(os.path.dirname(PNG_OUT), exist_ok=True)
Image.fromarray(heat_rgb).save(PNG_OUT)
print("✅ Heat-map guardado en:", PNG_OUT)
