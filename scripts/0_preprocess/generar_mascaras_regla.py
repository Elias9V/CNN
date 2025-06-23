#!/usr/bin/env python
"""
Genera máscara binaria 0/1 a partir de la banda de pendiente.
Guarda:  masks.pt   (B,128,128)  int8
         mosaico_mascaras_gt.png
Opcional: mosaico_mascaras_gt.tif (GeoTIFF si rasterio instalado)
"""

import os, math, numpy as np, torch
from PIL import Image

# ─── Rutas ───────────────────────────────────────────────
PATCHES_PATH   = "data/tensors/patches_input.pt"
MASKS_OUT_PT   = "data/tensors/masks.pt"
PNG_OUT_PATH   = "data/outputs/mosaico_mascaras_gt.png"
GEOTIFF_PATH   = "data/outputs/mosaico_mascaras_gt.tif"   # opcional

# ─── Parámetros ──────────────────────────────────────────
UMBRAL_PEND    = 0.33         # umbral de pendiente normalizada
PEND_BAND_IDX  = -1           # -1 = última banda
COLS_MOSAICO   = 4            # nº columnas deseadas en el mosaico

# ─── 1. Máscara por regla ───────────────────────────────
def generar_mascaras(patches: torch.Tensor) -> torch.Tensor:
    pendiente = patches[:, PEND_BAND_IDX].float()              # (B,128,128)
    masks     = (pendiente > UMBRAL_PEND).to(torch.int8)       # 0/1
    return masks

# ─── 2. Exportar mosaico PNG ─────────────────────────────
def exportar_png(masks: torch.Tensor, path: str, cols=COLS_MOSAICO):
    B, H, W = masks.shape
    rows    = math.ceil(B / cols)
    canvas  = np.zeros((rows*H, cols*W), dtype=np.uint8)

    for i in range(B):
        r, c = divmod(i, cols)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = masks[i].numpy() * 255

    Image.fromarray(canvas).save(path)
    print("🖼  PNG guardado →", path)

# ─── 3. (Opcional) GeoTIFF ───────────────────────────────
def exportar_tiff(canvas_uint8: np.ndarray, path: str):
    try:
        import rasterio
        with rasterio.open(
            path, "w", driver="GTiff",
            height=canvas_uint8.shape[0],
            width =canvas_uint8.shape[1],
            count=1, dtype=rasterio.uint8
        ) as dst:
            dst.write(canvas_uint8, 1)
        print("🌍 GeoTIFF guardado →", path)
    except ImportError:
        print("ℹ️  rasterio no instalado → se omite GeoTIFF")

# ─── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.dirname(MASKS_OUT_PT), exist_ok=True)
    os.makedirs(os.path.dirname(PNG_OUT_PATH),  exist_ok=True)

    patches = torch.load(PATCHES_PATH)            # (B,10,128,128)
    masks   = generar_mascaras(patches)

    torch.save(masks, MASKS_OUT_PT)
    print(f"✅ masks.pt guardado → {MASKS_OUT_PT}  shape={tuple(masks.shape)}")

    # mosaico png
    exportar_png(masks, PNG_OUT_PATH, cols=COLS_MOSAICO)

    # mosaico GeoTIFF opcional
    canvas = (masks.numpy().reshape(-1,128,128)  # (B,128,128)
              .transpose(1,0,2).reshape(128, -1))  # rápido si B=cols*rows
    exportar_tiff(canvas.astype(np.uint8)*255, GEOTIFF_PATH)
