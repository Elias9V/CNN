import torch
import os
from PIL import Image
import numpy as np
import math

# Rutas
PATCHES_PATH = "data/tensors/patches_input.pt"
OUTPUT_MASKS_PATH = "data/tensors/masks.pt"
OUTPUT_PNG_PATH = "data/outputs/mosaico_mascaras_gt.png"

# Umbral de pendiente (grados o normalizado)
UMBRAL_PENDIENTE = 0.33  # Usa 0.33 si está en [0, 1]

def generar_mascaras_por_regla(patch_tensor_path, output_path, umbral=UMBRAL_PENDIENTE):
    print(f"📦 Cargando parches desde: {patch_tensor_path}")
    patches = torch.load(patch_tensor_path)  # (B, 10, 128, 128)

    print("📉 Extrayendo banda de pendiente...")
    pendiente = patches[:, -1, :, :]  # Última banda

    print(f"⚙️ Aplicando umbral > {umbral} para clasificar riesgo")
    masks = (pendiente > umbral).int()  # 1 = riesgo, 0 = no riesgo

    print(f"💾 Guardando máscaras binarias en: {output_path}")
    torch.save(masks, output_path)

    print(f"✅ Máscaras generadas correctamente con shape: {masks.shape} y tipo: {masks.dtype}")
    return masks

def exportar_mosaico_png(masks_tensor, output_png_path, patch_size=128):
    B, H, W = masks_tensor.shape
    cols = int(np.sqrt(B))
    rows = math.ceil(B / cols)

    print(f"🧩 Reconstruyendo mosaico {rows}x{cols}...")
    canvas = np.zeros((rows * H, cols * W), dtype=np.uint8)

    for i in range(B):
        r = i // cols
        c = i % cols
        patch = masks_tensor[i].numpy() * 255  # binario a blanco/negro
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = patch

    Image.fromarray(canvas).save(output_png_path)
    print(f"🖼 Mosaico de máscaras guardado en: {output_png_path}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_MASKS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PNG_PATH), exist_ok=True)

    masks = generar_mascaras_por_regla(PATCHES_PATH, OUTPUT_MASKS_PATH)
    exportar_mosaico_png(masks, OUTPUT_PNG_PATH)
