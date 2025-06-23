import torch
import numpy as np
from PIL import Image, ImageDraw
import os
import math
import cv2

# Configuración
INPUT_DIR = "data/outputs/predicciones"
MASCARA_CIUDAD_PATH = "data/mapas/ciudad_original.png"
OUTPUT_PATH = "data/outputs/mosaico_predicciones_guindo_borde.png"
PATCH_SIZE = 128

# Colores
COLOR_RIESGO = [0, 0, 0]       # rojo
COLOR_BORDE = [255, 255, 0]       # Amarillo
COLOR_SEGURA = [225, 0, 0]          # Negro

def cargar_predicciones():
    archivos = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".pt")])
    parches = []

    for f in archivos:
        patch = torch.load(os.path.join(INPUT_DIR, f))
        if patch.ndim == 3:
            patch = patch.squeeze(0)
        parches.append(patch.int().numpy())

    return parches

def reconstruir_mosaico_color(parches, mascara_ciudad=None):
    B = len(parches)
    cols = 4
    rows = math.ceil(B / cols)

    H, W = PATCH_SIZE, PATCH_SIZE
    canvas = np.zeros((rows * H, cols * W), dtype=np.uint8)

    for i, patch in enumerate(parches):
        r = i // cols
        c = i % cols
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = patch

    # Redimensionar la máscara de ciudad
    if mascara_ciudad is not None:
        mascara_ciudad = Image.fromarray(mascara_ciudad).resize((cols * W, rows * H), resample=Image.NEAREST)
        mascara_ciudad = np.array(mascara_ciudad)

    # Crear imagen RGB final
    imagen_color = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)

    for y in range(canvas.shape[0]):
        for x in range(canvas.shape[1]):
            if mascara_ciudad is not None and mascara_ciudad[y, x] == 255:
                color = COLOR_SEGURA
            elif canvas[y, x] == 1:
                color = COLOR_RIESGO
            else:
                color = COLOR_SEGURA
            imagen_color[y, x] = color

    # Detectar bordes de zonas de riesgo y dibujar en amarillo
    bordes = cv2.Canny((canvas * 255).astype(np.uint8), 100, 200)
    for y in range(bordes.shape[0]):
        for x in range(bordes.shape[1]):
            if bordes[y, x] > 0:
                imagen_color[y, x] = COLOR_BORDE

    return imagen_color

def main():
    parches = cargar_predicciones()

    if os.path.exists(MASCARA_CIUDAD_PATH):
        mascara_ciudad = np.array(Image.open(MASCARA_CIUDAD_PATH).convert("L"))
        print("✅ Máscara de ciudad cargada y será aplicada")
    else:
        mascara_ciudad = None
        print("⚠️ No se encontró la máscara de ciudad, se generará todo el mosaico sin omitir")

    mosaico = reconstruir_mosaico_color(parches, mascara_ciudad)
    Image.fromarray(mosaico).save(OUTPUT_PATH)
    print(f"✅ Mosaico final guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
