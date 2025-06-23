import os
import numpy as np
import torch
from PIL import Image

# Ruta al mapa de riesgo JPG
MAP_PATH = "data/inputs/mapa_riesgo.jpg"  # cambia si es necesario
PATCH_SIZE = 128
OUTPUT_PATH = "data/masks.pt"

def cargar_imagen(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def extraer_mascara_binaria(img_rgb):
    # Define rojo como zonas de riesgo (alto y muy alto)
    lower = np.array([100, 0, 0])  # Rojos oscuros
    upper = np.array([255, 100, 100])
    mask = np.all((img_rgb >= lower) & (img_rgb <= upper), axis=-1)
    return mask.astype(np.uint8)

def dividir_en_patches(mask, patch_size=128):
    patches = []
    h, w = mask.shape
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = mask[y:y+patch_size, x:x+patch_size]
            if patch.shape == (patch_size, patch_size):
                patches.append(torch.tensor(patch, dtype=torch.long))
    return patches

def main():
    img_rgb = cargar_imagen(MAP_PATH)
    mask_bin = extraer_mascara_binaria(img_rgb)
    patches = dividir_en_patches(mask_bin)

    if not patches:
        raise ValueError("❌ No se generaron parches. Verifica la imagen y el tamaño.")

    tensor = torch.stack(patches)  # (N, 128, 128)
    os.makedirs("data", exist_ok=True)
    torch.save(tensor, OUTPUT_PATH)
    print(f"✅ Se guardaron {len(patches)} máscaras en {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
