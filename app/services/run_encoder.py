import os
import glob
import torch
import rasterio
import numpy as np
from app.models.encoder_cnn import EncoderCNN
from app.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = "data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def obtener_salida_numerada() -> str:
    """
    Generar un nombre de salida numerado (`salida_encoder_1.pt`, `salida_encoder_2.pt`, etc.).
    """
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "salida_encoder_*.pt"))
    next_num = len(existing_files) + 1
    output_path = os.path.join(OUTPUT_DIR, f"salida_encoder_{next_num}.pt")
    return output_path

def load_patches_as_tensor(patch_dir, patch_size=(128, 128), max_patches=4):
    """
    Cargar los parches desde el directorio y convertirlos a tensores.
    """
    tif_files = sorted(glob.glob(os.path.join(patch_dir, "*.tif")))[:max_patches]
    patch_tensors = []

    for path in tif_files:
        with rasterio.open(path) as src:
            patch = src.read()  # (bands, height, width)
            if patch.shape[1:] != patch_size or patch.shape[0] != 10:
                logger.warning(f"Omitiendo {path}: dimensiones incompatibles {patch.shape}")
                continue

            patch_tensors.append(torch.tensor(patch, dtype=torch.float32))

    if not patch_tensors:
        raise ValueError("No se encontraron parches válidos en el directorio.")

    logger.info(f"{len(patch_tensors)} parches cargados correctamente.")
    return torch.stack(patch_tensors)  # (B, 10, 128, 128)

def process_patches(patch_dir: str) -> str:
    """
    Procesar los parches utilizando el modelo EncoderCNN.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Procesando parches en: {patch_dir}")

    batch = load_patches_as_tensor(patch_dir).to(device)

    model = EncoderCNN().to(device)
    model.eval()

    output_path = obtener_salida_numerada()

    with torch.no_grad():
        output = model(batch)
        torch.save(output, output_path)

    logger.info(f"Procesamiento completado. Salida guardada en {output_path}")
    return output_path

def main():
    """Ejecutar el procesamiento manualmente."""
    patch_dir = "data/uploads"
    process_patches(patch_dir)

if __name__ == "__main__":
    main()
