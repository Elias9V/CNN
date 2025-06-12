import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from app.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = "data/outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "salida_encoder_1.pt")

def diagnosticar_tensor(tensor: torch.Tensor):
    """
    Mostrar estadísticas detalladas del tensor para depuración.
    """
    logger.info(f"Dimensiones del tensor: {tensor.shape}")
    logger.info(f"Tipo de datos: {tensor.dtype}")

    min_val = tensor.min().item()
    max_val = tensor.max().item()
    mean_val = tensor.mean().item()
    std_val = tensor.std().item()

    logger.info(f"Valores del tensor - Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")

    # Verificar si todos los valores son cero
    unique_vals = torch.unique(tensor)
    logger.info(f"Valores únicos en el tensor: {unique_vals}")

    if len(unique_vals) == 1 and unique_vals[0] == 0:
        logger.warning("El tensor contiene únicamente ceros. No se puede visualizar.")
        return False

    return True

def visualizar_feature_maps():
    """
    Visualizar los mapas de características del EncoderCNN.
    """
    if not os.path.exists(OUTPUT_PATH):
        logger.error(f"No se encontró el archivo {OUTPUT_PATH}")
        raise FileNotFoundError(f"No se encontró el archivo {OUTPUT_PATH}")

    # Cargar el tensor
    tensor = torch.load(OUTPUT_PATH)

    # Realizar diagnóstico
    if not diagnosticar_tensor(tensor):
        logger.warning("El tensor está vacío o contiene únicamente ceros.")
        return

    # Seleccionar los canales a visualizar
    canales = [0, 10, 50, 127] if tensor.shape[1] >= 128 else [0, 1, 2, 3]

    for i in range(min(tensor.shape[0], 1)):  # Solo visualizamos el primer patch para simplificar
        fig, axs = plt.subplots(1, len(canales), figsize=(16, 4))
        for j, ch in enumerate(canales):
            if ch >= tensor.shape[1]:
                logger.warning(f"Canal {ch} no existe en el tensor. Skipping...")
                continue

            img = tensor[i, ch].cpu().numpy()

            # Verificar si la imagen está vacía
            if np.all(img == 0):
                logger.warning(f"Canal {ch} del parche {i} está vacío.")
                axs[j].set_title(f"Canal {ch}: Vacío")
                axs[j].axis('off')
                continue

            axs[j].imshow(img, cmap='inferno')
            axs[j].set_title(f"Patch {i} - Channel {ch}")
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()

    logger.info("Visualización completada.")
