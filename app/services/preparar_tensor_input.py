import os
import torch
import rasterio

INPUT_DIR = "data/uploads/patches"
OUTPUT_PATH = "data/uploads/patches_input.pt"

def preparar_tensor_input() -> str:
    """
    Convierte archivos .tif en un tensor .pt compatible con el encoder.
    Devuelve la ruta del archivo generado.
    """
    tensors = []

    for filename in sorted(os.listdir(INPUT_DIR)):
        if filename.endswith(".tif"):
            path = os.path.join(INPUT_DIR, filename)
            with rasterio.open(path) as src:
                data = src.read()
                if data.shape != (10, 128, 128):
                    continue
                tensors.append(torch.tensor(data, dtype=torch.float32))

    if not tensors:
        raise ValueError("No se encontraron archivos .tif válidos.")

    tensor_final = torch.stack(tensors)
    torch.save(tensor_final, OUTPUT_PATH)
    return OUTPUT_PATH
