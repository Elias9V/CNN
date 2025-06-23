import os
import torch
import rasterio

INPUT_DIR = "data/uploads/patches"
OUTPUT_PATH = "data/uploads/patches_input.pt"

def preparar_tensor_input() -> str:
    """
    Convierte archivos .tif en un tensor .pt compatible con el encoder,
    limpiando NaN e infinitos automáticamente.
    """
    tensors = []

    for filename in sorted(os.listdir(INPUT_DIR)):
        if filename.endswith(".tif"):
            path = os.path.join(INPUT_DIR, filename)
            with rasterio.open(path) as src:
                data = src.read()  # (10, 128, 128)
                if data.shape != (10, 128, 128):
                    continue
                tensor = torch.tensor(data, dtype=torch.float32)

                # 🧼 Sanear: reemplazar NaN y ±inf
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

                tensors.append(tensor)

    if not tensors:
        raise ValueError("No se encontraron archivos .tif válidos.")

    tensor_final = torch.stack(tensors)  # (B, 10, 128, 128)
    torch.save(tensor_final, OUTPUT_PATH)
    return OUTPUT_PATH
