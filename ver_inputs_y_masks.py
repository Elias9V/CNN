import torch
import matplotlib.pyplot as plt
import numpy as np

# Cargar archivos
X = torch.load("data/uploads/patches_input.pt")        # (N, 10, 128, 128)
Y = torch.load("data/masks.pt")               # (N, 128, 128)

# Normalizar entrada (para visualizar bandas como RGB)
def normalizar(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min + 1e-6)

# Mostrar hasta 8 pares máximo
n = min(8, X.shape[0])

for i in range(n):
    img = X[i]  # (10, 128, 128)
    mask = Y[i] # (128, 128)

    # Seleccionar bandas simuladas para "RGB": rojo (3), verde (2), azul (1)
    img_rgb = torch.stack([img[3], img[2], img[1]], dim=0)  # (3, 128, 128)
    img_rgb = normalizar(img_rgb).permute(1, 2, 0).numpy()  # (128, 128, 3)

    mask_np = mask.numpy()

    # Mostrar lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img_rgb)
    axs[0].set_title(f"Entrada #{i}")
    axs[1].imshow(mask_np, cmap="Reds", vmin=0, vmax=1)
    axs[1].set_title(f"Máscara #{i}")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
