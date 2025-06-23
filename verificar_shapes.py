import torch

X = torch.load("data/uploads/patches_input.pt")
Y = torch.load("data/masks.pt")

print("Entradas (X):", X.shape)  # Esperado: (N, 10, 128, 128)
print("Máscaras (Y):", Y.shape)  # Esperado: (N, 128, 128)

assert X.shape[0] == Y.shape[0], "❌ X e Y tienen diferente cantidad de muestras"
print("✅ Coinciden en tamaño.")
