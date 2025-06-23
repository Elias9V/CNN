import torch

X = torch.load("data/uploads/patches_input.pt")
Y = torch.load("data/masks.pt")

NX = X.shape[0]
NY = Y.shape[0]

if NX != NY:
    print(f"⚠️ Cantidad de entradas: {NX}, máscaras: {NY}")
    min_N = min(NX, NY)
    Y_recortado = Y[:min_N]
    torch.save(Y_recortado, "data/masks.pt")
    print(f"✅ Se recortaron las máscaras a {min_N} muestras.")
else:
    print("✅ Ya están sincronizados.")
