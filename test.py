import torch
mask = torch.load("data/inputs/full_mask.pt")
print("Min:", mask.min().item())
print("Max:", mask.max().item())
print("Proporción de 1s:", mask.sum().item() / mask.numel())
