import torch
from torch.utils.data import Dataset

class LandslideDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        """
        x_path: ruta al tensor de entrada .pt (N, 10, 128, 128)
        y_path: ruta al tensor de máscaras .pt (N, 128, 128)
        """
        self.X = torch.load(x_path)  # (N, 10, 128, 128)
        self.Y = torch.load(y_path)  # (N, 128, 128)
        self.transform = transform

        assert self.X.shape[0] == self.Y.shape[0], "X y Y deben tener el mismo número de muestras"

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx].long()  # CrossEntropyLoss requiere tipo long

        if self.transform:
            x, y = self.transform(x, y)

        return x, y
