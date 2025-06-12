import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLSTM(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=256, output_dim=2):
        super(DecoderLSTM, self).__init__()

        # LSTM que procesa secuencias (B, 1024, 128)
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)

        # Mecanismo de atención sobre la salida de LSTM
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)  # atención temporal sobre secuencia
        )

        # Reconstrucción a imagen 128x128
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, output_dim, kernel_size=1)
        )

    def forward(self, x_seq):  # x_seq: (B, 1024, 128)
        lstm_out, _ = self.lstm(x_seq)  # (B, 1024, hidden_dim)

        # Atención
        weights = self.attention(lstm_out)  # (B, 1024, 1)
        context = torch.sum(weights * lstm_out, dim=1)  # (B, hidden_dim)

        # Expandir como mapa (B, hidden_dim, 32, 32)
        context = context.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 32)

        out = self.decoder(context)  # (B, output_dim, 128, 128)
        return out