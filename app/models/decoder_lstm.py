import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLSTM(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=256, output_dim=2):
        super(DecoderLSTM, self).__init__()

        # LSTM que procesa secuencias (B, T=1024, C=128)
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)

        # Mecanismo de atención temporal
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # Decoder: mapear (B, 256, 32, 32) → (B, 2, 128, 128)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, output_dim, kernel_size=1)
        )

    def forward(self, x_seq):  # x_seq: (B, 1024, 128)
        # Verifica entrada
        if torch.isnan(x_seq).any():
            print("⚠️ NaN detectado en x_seq")

        lstm_out, _ = self.lstm(x_seq)  # (B, 1024, 256)
        if torch.isnan(lstm_out).any():
            print("❌ NaN detectado en lstm_out")

        weights = self.attention(lstm_out)  # (B, 1024, 1)

        if torch.isnan(weights).any():
            print("❌ NaN detectado en weights")

        context = torch.sum(weights * lstm_out, dim=1)  # (B, 256)
        if torch.isnan(context).any():
            print("❌ NaN detectado en context")

        # Suma residual con último paso del LSTM
        last_out = lstm_out[:, -1, :]         # (B, 256)
        fused = context + last_out            # (B, 256)

        if torch.isnan(fused).any():
            print("❌ NaN detectado en fused (residual sum)")

        # Expandir como mapa (B, 256, 32, 32)
        fused = fused.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 32)

        out = self.decoder(fused)             # (B, 2, 128, 128)

        # Protección final
        if torch.isnan(out).any():
            print("❌ NaN detectado en out. Corrigiendo con nan_to_num.")
            out = torch.nan_to_num(out, nan=0.0)

        return out