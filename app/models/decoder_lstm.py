import torch
import torch.nn as nn
import torch.nn.functional as F  # Asegúrate de no sobrescribir esto

class DecoderLSTM(nn.Module):
    def __init__(self, feature_dim=131072, hidden_dim=256, output_dim=1):
        super(DecoderLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=1),
            nn.Conv2d(64, output_dim, kernel_size=1)
        )

    def forward(self, x_seq, output_size):
        # ✅ Verificar output_size
        if not isinstance(output_size, (tuple, list)) or len(output_size) != 2:
            raise ValueError(f"❌ El parámetro output_size debe ser una tupla (H, W), pero recibí: {output_size}")

        H, W = output_size
        B, T, F_ = x_seq.shape

        lstm_out, _ = self.lstm(x_seq)  # (B, T, hidden_dim)

        weights = self.attention(lstm_out)  # (B, T, 1)
        context = torch.sum(weights * lstm_out, dim=1)  # (B, hidden_dim)
        last_out = lstm_out[:, -1, :]  # (B, hidden_dim)
        fused = context + last_out     # (B, hidden_dim)

        fused = fused.unsqueeze(-1).unsqueeze(-1)  # (B, hidden_dim, 1, 1)
        fused = fused.expand(-1, -1, H // 4, W // 4)  # tamaño intermedio

        # ✅ Diagnóstico para evitar que F haya sido sobrescrito
        if not hasattr(F, "interpolate"):
            raise RuntimeError(f"❌ El módulo 'F' fue sobrescrito. Tipo actual: {type(F)}")

        out = self.decoder[0](fused)  # conv → (B,64,H/4,W/4)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        out = self.decoder[1](out)    # conv → (B,1,H,W)

        if torch.isnan(out).any():
            print("⚠️ NaNs detectados en salida del decoder. Corrigiendo.")
            out = torch.nan_to_num(out, nan=0.0)

        return out
