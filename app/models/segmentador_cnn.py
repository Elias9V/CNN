import torch.nn as nn
import torch.nn.functional as F
from app.models.encoder_cnn import EncoderCNN

class SegmentadorCNN(nn.Module):
    def __init__(self):
        super(SegmentadorCNN, self).__init__()
        self.encoder = EncoderCNN()

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 32→128
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),  # Segmentación binaria
        )

    def forward(self, x):
        encoded, _ = self.encoder(x)  # (B,128,32,32)
        return self.decoder(encoded)  # (B,1,128,128)
