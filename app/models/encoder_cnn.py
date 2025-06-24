import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pooling=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2) if pooling else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

class MFF_MSCA(nn.Module):
    def __init__(self, in_channels):
        super(MFF_MSCA, self).__init__()
        self.fusion = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        fused = self.fusion(x)
        weights = self.attention(fused)
        return fused * weights

class EncoderCNN(nn.Module):
    def __init__(self, in_channels=10):
        super(EncoderCNN, self).__init__()

        self.conv1 = ConvBlock(in_channels, 32, pooling=False)
        self.conv2 = ConvBlock(32, 32, pooling=True)
        self.conv3 = ConvBlock(32, 64, pooling=False)
        self.conv4 = ConvBlock(64, 64, pooling=True)
        self.conv5 = ConvBlock(64, 128, pooling=False)

        # Rama adicional: AvgPool desde x original (128→64)
        self.avgpool128_64 = nn.AvgPool2d(2)

        # Recalculamos los canales de entrada para la fusión MFF_MSCA (por el nuevo concat)
        self.mff_msca1 = MFF_MSCA(10 + 32 + 64 + in_channels)  # +10 por avgpool(x)
        self.mff_msca2 = MFF_MSCA(10 + 128 + 128)

        self.conv_final1 = ConvBlock(128, 128, pooling=False)
        self.conv_final2 = ConvBlock(128, 128, pooling=False)
        self.final_mff_msca = MFF_MSCA(128 + 128 + 128)

    def forward(self, x):
        # Preparar resoluciones
        x_64 = F.interpolate(x, size=(64, 64), mode='bilinear')
        x_32 = F.interpolate(x, size=(32, 32), mode='bilinear')
        x_avg_64 = self.avgpool128_64(x)  # ← NUEVO camino verde

        # Forward normal
        out1 = self.conv1(x)     # (B,32,128,128)
        out2 = self.conv2(out1)  # (B,32,64,64)
        out3 = self.conv3(out2)  # (B,64,64,64)
        out4 = self.conv4(out3)  # (B,64,32,32)
        out5 = self.conv5(out4)  # (B,128,32,32)

        # Fusión multiescala en 64x64, ahora con rama verde
        branch_64 = torch.cat([x_64, out2, out3, x_avg_64], dim=1)  # ← rama adicional
        fused_64 = self.mff_msca1(branch_64)

        # Fusión en 32x32
        branch_32 = torch.cat([x_32, out5, F.interpolate(fused_64, size=(32, 32))], dim=1)
        fused_32 = self.mff_msca2(branch_32)

        # Refinamiento
        out6 = self.conv_final1(fused_32)
        out7 = self.conv_final2(out6)

        final = torch.cat([fused_32, out6, out7], dim=1)
        encoded = self.final_mff_msca(final)

        return encoded