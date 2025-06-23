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
    def __init__(self, in_channels=1):
        super(EncoderCNN, self).__init__()

        self.conv1 = ConvBlock(in_channels, 32, pooling=False)
        self.conv2 = ConvBlock(32, 32, pooling=True)
        self.conv3 = ConvBlock(32, 64, pooling=False)
        self.conv4 = ConvBlock(64, 64, pooling=True)
        self.conv5 = ConvBlock(64, 128, pooling=False)

        # Fusión multiescala
        self.mff_msca1 = MFF_MSCA(98)   # 1 (x_64) + 32 (out2) + 64 (out3) + 1 (x_avg_64)
        self.mff_msca2 = MFF_MSCA(257)  # 1 (x_32) + 128 (out5) + 128 (fused_64)

        self.conv_final1 = ConvBlock(128, 128, pooling=False)
        self.conv_final2 = ConvBlock(128, 128, pooling=False)
        self.final_mff_msca = MFF_MSCA(384)  # 128 + 128 + 128

    def forward(self, x):
        # Escalado proporcional que se adapta a cualquier tamaño
        x_64 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_32 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_avg_64 = F.avg_pool2d(x, kernel_size=2, stride=2)

        # Forward CNN
        out1 = self.conv1(x)     # (B, 32, H, W)
        out2 = self.conv2(out1)  # (B, 32, H/2, W/2)
        out3 = self.conv3(out2)  # (B, 64, H/2, W/2)
        out4 = self.conv4(out3)  # (B, 64, H/4, W/4)
        out5 = self.conv5(out4)  # (B, 128, H/4, W/4)

        # Fusión 64x64
        x_64 = F.interpolate(x_64, size=out2.shape[-2:], mode='bilinear', align_corners=False)
        x_avg_64 = F.interpolate(x_avg_64, size=out2.shape[-2:], mode='bilinear', align_corners=False)
        branch_64 = torch.cat([x_64, out2, out3, x_avg_64], dim=1)
        fused_64 = self.mff_msca1(branch_64)

        # Fusión 32x32
        x_32 = F.interpolate(x_32, size=out5.shape[-2:], mode='bilinear', align_corners=False)
        fused_64_32 = F.interpolate(fused_64, size=out5.shape[-2:], mode='bilinear', align_corners=False)
        branch_32 = torch.cat([x_32, out5, fused_64_32], dim=1)
        fused_32 = self.mff_msca2(branch_32)

        # Refinamiento
        out6 = self.conv_final1(fused_32)
        out7 = self.conv_final2(out6)
        final = torch.cat([fused_32, out6, out7], dim=1)
        encoded = self.final_mff_msca(final)

        return encoded, x.shape[-2:]
