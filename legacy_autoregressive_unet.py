import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet16x16(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        # Input: 16x16
        self.inc = DoubleConv(in_channels, 64)
        
        # 16x16 -> 8x8
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        # 8x8 -> 4x4 (Bottleneck)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.conv2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.conv1 = DoubleConv(128, 64)
        
        # Output
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)         # Shape: (B, 64, 16, 16)
        x2 = self.down1(x1)      # Shape: (B, 128, 8, 8)
        x3 = self.down2(x2)      # Shape: (B, 256, 4, 4) - Bottleneck
        
        # Decoder 1: 4x4 -> 8x8
        g2 = self.up2(x3)
        x2_att = self.att2(g=g2, x=x2)
        d2 = torch.cat((x2_att, g2), dim=1)
        d2 = self.conv2(d2)      # Shape: (B, 128, 8, 8)
        
        # Decoder 2: 8x8 -> 16x16
        g1 = self.up1(d2)
        x1_att = self.att1(g=g1, x=x1)
        d1 = torch.cat((x1_att, g1), dim=1)
        d1 = self.conv1(d1)      # Shape: (B, 64, 16, 16)
        
        # Output probability map
        logits = self.outc(d1)
        probs = self.sigmoid(logits)
        
        return probs

# # --- Test the 16x16 shapes ---
# if __name__ == "__main__":
#     dummy_input = torch.randn(1, 3, 16, 16)
#     model = UNet16x16(in_channels=3, out_channels=1)
#     output = model(dummy_input)
#     print(f"Input shape: {dummy_input.shape}")
#     print(f"Output shape: {output.shape}") 
#     # Expected: torch.Size([1, 1, 16, 16])