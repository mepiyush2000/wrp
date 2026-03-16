import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    # (Keeping your exact AttentionBlock, it is perfectly fine)
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

class ResidualConv(nn.Module):
    """Replaces DoubleConv with a ResNet-style block to preserve crisp grid boundaries."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions if they change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res  # Add the residual identity
        return F.relu(out)

class UNet16x16(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Expected input channels: 3
        #   - Channel 0: grid map (obstacles=1, free=0)
        #   - Channel 1: current position (one-hot)
        #   - Channel 2: unseen map (not yet visible=1)
        
        # We add +2 to in_channels to account for the generated X and Y coordinate channels
        self.inc = ResidualConv(in_channels + 2, 64)
        
        # Encoder
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(128, 256))
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.conv2 = ResidualConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.conv1 = ResidualConv(128, 64)
        
        # Output
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _get_coord_channels(self, batch_size, h, w, device):
        """Generates normalized X and Y coordinate grids."""
        y_coords = torch.linspace(-1, 1, steps=h, device=device)
        x_coords = torch.linspace(-1, 1, steps=w, device=device)
        
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Shape: (Batch, 1, H, W)
        yy = yy.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        xx = xx.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        return torch.cat([yy, xx], dim=1) # Shape: (Batch, 2, H, W)

    def forward(self, x):
        # 1. Inject Coordinates dynamically
        b, c, h, w = x.shape
        coords = self._get_coord_channels(b, h, w, x.device)
        x = torch.cat([x, coords], dim=1)  # Now 5 channels going into the network
        
        # 2. Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # 3. Decoder
        g2 = self.up2(x3)
        x2_att = self.att2(g=g2, x=x2)
        d2 = torch.cat((x2_att, g2), dim=1)
        d2 = self.conv2(d2)
        
        g1 = self.up1(d2)
        x1_att = self.att1(g=g1, x=x1)
        d1 = torch.cat((x1_att, g1), dim=1)
        d1 = self.conv1(d1)
        
        # 4. Output
        logits = self.outc(d1)
        return self.sigmoid(logits)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ---------------------------------------------------------------------------
# # Building Blocks
# # ---------------------------------------------------------------------------

# class AddCoords(nn.Module):
#     """Appends normalised (row, col) coordinate channels to the input tensor.
    
#     Input:  (B, C, H, W)
#     Output: (B, C+2, H, W)   with channels for row ∈ [0,1] and col ∈ [0,1]
#     """
#     def forward(self, x):
#         B, _, H, W = x.shape
#         # row coordinates: 0 → 1 top-to-bottom
#         row = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
#         row = row.view(1, 1, H, 1).expand(B, 1, H, W)
#         # col coordinates: 0 → 1 left-to-right
#         col = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype)
#         col = col.view(1, 1, 1, W).expand(B, 1, H, W)
#         return torch.cat([x, row, col], dim=1)


# class AttentionBlock(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(AttentionBlock, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
        
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
        
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
        
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi


# class ResidualDoubleConv(nn.Module):
#     """Two 3×3 convolutions with a residual (skip) connection.
    
#     If in_channels != out_channels a 1×1 projection is used on the shortcut.
#     """
#     def __init__(self, in_channels, out_channels, dropout_p=0.0):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#         )
#         # 1×1 projection when channel counts differ
#         self.shortcut = (
#             nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_channels),
#             )
#             if in_channels != out_channels
#             else nn.Identity()
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.conv(x) + self.shortcut(x))



# class UNet16x16(nn.Module):
#     """Attention U-Net with CoordConv, residual blocks, dropout and spatial softmax.
    
#     Improvements over the original:
#       1. CoordConv  – prepends (row, col) coordinate channels for spatial awareness
#       2. Residual DoubleConv – skip-connections inside each conv block
#       3. Dropout2d  – regularisation at bottleneck & decoder stages
#       4. Spatial Softmax – output is a proper probability distribution over cells
#          (one value sums to 1 across all H×W positions instead of independent sigmoids)
#     """

#     def __init__(self, in_channels=3, out_channels=1, dropout_p=0.2):
#         super().__init__()

#         # --- CoordConv: adds 2 coordinate channels ---
#         self.add_coords = AddCoords()            # in_channels → in_channels + 2
#         coord_channels = in_channels + 2

#         # --- Encoder ---
#         self.inc = ResidualDoubleConv(coord_channels, 64)

#         # 16×16 → 8×8
#         self.pool1 = nn.MaxPool2d(2)
#         self.enc1  = ResidualDoubleConv(64, 128)

#         # 8×8 → 4×4  (bottleneck)
#         self.pool2 = nn.MaxPool2d(2)
#         self.enc2  = ResidualDoubleConv(128, 256, dropout_p=dropout_p)

#         # --- Decoder ---
#         self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
#         self.dec2 = ResidualDoubleConv(256, 128, dropout_p=dropout_p)

#         self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
#         self.dec1 = ResidualDoubleConv(128, 64)

#         # --- Output head ---
#         self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
#         # No sigmoid — we use spatial softmax in forward()

#     def forward(self, x):
#         # Prepend coordinate channels: (B, 3, H, W) → (B, 5, H, W)
#         x = self.add_coords(x)

#         # ---- Encoder ----
#         x1 = self.inc(x)                     # (B, 64,  H,   W)
#         x2 = self.enc1(self.pool1(x1))       # (B, 128, H/2, W/2)
#         x3 = self.enc2(self.pool2(x2))       # (B, 256, H/4, W/4)  bottleneck

#         # ---- Decoder ----
#         g2 = self.up2(x3)                    # (B, 128, H/2, W/2)
#         x2_att = self.att2(g=g2, x=x2)
#         d2 = self.dec2(torch.cat([x2_att, g2], dim=1))  # (B, 128, H/2, W/2)

#         g1 = self.up1(d2)                    # (B, 64, H, W)
#         x1_att = self.att1(g=g1, x=x1)
#         d1 = self.dec1(torch.cat([x1_att, g1], dim=1))  # (B, 64, H, W)

#         # ---- Output ----
#         logits = self.outc(d1)               # (B, 1, H, W)
#         return self.sigmoid(logits)

#         # # Spatial softmax: flatten spatial dims → softmax → reshape
#         # B, C, H, W = logits.shape
#         # logits_flat = logits.view(B, C, -1)           # (B, 1, H*W)
#         # probs_flat  = F.softmax(logits_flat, dim=-1)   # sum-to-1 over pixels
#         # probs = probs_flat.view(B, C, H, W)            # (B, 1, H, W)

#         # return probs