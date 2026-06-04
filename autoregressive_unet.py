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
        self.dropout = nn.Dropout2d(p=0.1)  # Optional dropout for regularization
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResidualConv(nn.Module):
    """Replaces DoubleConv with a ResNet-style block to preserve crisp grid boundaries."""
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_p)
        
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
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += res  # Add the residual identity
        return F.relu(out)

class UNet16x16(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.2):
        super().__init__()
        # Expected input channels: 3
        #   - Channel 0: grid map (obstacles=1, free=0)
        #   - Channel 1: current position (one-hot)
        #   - Channel 2: unseen map (not yet visible=1)
        
        # We add +2 to in_channels to account for the generated X and Y coordinate channels
        self.inc = ResidualConv(in_channels + 2, 64)
        
        # Encoder
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(128, 256, dropout_p=dropout_p))
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.conv2 = ResidualConv(256, 128, dropout_p=dropout_p)
        
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






import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. NEW COMPONENT: Time Embedding (Tells the network what stage of flow it is in)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

# 2. UPDATED COMPONENT: Inject Time into the Residual Blocks
class ResidualConvWithTime(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_p)
        
        # NEW: MLP to process the time embedding and inject it into the block
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, t_emb=None):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        
        # NEW: Add the time embedding to the feature map!
        if self.time_mlp is not None and t_emb is not None:
            time_hidden = self.time_mlp(t_emb)
            # Expand dimensions to match (Batch, Channels, Height, Width)
            out = out + time_hidden.unsqueeze(-1).unsqueeze(-1) 
            
        out = self.bn2(self.conv2(out))
        out += res
        return F.relu(out)

# 3. UPDATED U-NET: Accepts Time (t) and Noisy Path (x_t)
class FlowMatchingUNet(nn.Module):
    def __init__(self, context_channels=3, path_channels=1, out_channels=1, time_emb_dim=128, dropout_p=0.0):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # in_channels = context (3) + noisy_path (1) + coords (2) = 6
        total_in = context_channels + path_channels + 2 
        
        # Pass time_emb_dim to ResidualConv so it knows to expect time
        self.inc = ResidualConvWithTime(total_in, 64, time_emb_dim)
        
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualConvWithTime(64, 128, time_emb_dim))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualConvWithTime(128, 256, time_emb_dim, dropout_p))
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.conv2 = ResidualConvWithTime(256, 128, time_emb_dim, dropout_p)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.conv1 = ResidualConvWithTime(128, 64, time_emb_dim)
        
        # Output: NO SIGMOID! We are predicting velocity (dx/dt), which can be negative.
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def _get_coord_channels(self, batch_size, h, w, device):
        """Generates normalized X and Y coordinate grids."""
        y_coords = torch.linspace(-1, 1, steps=h, device=device)
        x_coords = torch.linspace(-1, 1, steps=w, device=device)
        
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Shape: (Batch, 1, H, W)
        yy = yy.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        xx = xx.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        return torch.cat([yy, xx], dim=1) # Shape: (Batch, 2, H, W)

    # NEW: Forward pass now requires 't' (a tensor of times between 0 and 1)
    def forward(self, context, noisy_path, t):
        # 1. Time Embedding
        t_emb = self.time_mlp(t)
        
        # 2. Inject Coordinates and Concatenate
        b, c, h, w = context.shape
        coords = self._get_coord_channels(b, h, w, context.device)
        
        # Stack: [Obstacles, Pos, Fog, Noisy Path, Y_coord, X_coord]
        x = torch.cat([context, noisy_path, coords], dim=1) 
        
        # 3. Encoder (Must pass t_emb into the blocks)
        x1 = self.inc(x, t_emb)
        # Note: We index [1] because down blocks are wrapped in nn.Sequential
        x2 = self.down1[1](self.down1[0](x1), t_emb) 
        x3 = self.down2[1](self.down2[0](x2), t_emb)
        
        # 4. Decoder
        g2 = self.up2(x3)
        x2_att = self.att2(g=g2, x=x2)
        d2 = torch.cat((x2_att, g2), dim=1)
        d2 = self.conv2(d2, t_emb)
        
        g1 = self.up1(d2)
        x1_att = self.att1(g=g1, x=x1)
        d1 = torch.cat((x1_att, g1), dim=1)
        d1 = self.conv1(d1, t_emb)
        
        # 5. Output Velocity Vector
        velocity = self.outc(d1)
        return velocity