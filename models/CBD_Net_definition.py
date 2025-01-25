import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Basic Residual Block
# ----------------------------------------------------------------------
class ResBlock(nn.Module):
    """
    A simple residual block with two 3x3 convolutions (Conv->BN->ReLU),
    then a skip connection around them.
    """
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # For matching dimensions if in_channels != out_channels
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) \
            if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(identity)
        out += identity
        return F.relu(out, inplace=True)

# ----------------------------------------------------------------------
# scSE Attention Block
#   - cSE: Channel Squeeze & Excitation
#   - sSE: Spatial Squeeze & Excitation
# ----------------------------------------------------------------------
    
class scSEBlock(nn.Module):
    """
    The 'scSE' module applies:
      - cSE: Global Avg Pool -> FC -> ReLU -> FC -> Sigmoid to reweight channels
      - sSE: 1x1 Conv -> Sigmoid to reweight spatially
    Then combines them (elementwise addition of the masks).
    """
    def __init__(self, in_channels, reduction=16):
        super(scSEBlock, self).__init__()
        # Channel squeeze-excitation (cSE)
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial squeeze-excitation (sSE)
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel SE
        cse_mask = self.channel_excitation(x)
        # Spatial SE
        sse_mask = self.spatial_excitation(x)
        # Merge (elementwise) and apply
        x = x * cse_mask + x * sse_mask
        return x

# ----------------------------------------------------------------------
# Multi-parallel Dilated Convolution (MPDC) Block
#   - Parallel branches with different dilation rates
# ----------------------------------------------------------------------
class MPDCBlock(nn.Module):
    """
    Merges features from multiple dilated convolutions (rates = 1,2,4,8).
    Each branch:
      Conv -> BN -> ReLU
    The outputs of each branch are summed.
    """
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8]):
        super(MPDCBlock, self).__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        return torch.sum(torch.stack(outs), dim=0)  # sum along branch dimension

# ----------------------------------------------------------------------
# Encoder / Decoder Blocks
# ----------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """
    One stage of the encoder:
      - Repeated ResBlocks (configurable)
      - scSE attention
      - Downsample by factor of 2 (optional, except last stage might not downsample)
    """
    def __init__(self, in_channels, out_channels, num_res=2, downsample=True):
        super(EncoderBlock, self).__init__()
        layers = []
        ch = in_channels
        for i in range(num_res):
            layers.append(ResBlock(ch, out_channels))
            ch = out_channels
        self.res_blocks = nn.Sequential(*layers)
        self.scse = scSEBlock(out_channels)
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.scse(x)
        before_down = x
        if self.downsample:
            x = self.pool(x)
        return x, before_down

class DecoderBlock(nn.Module):
    """
    One stage of the decoder:
      - TransposedConv to upsample
      - Concatenate (skip connection)
      - Some ResBlocks
      - scSE attention
    """
    def __init__(self, in_channels, out_channels, num_res=2):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        layers = []
        ch = out_channels * 2  # after concat with skip
        for i in range(num_res):
            layers.append(ResBlock(ch, out_channels))
            ch = out_channels
        self.res_blocks = nn.Sequential(*layers)
        self.scse = scSEBlock(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # skip is from encoder
        x = torch.cat([x, skip], dim=1)
        x = self.res_blocks(x)
        x = self.scse(x)
        return x

# ----------------------------------------------------------------------
# The Overall Network Architecture (CBD-Net)
# ----------------------------------------------------------------------

class CBDNet(nn.Module):
    def __init__(self, in_channels=1, base_ch=64, num_classes=5):
        super(CBDNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=7, padding=3, stride=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.encoder1 = EncoderBlock(base_ch,    base_ch,    num_res=2, downsample=True)
        self.encoder2 = EncoderBlock(base_ch,    base_ch*2,  num_res=2, downsample=True)
        self.encoder3 = EncoderBlock(base_ch*2,  base_ch*4,  num_res=2, downsample=True)
        self.encoder4 = EncoderBlock(base_ch*4,  base_ch*8,  num_res=2, downsample=True)

        self.mpdc = MPDCBlock(in_channels=base_ch*8, out_channels=base_ch*8, 
                              dilations=[1,2,4,8])

        self.decoder1 = DecoderBlock(base_ch*8, base_ch*8)
        self.decoder2 = DecoderBlock(base_ch*8, base_ch*4)
        self.decoder3 = DecoderBlock(base_ch*4, base_ch*2)
        self.decoder4 = DecoderBlock(base_ch*2, base_ch)


        # Now output 5 channels for 5 classes
        self.seg_head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.initial(x)
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)
        x = self.mpdc(x)
        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)
        logits = self.seg_head(x)

        return logits


