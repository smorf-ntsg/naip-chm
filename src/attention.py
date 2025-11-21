"""
CBAM: Convolutional Block Attention Module
Implementation for integrating channel and spatial attention into UNet blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module of CBAM"""
    
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module of CBAM"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module
    
    Combines channel and spatial attention to enhance feature representation.
    Applied after convolution layers to refine feature maps.
    
    Args:
        in_planes (int): Number of input channels
        ratio (int): Channel attention reduction ratio (default: 16)
        kernel_size (int): Spatial attention kernel size, 3 or 7 (default: 7)
    """
    
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention
        x = self.ca(x) * x
        # Apply spatial attention
        x = self.sa(x) * x
        return x


if __name__ == "__main__":
    # Test CBAM module
    print("Testing CBAM module...")
    
    # Test with different channel sizes
    test_cases = [
        (64, 32, 32),   # Typical early UNet features
        (128, 16, 16),  # Mid-level features
        (512, 8, 8),    # Deep features
        (1024, 4, 4)    # Bottleneck features
    ]
    
    for channels, h, w in test_cases:
        print(f"\nTesting CBAM with {channels} channels, {h}x{w} spatial size...")
        
        # Create test input
        x = torch.randn(2, channels, h, w)  # Batch size 2
        
        # Test with default parameters
        cbam = CBAM(channels)
        output = cbam(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in cbam.parameters()):,}")
        
        # Verify output shape matches input
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        
        # Test with different ratio
        cbam_r8 = CBAM(channels, ratio=8)
        output_r8 = cbam_r8(x)
        print(f"  Parameters (ratio=8): {sum(p.numel() for p in cbam_r8.parameters()):,}")
        
        # Test with different kernel size
        cbam_k3 = CBAM(channels, kernel_size=3)
        output_k3 = cbam_k3(x)
        print(f"  Parameters (kernel=3): {sum(p.numel() for p in cbam_k3.parameters()):,}")
    
    print("\n✅ CBAM module tests passed!")
