"""
UNet with FiLM, GroupNorm, and CBAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CBAM


class DoubleConv(nn.Module):
    """Conv block: (Conv-GN-ReLU)x2 + CBAM."""

    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=32, use_cbam=False, cbam_ratio=16, cbam_kernel_size=7):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels)
        )
        
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(out_channels, ratio=cbam_ratio, kernel_size=cbam_kernel_size)
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.double_conv(x)
        if self.use_cbam:
            x = self.cbam(x)
        return self.relu(x)


class Down(nn.Module):
    """Downscaling."""

    def __init__(self, in_channels, out_channels, num_groups=32, use_cbam=False, cbam_ratio=16, cbam_kernel_size=7):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling."""

    def __init__(self, in_channels, out_channels, bilinear=True, num_groups=32, use_cbam=False, cbam_ratio=16, cbam_kernel_size=7):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FiLMLayer(nn.Module):
    """FiLM layer: y = gamma * x + beta."""
    
    def __init__(self, feature_channels):
        super().__init__()
        self.feature_channels = feature_channels
    
    def forward(self, features, film_params):
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        beta = beta.unsqueeze(2).unsqueeze(3)    # (B, C, 1, 1)
        
        return gamma * features + beta


class AuxiliaryMLP(nn.Module):
    """Auxiliary feature processing MLP."""
    
    def __init__(self, continuous_dim, nlcd_classes, nlcd_embedding_dim, 
                 ecoregion_classes, ecoregion_embedding_dim, hidden_dims, output_dim,
                 activation='relu', dropout_rates=None):
        super().__init__()
        
        # NLCD embedding
        self.nlcd_embed = nn.Embedding(nlcd_classes, nlcd_embedding_dim)
        self.ecoregion_embed = nn.Embedding(ecoregion_classes, ecoregion_embedding_dim)
        
        # Activation function mapping
        activations = {'relu': nn.ReLU(inplace=True), 'gelu': nn.GELU()}
        if activation not in activations:
            raise ValueError(f"Activation '{activation}' not supported. Use 'relu' or 'gelu'.")
        
        # MLP layers
        input_dim = continuous_dim + nlcd_embedding_dim + ecoregion_embedding_dim
        layers = []
        
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activations[activation])
            layers.append(nn.LayerNorm(hidden_dim))
            if dropout_rates and i < len(dropout_rates):
                layers.append(nn.Dropout(p=dropout_rates[i]))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, continuous, nlcd_idx, ecoregion_idx):
        nlcd_emb = self.nlcd_embed(nlcd_idx)
        ecoregion_emb = self.ecoregion_embed(ecoregion_idx)
        x = torch.cat([continuous, nlcd_emb, ecoregion_emb], dim=1)
        return self.mlp(x)


class UNetFiLM(nn.Module):
    """UNet with FiLM, GroupNorm, CBAM, and Dropout."""
    
    def __init__(self, config):
        super().__init__()
        
        # Config
        n_channels = config['model']['n_image_channels']
        n_classes = config['model']['n_output_channels']
        unet_channels = config['model']['unet_channels']
        bilinear = config['model']['bilinear']
        
        continuous_dim = config['model']['continuous_dim']
        nlcd_classes = config['model']['nlcd_classes']
        nlcd_embedding_dim = config['model']['nlcd_embedding_dim']
        ecoregion_classes = config['model']['ecoregion_classes']
        ecoregion_embedding_dim = config['model']['ecoregion_embedding_dim']
        mlp_hidden_dims = config['model']['mlp_hidden_dims']
        mlp_shared_dim = config['model']['mlp_shared_dim']
        
        # Architecture config with defaults
        dropout_rates = config['model'].get('dropout_rates', [0.1, 0.1, 0.15, 0.2])
        num_groups = config['model'].get('num_groups', 32)
        
        # New MLP and CBAM configs with defaults for backward compatibility
        mlp_activation = config['model'].get('mlp_activation', 'relu')
        mlp_dropout_rates = config['model'].get('mlp_dropout_rates', None)
        use_cbam = config['model'].get('use_cbam', False)
        cbam_ratio = config['model'].get('cbam_ratio', 8)
        cbam_kernel_size = config['model'].get('cbam_kernel_size', 7)
        
        # UNet encoder (add CBAM parameters)
        self.inc = DoubleConv(n_channels, unet_channels[0], num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
        self.down1 = Down(unet_channels[0], unet_channels[1], num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
        self.down2 = Down(unet_channels[1], unet_channels[2], num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
        self.down3 = Down(unet_channels[2], unet_channels[3], num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
        factor = 2 if bilinear else 1
        self.down4 = Down(unet_channels[3], unet_channels[4] // factor, num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
        
        # UNet decoder (add CBAM parameters)
        self.up1 = Up(unet_channels[4], unet_channels[3] // factor, bilinear, num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
        self.up2 = Up(unet_channels[3], unet_channels[2] // factor, bilinear, num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
        self.up3 = Up(unet_channels[2], unet_channels[1] // factor, bilinear, num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
        self.up4 = Up(unet_channels[1], unet_channels[0], bilinear, num_groups=num_groups, use_cbam=use_cbam, cbam_ratio=cbam_ratio, cbam_kernel_size=cbam_kernel_size)
        self.outc = nn.Conv2d(unet_channels[0], n_classes, kernel_size=1)
        
        # Spatial dropout layers (only in encoder, after FiLM)
        self.spatial_dropout = nn.ModuleDict({
            'down1': nn.Dropout2d(p=dropout_rates[0]),
            'down2': nn.Dropout2d(p=dropout_rates[1]), 
            'down3': nn.Dropout2d(p=dropout_rates[2]),
            'bottleneck': nn.Dropout2d(p=dropout_rates[3])
        })
        
        # Auxiliary MLP (pass new parameters)
        self.aux_mlp = AuxiliaryMLP(
            continuous_dim, nlcd_classes, nlcd_embedding_dim,
            ecoregion_classes, ecoregion_embedding_dim,
            mlp_hidden_dims, mlp_shared_dim,
            activation=mlp_activation,
            dropout_rates=mlp_dropout_rates
        )
        
        # FiLM layers and projections
        film_channels = [
            unet_channels[0],  # After inc
            unet_channels[1],  # After down1
            unet_channels[2],  # After down2
            unet_channels[3],  # After down3
            unet_channels[4] // factor,  # Bottleneck
            unet_channels[3] // factor,  # After up1
            unet_channels[2] // factor,  # After up2
            unet_channels[1] // factor,  # After up3
            unet_channels[0]   # After up4
        ]
        
        self.film_layers = nn.ModuleList([
            FiLMLayer(channels) for channels in film_channels
        ])
        
        self.film_projections = nn.ModuleList([
            nn.Linear(mlp_shared_dim, 2 * channels)
            for channels in film_channels
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        self._init_film_projections()
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Default linear layer initialization
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _init_film_projections(self):
        """Initialize FiLM projections (near-identity)."""
        for proj in self.film_projections:
            # Small weights
            nn.init.normal_(proj.weight, mean=0, std=0.02)
            
            # Init bias: gamma=1, beta=0
            with torch.no_grad():
                output_size = proj.out_features
                proj.bias[:output_size//2] = 1.0
                proj.bias[output_size//2:] = 0.0
        
        # Init output
        nn.init.normal_(self.outc.weight, mean=0, std=0.001)
        nn.init.constant_(self.outc.bias, 15.0)
    
    def freeze_unet_weights(self):
        """Freeze UNet weights."""
        print("Freezing all UNet weights.")
        
        unet_modules = [
            self.inc, self.down1, self.down2, self.down3, self.down4,
            self.up1, self.up2, self.up3, self.up4, self.outc
        ]
        
        for module in unet_modules:
            for param in module.parameters():
                param.requires_grad = False
        
        frozen_params = sum(p.numel() for m in unet_modules for p in m.parameters())
        print(f"Froze {frozen_params:,} UNet parameters.")

    def freeze_film_weights(self):
        """Freeze FiLM weights."""
        print("Freezing FiLM weights.")
        
        # Freeze MLP
        for param in self.aux_mlp.parameters():
            param.requires_grad = False
        
        # Freeze projections
        for projection in self.film_projections:
            for param in projection.parameters():
                param.requires_grad = False
        
        frozen_params = sum(p.numel() for p in self.aux_mlp.parameters()) + \
                       sum(p.numel() for proj in self.film_projections for p in proj.parameters())
        print(f"Frozen {frozen_params:,} FiLM parameters")
    
    def unfreeze_film_weights(self):
        """Unfreeze FiLM weights."""
        print("Unfreezing FiLM weights.")
        
        # Unfreeze MLP
        for param in self.aux_mlp.parameters():
            param.requires_grad = True
        
        # Unfreeze projections
        for projection in self.film_projections:
            for param in projection.parameters():
                param.requires_grad = True
        
        unfrozen_params = sum(p.numel() for p in self.aux_mlp.parameters()) + \
                         sum(p.numel() for proj in self.film_projections for p in proj.parameters())
        print(f"Unfrozen {unfrozen_params:,} FiLM parameters")
    
    def get_trainable_parameters(self):
        """Get trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def print_parameter_status(self):
        """Print freeze status."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Parameter Status:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {frozen_params:,}")
        print(f"  Frozen %: {frozen_params/total_params*100:.1f}%")

    def forward(self, image, continuous, nlcd_idx, ecoregion_idx):
        # Aux features
        aux_repr = self.aux_mlp(continuous, nlcd_idx, ecoregion_idx)
        
        # FiLM params
        film_params = [proj(aux_repr) for proj in self.film_projections]
        
        # Encoder with FiLM and dropout
        x1 = self.inc(image)
        x1 = self.film_layers[0](x1, film_params[0])
        
        x2 = self.down1(x1)
        x2 = self.film_layers[1](x2, film_params[1])
        if self.training:  # Only apply dropout during training
            x2 = self.spatial_dropout['down1'](x2)
        
        x3 = self.down2(x2)
        x3 = self.film_layers[2](x3, film_params[2])
        if self.training:
            x3 = self.spatial_dropout['down2'](x3)
        
        x4 = self.down3(x3)
        x4 = self.film_layers[3](x4, film_params[3])
        if self.training:
            x4 = self.spatial_dropout['down3'](x4)
        
        x5 = self.down4(x4)
        x5 = self.film_layers[4](x5, film_params[4])
        if self.training:
            x5 = self.spatial_dropout['bottleneck'](x5)
        
        # Decoder with FiLM (no dropout in decoder)
        x = self.up1(x5, x4)
        x = self.film_layers[5](x, film_params[5])
        
        x = self.up2(x, x3)
        x = self.film_layers[6](x, film_params[6])
        
        x = self.up3(x, x2)
        x = self.film_layers[7](x, film_params[7])
        
        x = self.up4(x, x1)
        x = self.film_layers[8](x, film_params[8])
        
        # Output
        logits = self.outc(x)
        return logits


def create_model(config):
    """Create model."""
    return UNetFiLM(config)


if __name__ == "__main__":
    # Test model creation
    import yaml
    
    # Mock config
    config = {
        'model': {
            'n_image_channels': 4,
            'n_output_channels': 1,
            'unet_channels': [64, 128, 256, 512, 1024],
            'bilinear': True,
            'continuous_dim': 15,
            'nlcd_classes': 9,
            'nlcd_embedding_dim': 8,
            'ecoregion_classes': 86,
            'ecoregion_embedding_dim': 16,
            'mlp_hidden_dims': [128, 256, 128],
            'mlp_shared_dim': 128,
            'film_stages': 9
        }
    }
    
    # Create model
    model = create_model(config)
    
    # Test forward pass
    batch_size = 2
    image = torch.randn(batch_size, 4, 432, 432)
    continuous = torch.randn(batch_size, 15)
    nlcd_idx = torch.randint(0, 9, (batch_size,))
    ecoregion_idx = torch.randint(0, 86, (batch_size,))
    
    output = model(image, continuous, nlcd_idx, ecoregion_idx)
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
