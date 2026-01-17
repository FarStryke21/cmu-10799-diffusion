"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # TODO: build your own unet architecture here
        # Pro tips: remember to take care of the time embeddings!
        # 1. Time Embedding MLP
        # Maps time scalar -> vector of size 4*base_channels
        time_embed_dim = base_channels * 4
        self.time_embedding = TimestepEmbedding(time_embed_dim)

        # 2. Initial Convolution
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 3. Downsampling Path (Encoder)
        self.down_blocks = nn.ModuleList()
        
        # We need to track the number of channels at each step for skip connections
        ch = base_channels # Current number of channels
        input_block_ch = [base_channels] # To store channels for skip connections
        ds = 1 # Downsampling factor
        
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, 
                        int(out_ch),
                        time_embed_dim, 
                        dropout, 
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = out_ch
                
                if ds in [1, 2, 4, 8] and (64 // ds) in attention_resolutions: 
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                
                self.down_blocks.append(nn.Sequential(*layers))
                input_block_ch.append(ch)
            
            # Downsample (except at the last level)
            if i != len(channel_mult) - 1:
                self.down_blocks.append(Downsample(ch))
                input_block_ch.append(ch)
                ds *= 2 # Update downsampling factor

        # 4. Middle Block (Bottleneck)
        self.middle_block = nn.Sequential(
            ResBlock(ch, int(ch), time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, int(ch), time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
        )
        
        # 5. Upsampling Path (Decoder)
        self.up_blocks = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks + 1):
                # +1 because we concatenate skip connection
                skip_ch = input_block_ch.pop()
                layers = [
                    ResBlock(
                        ch + skip_ch, 
                        out_ch,
                        time_embed_dim, 
                        dropout, 
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = out_ch
                
                # Add Attention
                if ds in [1, 2, 4, 8] and (64 // ds) in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                
                self.up_blocks.append(nn.Sequential(*layers))
                
            # Upsample
            if i != 0:
                self.up_blocks.append(Upsample(ch))
                ds //= 2

        # 6. Final Output
        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

        # Zero-out the last layer so the model initially predicts 0 residuals
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement the forward pass of the unet
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """

        # 1. Time Embedding
        t_emb = self.time_embedding(t)
        
        # 2. Initial Conv
        h = self.head(x)
        
        # Store skip connections
        hs = [h]
        
        # 3. Downsampling
        for module in self.down_blocks:
            if isinstance(module, Downsample):
                h = module(h)
            else:
                # It's a Sequential of ResBlock + (Optional) Attention
                # We need to manually dig into Sequential to pass t_emb to ResBlock
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            hs.append(h)
            
        # 4. Middle
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
                
        # 5. Upsampling
        for module in self.up_blocks:
            if isinstance(module, Upsample):
                h = module(h)
            else:
                # Concatenate skip connection
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
                        
        # 6. Final projection
        return self.out(h)


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
    return UNet(
        in_channels=data_config['channels'],
        out_channels=data_config['channels'],
        base_channels=model_config['base_channels'],
        channel_mult=tuple(model_config['channel_mult']),
        num_res_blocks=model_config['num_res_blocks'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
