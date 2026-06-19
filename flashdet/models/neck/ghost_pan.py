"""
GhostPAN Feature Pyramid Network for FlashDet.
Matches official FlashDet implementation.
"""

import math
import torch
import torch.nn as nn

from .conv_module import ConvModule, DepthwiseConvModule


class GhostModule(nn.Module):
    """
    Ghost module for efficient feature extraction.
    Reference: GhostNet (https://arxiv.org/abs/1911.11907)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        activation: str = "LeakyReLU"
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        # Handle activation - None means no activation (linear projection)
        if activation is None:
            act = nn.Identity()
        elif activation == "LeakyReLU":
            act = nn.LeakyReLU(0.1, inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            act,
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            act,
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        # Official FlashDet returns full concat without slicing
        # For standard configs (even out_channels, ratio=2), this equals out_channels
        return out


class GhostBottleneck(nn.Module):
    """
    Ghost bottleneck block.
    Reference: GhostNet (https://arxiv.org/abs/1911.11907)
    """
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dw_kernel_size: int = 3,
        stride: int = 1,
        activation: str = "LeakyReLU"
    ):
        super().__init__()
        self.stride = stride
        
        # Point-wise expansion
        self.ghost1 = GhostModule(in_channels, mid_channels, activation=activation)
        
        # Depth-wise convolution
        if stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_channels, mid_channels, dw_kernel_size,
                stride=stride, padding=(dw_kernel_size - 1) // 2,
                groups=mid_channels, bias=False
            )
            self.bn_dw = nn.BatchNorm2d(mid_channels)
        
        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_channels, out_channels, activation=None)
        
        # Shortcut
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels, dw_kernel_size,
                    stride=stride, padding=(dw_kernel_size - 1) // 2,
                    groups=in_channels, bias=False
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        residual = x
        
        # 1st ghost bottleneck
        x = self.ghost1(x)
        
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        
        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        # Shortcut
        x = x + self.shortcut(residual)
        return x


class GhostBlocks(nn.Module):
    """Stack of GhostBottleneck blocks used in GhostPAN."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand: int = 1,
        kernel_size: int = 5,
        num_blocks: int = 1,
        use_res: bool = False,
        activation: str = "LeakyReLU"
    ):
        super().__init__()
        self.use_res = use_res
        
        if use_res:
            self.reduce_conv = ConvModule(
                in_channels, out_channels, 1, activation=activation
            )
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                GhostBottleneck(
                    in_channels, int(out_channels * expand), out_channels,
                    dw_kernel_size=kernel_size, activation=activation
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        out = self.blocks(x)
        if self.use_res:
            out = out + self.reduce_conv(x)
        return out


class GhostPAN(nn.Module):
    """
    Ghost Path Aggregation Network (PAN) for feature fusion.
    Matches official FlashDet implementation.
    
    Args:
        in_channels: Input channel list from backbone.
        out_channels: Output channels for all levels.
        kernel_size: Convolution kernel size.
        num_extra_level: Number of extra output levels.
        use_depthwise: Whether to use depthwise separable conv.
        activation: Activation function name.
    """
    
    def __init__(
        self,
        in_channels: list,
        out_channels: int = 96,
        kernel_size: int = 5,
        expand: int = 1,
        num_blocks: int = 1,
        use_res: bool = False,
        num_extra_level: int = 0,
        use_depthwise: bool = True,
        activation: str = "LeakyReLU"
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Official uses nn.Upsample(scale_factor=2, mode="bilinear") as a registered sub-module
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        Conv = DepthwiseConvModule if use_depthwise else ConvModule

        # Reduce layers (lateral connections)
        self.reduce_layers = nn.ModuleList()
        for ch in in_channels:
            self.reduce_layers.append(
                ConvModule(ch, out_channels, 1, activation=activation)
            )
        
        # Top-down blocks
        self.top_down_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.top_down_blocks.append(
                GhostBlocks(
                    out_channels * 2, out_channels, expand,
                    kernel_size=kernel_size, num_blocks=num_blocks,
                    use_res=use_res, activation=activation
                )
            )
        
        # Bottom-up path
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsamples.append(
                Conv(out_channels, out_channels, kernel_size, stride=2, activation=activation)
            )
            self.bottom_up_blocks.append(
                GhostBlocks(
                    out_channels * 2, out_channels, expand,
                    kernel_size=kernel_size, num_blocks=num_blocks,
                    use_res=use_res, activation=activation
                )
            )
        
        # Extra levels for additional feature scales
        # Each extra level takes input from both the backbone's last feature and PAN's last output
        self.num_extra_level = num_extra_level
        self.extra_lvl_in_conv = nn.ModuleList()
        self.extra_lvl_out_conv = nn.ModuleList()
        for i in range(num_extra_level):
            self.extra_lvl_in_conv.append(
                Conv(out_channels, out_channels, kernel_size, stride=2, activation=activation)
            )
            self.extra_lvl_out_conv.append(
                Conv(out_channels, out_channels, kernel_size, stride=2, activation=activation)
            )
    
    def forward(self, inputs: list) -> list:
        """
        Forward pass.
        
        Args:
            inputs: List of feature maps from backbone.
            
        Returns:
            List of fused feature maps.
        """
        assert len(inputs) == len(self.in_channels)
        
        # Reduce channels
        inputs = [reduce(x) for reduce, x in zip(self.reduce_layers, inputs)]
        
        # Top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            
            # Upsample (matches official: nn.Upsample scale_factor=2)
            upsample_feat = self.upsample(feat_high)
            
            # Concat and process
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)
        
        # Bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            
            # Downsample
            downsample_feat = self.downsamples[idx](feat_low)
            
            # Concat and process
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], dim=1)
            )
            outs.append(out)
        
        # Extra levels: combine downsampled backbone feature and PAN output
        # This creates additional high-stride feature maps for detecting small objects
        for i, (extra_in, extra_out) in enumerate(zip(self.extra_lvl_in_conv, self.extra_lvl_out_conv)):
            # extra_in processes the reduced backbone feature (from inputs)
            # extra_out processes the last PAN output
            # Sum them together for the extra level output
            outs.append(extra_in(inputs[-1]) + extra_out(outs[-1]))
        
        return outs
