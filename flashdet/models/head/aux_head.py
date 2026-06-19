"""
Auxiliary Head for FlashDet training.
"""

import torch
import torch.nn as nn
from typing import List


class ConvModule(nn.Module):
    """Standard convolution with BN and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = None,
        activation: str = "LeakyReLU"
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "LeakyReLU":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "ReLU":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SimpleConvHead(nn.Module):
    """
    Simple convolutional head used as auxiliary head in FlashDet.
    
    This head is only used during training to provide auxiliary supervision.
    It helps with target assignment by providing additional predictions.
    
    Args:
        num_classes: Number of detection classes.
        input_channel: Input feature channel.
        feat_channels: Feature channels in head.
        stacked_convs: Number of stacked conv layers.
        strides: Feature map strides.
        reg_max: Max value for distribution focal loss.
        activation: Activation function name.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_channel: int = 192,
        feat_channels: int = 192,
        stacked_convs: int = 4,
        strides: List[int] = None,
        reg_max: int = 7,
        activation: str = "LeakyReLU",
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = strides or [8, 16, 32, 64]
        self.reg_max = reg_max
        
        # Shared conv layers
        convs = []
        for i in range(stacked_convs):
            in_ch = input_channel if i == 0 else feat_channels
            convs.append(ConvModule(in_ch, feat_channels, 3, activation=activation))
        self.cls_convs = nn.Sequential(*convs)
        
        # Output layer
        self.gfl_cls = nn.Conv2d(
            feat_channels, num_classes + 4 * (reg_max + 1), 1
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Init cls head with confidence = 0.01
        bias_cls = -4.595
        nn.init.normal_(self.gfl_cls.weight, std=0.01)
        nn.init.constant_(self.gfl_cls.bias, bias_cls)
    
    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feats: Multi-scale feature maps. Should be concatenated FPN features.
            
        Returns:
            Predictions tensor [B, num_points, num_classes + 4*(reg_max+1)].
        """
        outputs = []
        for feat in feats:
            feat = self.cls_convs(feat)
            output = self.gfl_cls(feat)
            outputs.append(output.flatten(start_dim=2))
        
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs
