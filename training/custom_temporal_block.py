import torch
import torch.nn as nn
from super_gradients.common.registry.registry import register_detection_module

class ChompTemporal(nn.Module):
    """
    Removes extra elements along the temporal dimension to maintain causality.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, T, H, W]
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size, :, :]
        return x

@register_detection_module("TemporalBlock3D")
class TemporalBlock3D(nn.Module):
    """
    A temporal block for 3D feature maps.
    Processes a temporal stack of features (shape [B, T, C, H, W]), then aggregates
    the temporal dimension to produce an output of shape [B, C_out, H, W].
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        temporal_kernel_size,
        stride,
        dilation,
        dropout,
    ):
        super().__init__()
        # Compute necessary padding for the temporal dimension.
        # This padding value ensures the conv layer is causal.
        self.temporal_padding = (temporal_kernel_size - 1) * dilation
        
        # First temporal convolution:
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            stride=(stride, 1, 1),
            padding=(self.temporal_padding, 0, 0),
            dilation=(dilation, 1, 1)
        )
        self.chomp1 = ChompTemporal(self.temporal_padding)
        self.relu1 = nn.ReLU(inplace=False)
        self.dropout1 = nn.Dropout3d(dropout)
        
        # Second temporal convolution:
        self.conv2 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            stride=(stride, 1, 1),
            padding=(self.temporal_padding, 0, 0),
            dilation=(dilation, 1, 1)
        )
        self.chomp2 = ChompTemporal(self.temporal_padding)
        self.relu2 = nn.ReLU(inplace=False)
        self.dropout2 = nn.Dropout3d(dropout)
        
        # Residual connection: if the number of input channels differs, downsample.
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU(inplace=False)
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape [B, T, C, H, W].
        :return: Aggregated feature map of shape [B, n_outputs, H, W].
        """
        # Permute to [B, C, T, H, W] since Conv3d expects the channels dimension in position 1.
        x = x.permute(0, 2, 1, 3, 4).detach().clone()
        
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # To ensure matching temporal dimensions, trim if necessary:
        T_min = min(out.size(2), residual.size(2))
        out = out[:, :, :T_min, :, :]
        residual = residual[:, :, :T_min, :, :]
        
        out = self.final_relu(out + residual)
        
        # Aggregate the temporal dimension.
        out = out[:, :, -1, :, :]
        return out
