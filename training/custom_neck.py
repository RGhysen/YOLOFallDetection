import torch
import torch.nn as nn
from typing import Union, List, Tuple
from omegaconf import DictConfig
from torch import Tensor
from super_gradients.modules.detection_modules import BaseDetectionModule
from super_gradients.training.utils.utils import HpmStruct
import super_gradients.common.factories.detection_modules_factory as det_factory
from super_gradients.common.registry.registry import register_detection_module
from collections import deque

class TemporalBlockWrapper(nn.Module):
    """
    Wraps a temporal block by accumulating a fixed number of frames before processing.
    """
    def __init__(self, temporal_block, window_size):
        """
        :param temporal_block: TemporalBlock3D that processes a stack of frames.
        :param window_size: Number of frames to accumulate before temporal processing.
        """
        super().__init__()
        self.temporal_block = temporal_block
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.alpha = 1.0

    def forward(self, x):
        # detach and clone so we don't hold the old graph
        x = x.detach().clone().requires_grad_(True)
        self.buffer.append(x)

        if len(self.buffer) < self.window_size:
            return x  # just pass current frame through
        
        # when buffer is full: stack along time dimension  [B, T, C, H, W]
        buf_list = list(self.buffer)  
        buf_stack = torch.stack(buf_list, dim=1) 

        # run temporal block
        fused = self.temporal_block(buf_stack)   
        
        # add it back onto the *raw* feature of frame t
        x_fused = x + self.alpha * fused
        return x_fused
    
    def reset(self):
        self.buffer.clear()

@register_detection_module(name="YoloNASPANNeckWithTemporalBuffer")
class YoloNASPANNeckWithTemporalBuffer(BaseDetectionModule):
    """
    The original PAN neck that processes frames individually until the temporal block.
    For each of the three outputs (p3, p4, p5) from the PAN neck, a corresponding
    TemporalBlockWrapper accumulates a fixed number of frames and then processes them together.
    """
    def __init__(
        self,
        in_channels: List[int],
        neck1: Union[str, HpmStruct, DictConfig],
        neck2: Union[str, HpmStruct, DictConfig],
        neck3: Union[str, HpmStruct, DictConfig],
        neck4: Union[str, HpmStruct, DictConfig],
        temporal_block1: Union[str, HpmStruct, DictConfig],
        temporal_block2: Union[str, HpmStruct, DictConfig],
        temporal_block3: Union[str, HpmStruct, DictConfig],
        temporal_window: int = 12
    ):
        """
        Initialize the PAN neck with three temporal wrappers.
        :param in_channels: Input channels of the 4 feature maps from the backbone.
        :param neck1: First neck stage configuration.
        :param neck2: Second neck stage configuration.
        :param neck3: Third neck stage configuration.
        :param neck4: Fourth neck stage configuration.
        :param temporal_block1: Temporal block configuration for the p3 feature map.
        :param temporal_block2: Temporal block configuration for the p4 feature map.
        :param temporal_block3: Temporal block configuration for the p5 feature map.
        :param temporal_window: Number of frames to accumulate before processing.
        """
        super().__init__(in_channels)
        c2_out_channels, c3_out_channels, c4_out_channels, c5_out_channels = in_channels

        factory = det_factory.DetectionModulesFactory()
        self.neck1 = factory.get(factory.insert_module_param(neck1, "in_channels", [c5_out_channels, c4_out_channels, c3_out_channels]))
        self.neck2 = factory.get(factory.insert_module_param(neck2, "in_channels", [self.neck1.out_channels[1], c3_out_channels, c2_out_channels]))
        self.neck3 = factory.get(factory.insert_module_param(neck3, "in_channels", [self.neck2.out_channels[1], self.neck2.out_channels[0]]))
        self.neck4 = factory.get(factory.insert_module_param(neck4, "in_channels", [self.neck3.out_channels, self.neck1.out_channels[0]]))

        tb1 = factory.get(factory.insert_module_param(temporal_block1, "in_channels", self.neck2.out_channels[1]))
        tb2 = factory.get(factory.insert_module_param(temporal_block2, "in_channels", self.neck3.out_channels))
        tb3 = factory.get(factory.insert_module_param(temporal_block3, "in_channels", self.neck4.out_channels))

        self.temporal_wrapper1 = TemporalBlockWrapper(tb1, temporal_window)
        self.temporal_wrapper2 = TemporalBlockWrapper(tb2, temporal_window)
        self.temporal_wrapper3 = TemporalBlockWrapper(tb3, temporal_window)

        self._out_channels = [
            self.temporal_wrapper1.temporal_block.out_channels,
            self.temporal_wrapper2.temporal_block.out_channels,
            self.temporal_wrapper3.temporal_block.out_channels,
        ]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for a single frame input.
        :param inputs: A tuple of 4 tensors (c2, c3, c4, c5) from the backbone,
                       each with shape [B, C, H, W].
        :return: A tuple of 3 tensors corresponding to the outputs of the temporal wrappers.
        """
        c2, c3, c4, c5 = inputs

        # Process through the PAN neck stages for the current frame.
        x_n1_inter, x = self.neck1([c5, c4, c3])
        x_n2_inter, p3 = self.neck2([x, c3, c2])
        p4 = self.neck3([p3, x_n2_inter])
        p5 = self.neck4([p4, x_n1_inter])

        # Process each output through its corresponding temporal block wrapper.
        p3_temp = self.temporal_wrapper1(p3)
        p4_temp = self.temporal_wrapper2(p4)
        p5_temp = self.temporal_wrapper3(p5)

        return p3_temp, p4_temp, p5_temp
