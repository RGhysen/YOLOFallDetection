# export_fusion.py
import torch
import torch.nn as nn
from fusion_modelv2 import SpeedClassificationModule

ckpt = torch.load("v2_checkpoint_epoch_3.pth", map_location="cpu")
cfg = ckpt["cfg"]
fusion_state = ckpt["fusion_state_dict"]

fusion = SpeedClassificationModule(
    feat_dim        = 5,
    hidden_dim      = cfg["hidden_dim"],    
    num_hidden_layers = cfg["num_hidden_layers"],
    dropout_p       = cfg["dropout_p"],
    pruning_lag     = cfg["pruning_lag"]
)
fusion.load_state_dict(fusion_state)
fusion.eval()
fusion.reset()

# Wrapping only the pure compute (forward_cell) so ONNX sees feat->logits
class FusionONNXWrapper(nn.Module):
    def __init__(self, fusion_module):
        super().__init__()
        self.fusion = fusion_module

    def forward(self, feat: torch.Tensor):
        # forward_cell returns a [1×2] tensor of logits
        return self.fusion.forward_cell(feat, tid=0, device=feat.device)

wrapper = FusionONNXWrapper(fusion)

dummy_input = torch.randn(1, 5)  # batch=1, feat_dim=5
torch.onnx.export(
    wrapper,
    dummy_input,
    "fusion_head.onnx",
    opset_version=13,
    input_names   = ["feat"],
    output_names  = ["logits"],
    dynamic_axes  = {"feat":   {0: "batch"},
                     "logits": {0: "batch"}}
)
print("✔️ Exported fusion_head.onnx")
