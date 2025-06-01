import torch
import torch.nn.functional as F
from onnxruntime.quantization import quantize_dynamic, QuantType
from fusion_modelv2 import SpeedClassificationModule

CKPT_PATH = "7fps_checkpoint_epoch_10.pth"
ckpt = torch.load(CKPT_PATH, map_location="cpu")

model_fp32 = SpeedClassificationModule(
    feat_dim=5,
    hidden_dim=128,
    num_hidden_layers=3,
    dropout_p=0.0,
    pruning_lag=6
)
model_fp32.load_state_dict(ckpt["fusion_state_dict"])
model_fp32.eval()

class OnnxWrapper(torch.nn.Module):
    def __init__(self, base: SpeedClassificationModule):
        super().__init__()
        self.gru_cell   = base.gru_cell
        self.dropout    = base.dropout
        self.mlp_layers = base.mlp_layers
        self.fc         = base.fc

    def forward(self, feat, h_prev):
        # feat: [N, 5], h_prev: [N, 128]
        h = self.gru_cell(feat, h_prev)
        for layer in self.mlp_layers:
            h = layer(h).relu()
            h = self.dropout(h)
        logits = self.fc(h)
        probs  = F.softmax(logits, dim=-1)
        return probs, h

wrapper = OnnxWrapper(model_fp32)
wrapper.eval()

# Export FP32 ONNX
dummy_feat  = torch.randn(1, wrapper.gru_cell.input_size, dtype=torch.float32)
dummy_hprev = torch.zeros(1, wrapper.gru_cell.hidden_size, dtype=torch.float32)

torch.onnx.export(
    wrapper,
    (dummy_feat, dummy_hprev),
    "temporal_head_fp32_probs_7fps.onnx",
    opset_version=14,
    input_names   = ["feat",   "h_prev"],
    output_names  = ["probs",  "h_next"],
    dynamic_axes  = {
      "feat":   {0: "batch"},
      "h_prev": {0: "batch"},
      "probs":  {0: "batch"},
      "h_next": {0: "batch"},
    },
    export_params = True,
)
print("FP32 ONNX saved to temporal_head_fp32_probs_7fps.onnx")

# Quantize INT8
quantize_dynamic(
    "temporal_head_fp32_probs_7fps.onnx",
    "temporal_head_int8_probs_7fps.onnx",
    weight_type=QuantType.QInt8,
    optimize_model=True
)
print("INT8-quantized ONNX saved to temporal_head_int8_probs_7fps.onnx")
