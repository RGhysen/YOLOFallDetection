import os
import numpy as np
from PIL import Image
import onnx
import torch

from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.conversion import ExportTargetBackend
from super_gradients.training.processing import (
    ComposeProcessing,
    DetectionLongestMaxSizeRescale,
    ImagePermute,
)
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)

model = models.get(
    model_name=Models.YOLO_NAS_S,
    num_classes=2,
    checkpoint_path="ckpt_best.pth",
).eval()

preprocess = ComposeProcessing([
    DetectionLongestMaxSizeRescale(output_shape=[320, 320]),
    ImagePermute(),
])
postprocess = {"conf_thres": 0.25, "iou_thres": 0.45, "max_det": 100}

fp32_onnx = "yolo_nas_s_fp32.onnx"
model.export(
    output=fp32_onnx,
    engine=ExportTargetBackend.ONNXRUNTIME,
    input_image_shape=[320, 320],
    batch_size=1,
    preprocessing=preprocess,
    postprocessing=postprocess,
    onnx_export_kwargs={"opset_version": 14},
    onnx_simplify=True,
)
print(f"Exported FP32 ONNX to {fp32_onnx}")


onnx_model = onnx.load(fp32_onnx)
input_name = onnx_model.graph.input[0].name
print(f"Detected ONNX input name: '{input_name}'")


class FixedSizeCalibReader(CalibrationDataReader):
    def __init__(self, folder, input_name, size=(320, 320)):
        super().__init__()
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".bmp"))
        ]
        self.input_name = input_name
        self.size = size
        self._batches = None

    def get_next(self):
        if self._batches is None:
            self._batches = []
            for p in self.files:
                img = Image.open(p).convert("RGB")
                # resize to EXACTLY 320×320
                img = img.resize(self.size, Image.BILINEAR)
                arr = np.array(img).astype(np.float32) / 255.0 
                # HWC -> CHW
                arr = np.transpose(arr, (2, 0, 1))
                # add batch dim
                batch = arr[np.newaxis, ...]
                self._batches.append({self.input_name: batch})
        try:
            return self._batches.pop(0)
        except IndexError:
            return None

calib_folder = "own_data/full_dataset/images"
calib_reader = FixedSizeCalibReader(
    folder=calib_folder,
    input_name=input_name,
    size=(320, 320),
)

# Static INT8 quantization with ONNX Runtime
int8_onnx = "yolo_nas_s_int8.onnx"
quantize_static(
    model_input=fp32_onnx,
    model_output=int8_onnx,
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    optimize_model=False,
)

print(f"Created INT8 quantized ONNX at {int8_onnx}")
