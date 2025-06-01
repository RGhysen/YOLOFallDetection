import numpy as np
from super_gradients.training import models
import torch
from super_gradients.training.processing import (
    DetectionLongestMaxSizeRescale,
    DetectionCenterPadding,
    StandardizeImage,
    ImagePermute,
    ComposeProcessing
)

# Load trained weights instead of COCO
base_model = models.get(
    "yolo_nas_s",
    checkpoint_path="ckpt_best.pth",
    num_classes=2
)
base_model.eval()
for p in base_model.parameters():
    p.requires_grad = False

class_names = ["fall", "stand"] 
image_processor = ComposeProcessing([
    DetectionLongestMaxSizeRescale(output_shape=(320, 320)),
    #DetectionCenterPadding(output_shape=(320, 320), pad_value=114),
    #StandardizeImage(max_value=255.0),
    #ImagePermute(permutation=(2, 0, 1)),
])

base_model.set_dataset_processing_params(
    class_names=class_names,
    image_processor=image_processor,
)

def frame_predict(input_frame: np.ndarray, conf_threshold=0.8, iou_threshold=0.5):
    preds = base_model.predict(
        input_frame, conf=conf_threshold, iou=iou_threshold, fuse_model=False
    ).prediction

    bboxes, labels, confs = preds.bboxes_xyxy, preds.labels, preds.confidence

    out = []
    for box, lab, conf in zip(bboxes, labels, confs):
        out.append([ box[0], box[1], box[2], box[3], lab, conf])
    return np.stack(out, 0) if out else np.empty((0,6), float)