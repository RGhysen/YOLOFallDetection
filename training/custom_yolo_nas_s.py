import torch, yaml
import torch
from super_gradients.common.registry.registry import register_model
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.utils.utils import HpmStruct

@register_model("custom_yolo_nas")
class CustomYoloNAS(CustomizableDetector):
    def __init__(self, num_classes: int, pretrained_weights: str = None):

        # load your YAML
        arch = yaml.safe_load(open("my_arch_params.yaml", "r"))
        params = HpmStruct(**arch).to_dict()
        backbone    = params["backbone"]
        heads       = params["heads"]
        neck        = params["neck"]
        bn_eps      = params["bn_eps"]
        bn_momentum = params["bn_momentum"]
        inplace_act = params["inplace_act"]
        in_channels = params["in_channels"]

        # call the parent ctor
        super().__init__(
            backbone,
            heads,
            neck,
            num_classes,
            bn_eps,
            bn_momentum,
            inplace_act,
            in_channels,
        )

        # optionally load weights
        if pretrained_weights:
            sd = torch.load(pretrained_weights, map_location="cpu")
            self.load_state_dict(sd, strict=False)
    
    def forward(self, images, targets=None):

        feats = self.backbone(images)           
        p3, p4, p5 = self.neck(feats)           
        detections = self.heads((p3, p4, p5))   
        return {"detections": detections}
    