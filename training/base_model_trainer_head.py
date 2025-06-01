import os
import yaml
import torch
import wandb
import multiprocessing
from torch.utils.data import DataLoader, IterableDataset
from super_gradients.training import Trainer, models
from super_gradients.common.sg_loggers.wandb_sg_logger import WandBSGLogger, WANDB_ID_PREFIX
from super_gradients.common.object_names import Models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from data_splitter import train_loader, val_loader

def _set_wandb_id_no_mknod(self, wandb_id):
    path = os.path.join(self._local_dir, f"{WANDB_ID_PREFIX}{wandb_id}")
    with open(path, "w"): pass
WandBSGLogger._set_wandb_id = _set_wandb_id_no_mknod
multiprocessing.set_start_method("fork", force=True)
_original_add_config = WandBSGLogger.add_config
def _add_config_allow_change(self, key, config_dict):
    wandb.config.update({key: config_dict}, allow_val_change=True)
WandBSGLogger.add_config = _add_config_allow_change

def deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            a[k] = deep_update(a[k], v)
        else:
            a[k] = v
    return a

default_cfg = yaml.safe_load(open("default_train_params.yaml"))
coco_cfg    = yaml.safe_load(open("coco2017_yolo_nas_train_params.yaml"))
cfg         = deep_update(default_cfg, coco_cfg)

BATCH_SIZE      = 128
INITIAL_LR      = float(0.0016)
WEIGHT_DECAY    = cfg["optimizer_params"]["weight_decay"]
OPTIMIZER_NAME  = cfg["optimizer"].lower()

th = cfg
MAX_EPOCHS      = th["max_epochs"]
WARMUP_MODE     = th["warmup_mode"]
LR_MODE         = th["lr_mode"]
LR_WARMUP_EPOCHS= th["lr_warmup_epochs"]
WARMUP_INITIAL_LR = float(th["warmup_initial_lr"])
COSINE_FINAL_LR = th["cosine_final_lr_ratio"]
MIXED_PRECISION = th["mixed_precision"]
SILENT_MODE     = th["silent_mode"]
SG_LOGGER       = th.get("sg_logger", "wandb_sg_logger")
SG_LOGGER_PARAMS= th.get("sg_logger_params", {})

class FrameLevelDataset(IterableDataset):
    def __init__(self, video_loader):
        self.video_loader = video_loader
        self.video_ds     = video_loader.dataset
    def __iter__(self):
        for batch in self.video_loader:
            imgs, tgs = batch["images"][0], batch["targets"][0]
            import random; idxs = list(range(len(imgs))); random.shuffle(idxs)
            for i in idxs: yield imgs[i], tgs[i]
    def __len__(self):
        return sum(len(frames) for frames in self.video_ds.video_dict.values())

def detection_collate_fn(batch):
    imgs, tgts = zip(*batch)
    imgs = torch.stack(imgs, 0)
    all_t = []
    for i, tgt in enumerate(tgts):
        boxes, labels = tgt["boxes"], tgt["labels"]
        x1,y1,x2,y2 = boxes.unbind(1)
        w, h        = x2 - x1, y2 - y1
        cx, cy      = x1 + w/2, y1 + h/2
        idxs        = torch.full((boxes.size(0),1), i, device=boxes.device, dtype=boxes.dtype)
        labs        = labels.unsqueeze(1).to(boxes.dtype)
        all_t.append(torch.cat([idxs, labs, cx.unsqueeze(1), cy.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1)], dim=1))
    targets = torch.cat(all_t,0) if all_t else torch.zeros((0,6), dtype=torch.float32, device=imgs.device)
    return imgs, targets

frame_train_loader = DataLoader(FrameLevelDataset(train_loader), batch_size=BATCH_SIZE, collate_fn=detection_collate_fn)
frame_val_loader   = DataLoader(FrameLevelDataset(val_loader),   batch_size=BATCH_SIZE, collate_fn=detection_collate_fn)



if __name__ == "__main__":
    run = wandb.init(project=SG_LOGGER_PARAMS.get("project_name","yolo-base-trainer"),
                     entity=SG_LOGGER_PARAMS.get("entity",None),
                     config=cfg)

    net = models.get(Models.YOLO_NAS_S, pretrained_weights="coco", num_classes=2)

    train_params = {
        "finetune": True,
        "run_validation_freq": 1,
        "average_best_models": False,
        "optimizer": OPTIMIZER_NAME,
        "optimizer_params": {"weight_decay": WEIGHT_DECAY},
        "initial_lr": INITIAL_LR,
        "lr_mode":    LR_MODE,
        "warmup_mode":      WARMUP_MODE,
        "lr_warmup_epochs": LR_WARMUP_EPOCHS,
        "cosine_final_lr_ratio": COSINE_FINAL_LR,
        "max_epochs": MAX_EPOCHS,
        "mixed_precision": MIXED_PRECISION,
        "silent_mode":     SILENT_MODE,
        "warmup_initial_lr": WARMUP_INITIAL_LR,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=2,
            reg_max=16,
            #iou_loss_weight=0.0,
            #dfl_loss_weight=0.0,
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=2,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": "F1@0.50",
        "sg_logger":      SG_LOGGER,
        "sg_logger_params": SG_LOGGER_PARAMS,
    }

    trainer = Trainer(
        experiment_name='yolo-base-final',
        ckpt_root_dir="base_model_head",
    )
    trainer.train(
        model=net,
        training_params=train_params,
        train_loader=frame_train_loader,
        valid_loader=frame_val_loader,
    )
    run.finish()
