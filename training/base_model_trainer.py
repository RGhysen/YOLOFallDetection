import os
import json
import multiprocessing
import torch

# Patch W&B logger to not use `mknod` (not supported on macOS M3)
from super_gradients.common.sg_loggers.wandb_sg_logger import WandBSGLogger, WANDB_ID_PREFIX
def _set_wandb_id_no_mknod(self, wandb_id):
    path = os.path.join(self._local_dir, f"{WANDB_ID_PREFIX}{wandb_id}")
    with open(path, "w"):
        pass
WandBSGLogger._set_wandb_id = _set_wandb_id_no_mknod
multiprocessing.set_start_method("fork", force=True)

from super_gradients.common.sg_loggers.wandb_sg_logger import WandBSGLogger
import wandb

_original_add_config = WandBSGLogger.add_config
def _add_config_allow_change(self, key, config_dict):
    wandb.config.update({key: config_dict}, allow_val_change=True)
    # _original_add_config(self, key, config_dict)

WandBSGLogger.add_config = _add_config_allow_change

import wandb
from torch.utils.data import DataLoader, IterableDataset
from super_gradients.training import Trainer, models
from super_gradients.common.object_names import Models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

from data_splitter import train_loader, val_loader  

# Load best sweep config
with open("best_config_base.json", "r") as f:
    best = json.load(f)

BATCH_SIZE            = best["batch_size"]
INITIAL_LR            = best["initial_lr"]
WEIGHT_DECAY          = best["weight_decay"]
OPTIMIZER_NAME        = best["optimizer"].lower()

LR_WARMUP_EPOCHS      = best.get("lr_warmup_epochs", 0)
COSINE_FINAL_LR_RATIO = best.get("cosine_final_lr_ratio", 0.1)

th = best["training_hyperparams"]
MAX_EPOCHS            = th["max_epochs"]
WARMUP_MODE           = th["warmup_mode"]
WARMUP_INITIAL_LR     = th.get("warmup_initial_lr", INITIAL_LR * 0.1)
LR_MODE               = th["lr_mode"]
EMA_ENABLED           = th["ema"]
EMA_PARAMS            = th.get("ema_params", {"decay": 0.9, "decay_type": "threshold"})
AVERAGE_BEST_MODELS   = th.get("average_best_models", False)
SILENT_MODE           = th.get("silent_mode", False)

SG_LOGGER             = th.get("sg_logger", "wandb_sg_logger")
SG_LOGGER_PARAMS      = th.get("sg_logger_params", {})

class FrameLevelDataset(IterableDataset):
    def __init__(self, video_loader):
        self.video_loader = video_loader
        self.video_ds     = video_loader.dataset

    def __iter__(self):
        for batch in self.video_loader:
            vid_imgs, vid_targets = batch["images"][0], batch["targets"][0]
            # randomize per‐video frame order
            import random
            idxs = list(range(len(vid_imgs)))
            random.shuffle(idxs)
            for i in idxs:
                yield vid_imgs[i], vid_targets[i]

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
        all_t.append(torch.cat([idxs, labs, cx.unsqueeze(1), cy.unsqueeze(1),
                                w.unsqueeze(1), h.unsqueeze(1)], dim=1))
    if all_t:
        targets = torch.cat(all_t, 0)
    else:
        targets = torch.zeros((0,6), dtype=torch.float32, device=imgs.device)
    return imgs, targets

# Build frame‐level loaders
def make_frame_loader(video_loader):
    return DataLoader(
        FrameLevelDataset(video_loader),
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=detection_collate_fn,
        drop_last=False,
    )

frame_train_loader = make_frame_loader(train_loader)
frame_val_loader   = make_frame_loader(val_loader)


if __name__ == "__main__":
    # Init W&B
    run = wandb.init(
        project=SG_LOGGER_PARAMS.get("project_name", "yolo-base-trainer"),
        entity=SG_LOGGER_PARAMS.get("entity", None),
        config=best
    )

    # Build model
    net = models.get(Models.YOLO_NAS_S, pretrained_weights="coco", num_classes=2)

    # Training params dict entirely from best config
    train_params = {
        "silent_mode": SILENT_MODE,
        "average_best_models": AVERAGE_BEST_MODELS,
        "warmup_mode": WARMUP_MODE,
        "warmup_initial_lr": WARMUP_INITIAL_LR,
        "lr_warmup_epochs": LR_WARMUP_EPOCHS,
        "initial_lr": INITIAL_LR,
        "lr_mode": LR_MODE,
        "cosine_final_lr_ratio": COSINE_FINAL_LR_RATIO,
        "optimizer": OPTIMIZER_NAME,
        "optimizer_params": {"weight_decay": WEIGHT_DECAY},
        "zero_weight_decay_on_bias_and_bn": th.get("zero_weight_decay_on_bias_and_bn", True),
        "ema": EMA_ENABLED,
        "ema_params": EMA_PARAMS,
        "max_epochs": MAX_EPOCHS,
        "mixed_precision": th.get("mixed_precision", True),
        "loss": PPYoloELoss(use_static_assigner=False, num_classes=2, reg_max=16),
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
        "metric_to_watch": th.get("metric_to_watch", "F1@0.50"),
        "sg_logger": SG_LOGGER,
        "sg_logger_params": SG_LOGGER_PARAMS,
    }

    trainer = Trainer(
        experiment_name='yolo-base-final',
        ckpt_root_dir="base_model_head"
    )
    trainer.train(
        model=net,
        training_params=train_params,
        train_loader=frame_train_loader,
        valid_loader=frame_val_loader
    )

    run.finish()
