import os
import multiprocessing
import torch
import wandb

from super_gradients.common.sg_loggers.wandb_sg_logger import WandBSGLogger, WANDB_ID_PREFIX

def _set_wandb_id_no_mknod(self, wandb_id):
    path = os.path.join(self._local_dir, f"{WANDB_ID_PREFIX}{wandb_id}")
    with open(path, "w"):
        pass

WandBSGLogger._set_wandb_id = _set_wandb_id_no_mknod
multiprocessing.set_start_method("fork", force=True)

from torch.utils.data import DataLoader, IterableDataset
from super_gradients.training import Trainer, models
from super_gradients.common.object_names import Models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

from data_splitter import train_loader, val_loader 

class FrameLevelDataset(IterableDataset):
    def __init__(self, video_loader):
        self.video_loader = video_loader
        self.video_ds     = video_loader.dataset

    def __iter__(self):
        for batch in self.video_loader:
            imgs, targs = batch["images"][0], batch["targets"][0]
            for img, tgt in zip(imgs, targs):
                yield img, tgt

    def __len__(self):
        return sum(len(frames) for frames in self.video_ds.video_dict.values())

def detection_collate_fn(batch):
    imgs, tgts = zip(*batch)
    imgs = torch.stack(imgs, 0)
    all_t = []
    for i, tgt in enumerate(tgts):
        boxes, labels = tgt["boxes"], tgt["labels"]
        x1,y1,x2,y2 = boxes.unbind(1)
        w,h = x2-x1, y2-y1
        cx,cy = x1 + w/2, y1 + h/2
        idxs = torch.full((boxes.size(0),1), i, device=boxes.device, dtype=boxes.dtype)
        labs = labels.unsqueeze(1).to(boxes.dtype)
        all_t.append(torch.cat([idxs, labs, cx.unsqueeze(1), cy.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1)], dim=1))
    if all_t:
        return imgs, torch.cat(all_t, 0)
    return imgs, torch.zeros((0,6), dtype=torch.float32, device=imgs.device)

def train():
    # Initialize a W&B run for this trial
    wandb.init()
    cfg = wandb.config

    # Build frame-level loaders
    frame_train = DataLoader(
        FrameLevelDataset(train_loader),
        batch_size=cfg.batch_size,
        num_workers=0,
        collate_fn=detection_collate_fn,
        drop_last=False,
    )
    frame_val = DataLoader(
        FrameLevelDataset(val_loader),
        batch_size=cfg.batch_size,
        num_workers=0,
        collate_fn=detection_collate_fn,
        drop_last=False,
    )

    # Model + hyperparams
    model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco", num_classes=2)
    params = {
        "initial_lr": cfg.initial_lr,
        "lr_warmup_epochs": cfg.lr_warmup_epochs,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": cfg.cosine_final_lr_ratio,
        "optimizer": cfg.optimizer,
        "optimizer_params": {"weight_decay": cfg.weight_decay},
        "max_epochs": 10,
        "mixed_precision": True,
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
        "metric_to_watch": "mAP@0.50",
        "sg_logger": "wandb_sg_logger",
        "sg_logger_params": {
            "project_name": "yolo-base-trainer",
            "entity": "robbeghysen-ku-leuven",
            "save_logs_remote": True
        }
    }

    trainer = Trainer(
        experiment_name="sweep_trial",
        ckpt_root_dir="./checkpoints",
    )
    trainer.train(
        model=model,
        training_params=params,
        train_loader=frame_train,
        valid_loader=frame_val,
    )

if __name__ == "__main__":
    # Define sweep config
    sweep_config = {
      "name": "yolo_nas_s_sweep",
      "method": "bayes",
      "metric": {"name": "Valid_mAP@0.50", "goal": "maximize"},
      "early_terminate": {"type": "hyperband", "max_iter": 10, "s": 2},
      "parameters": {
        "initial_lr": {
          "distribution": "log_uniform_values",
          "min": 1e-5,
          "max": 5e-4
        },
        "weight_decay": {"values": [0.0, 1e-4, 5e-4, 1e-3]},
        "batch_size":   {"values": [8, 16, 32]},
        "lr_warmup_epochs": {"values": [1, 3, 5]},
        "cosine_final_lr_ratio": {
          "distribution": "uniform",
          "min": 0.01,
          "max": 0.2
        },
        "optimizer": {"values": ["Adam", "SGD"]}
      }
    }
    # Launch the sweep and start an agent
    sweep_id = wandb.sweep(sweep_config, project="yolo-base-trainer", entity="robbeghysen-ku-leuven")
    wandb.agent(sweep_id, function=train)
