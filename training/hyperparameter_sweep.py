import wandb
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torchvision.ops import box_iou
from torch import nn
import torch.nn.functional as F
from dataloader import collate_fn
from loss import classification_loss
from integrator_onnx import YoloNASSpeedFusion, ByteTrackArgument
from fusion_modelv2 import SpeedClassificationModule
from data_splitter import data_folder, train_vids, val_vids
from data_splitter import train_ds, val_ds

def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🏃‍♀️ Running on Apple MPS")
    else:
        device = torch.device("cpu")
        print("⚠️  MPS not available, falling back to CPU")
        
    run = wandb.init(
        project="fall-detection-speed-fusion",
        config={
            # Fusion head hyperparams
            "lr_head":           1e-3,
            "hidden_dim":        32,
            "num_hidden_layers": 1,
            "dropout_p":         0.0,
            "pruning_lag":       5,

            # Feature engineering
            "speed_frames":      5,

            # Detection & tracking
            "conf_threshold":    0.8,
            "match_thresh":      0.8,

            # Loss‐matching IoU
            "loss_iou_thresh":   0.5,
            "fall_weight":    1.0,

            # Optimization
            "weight_decay":      1e-4,
            "batch_size":        1,

            # Training schedule
            "epochs":            3,
        }
    )
    cfg = run.config

    tracker_args = ByteTrackArgument()
    tracker_args.track_thresh = cfg.conf_threshold
    tracker_args.match_thresh = cfg.match_thresh

    class_weights = torch.tensor([cfg.fall_weight, 1.0])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    fusion_module = SpeedClassificationModule(
        feat_dim=5,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        dropout_p=cfg.dropout_p,
        pruning_lag=cfg.pruning_lag
    ).to(device)
    model_joint = YoloNASSpeedFusion(
        fusion_module,
        tracker_args,
        conf_threshold=cfg.conf_threshold,
        speed_average_frames=cfg.speed_frames,
        visualize=False
    ).to(device).train()

    optimizer = torch.optim.AdamW(
        fusion_module.parameters(),
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs
    )
    
    wandb.watch(model_joint, log="all", log_freq=50)

    # Training & validation loop
    for epoch in range(1, cfg.epochs + 1):
        # TRAIN
        model_joint.train()
        train_losses = []
        train_matches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            batch_loss    = 0.0
            batch_matches = 0

            for frames, targets in zip(batch["images"], batch["targets"]):
                frames = frames.to(device, dtype=torch.float32)
                # one fresh video -> clear hidden states
                fusion_module.reset()

                # forward
                bboxes_list, logits_list = model_joint(frames, targets)

                # unpack (loss_tensor, match_count)
                loss_tensor, match_count = classification_loss(
                    bboxes_list,
                    logits_list,
                    targets,
                    iou_thresh=cfg.loss_iou_thresh,
                    weight=class_weights,
                    device=device
                )
                batch_loss    += loss_tensor
                batch_matches += match_count

            # average over videos in this batch
            batch_loss = batch_loss / len(batch["images"])
            batch_loss.backward()
            optimizer.step()

            train_losses.append(batch_loss.item())
            train_matches += batch_matches

        avg_train_loss = sum(train_losses) / len(train_losses)

        # VALIDATE
        model_joint.eval()
        val_losses = []
        val_matches = 0
        all_labels = []
        all_preds  = []

        with torch.no_grad():
            for batch in val_loader:
                for frames, targets in zip(batch["images"], batch["targets"]):
                    frames = frames.to(device, dtype=torch.float32)
                    
                    fusion_module.reset()
                    for tgt in targets:
                        tgt["boxes"]  = tgt["boxes"].to(device)
                        tgt["labels"] = tgt["labels"].to(device)

                    bboxes_list, logits_list = model_joint(frames, targets)
                    loss_tensor, match_count = classification_loss(
                        bboxes_list,
                        logits_list,
                        targets,
                        iou_thresh=cfg.loss_iou_thresh,
                        weight=class_weights,
                        device=device
                    )
                    val_losses.append(loss_tensor.item())
                    val_matches += match_count

                    # collect preds & GT labels for F1
                    for pb, lg, tgt in zip(bboxes_list, logits_list, targets):
                        gt_boxes, gt_labels = tgt["boxes"], tgt["labels"]
                        if lg.numel() == 0:
                            all_labels.extend(gt_labels.tolist())
                            all_preds.extend([1] * len(gt_labels))
                            continue

                        ious      = box_iou(pb, gt_boxes)
                        best_pred = ious.argmax(dim=0)
                        max_iou   = ious.max(dim=0).values
                        mask      = max_iou >= cfg.loss_iou_thresh

                        if mask.any():
                            probs = F.softmax(lg, dim=1)
                            all_labels.extend(gt_labels[mask].tolist())
                            all_preds.extend(probs[best_pred[mask]].argmax(1).tolist())

                        # unmatched GTs -> stand (1)
                        num_unmatched = (~mask).sum().item()
                        if num_unmatched:
                            all_labels.extend(gt_labels[~mask].tolist())
                            all_preds.extend([1] * num_unmatched)

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")

        # compute fall‐class F1
        rpt = classification_report(
            all_labels, all_preds,
            labels=[0, 1], target_names=["fall", "stand"],
            output_dict=True, zero_division=0
        )
        fall_f1   = rpt["fall"]["f1-score"]
        fall_prec = rpt["fall"]["precision"]
        fall_rec  = rpt["fall"]["recall"]

        # confusion matrix plot
        cm_plot = wandb.plot.confusion_matrix(
            y_true=all_labels,
            preds=all_preds,
            class_names=["fall", "stand"]
        )

        wandb.log({
            "epoch":           epoch,
            "train/lr":        optimizer.param_groups[0]['lr'],
            "train/loss":      avg_train_loss,
            "train/matches":   train_matches,
            "val/loss":        avg_val_loss,
            "val/matches":     val_matches,
            "fall/precision":  fall_prec,
            "fall/recall":     fall_rec,
            "fall/f1":         fall_f1,
            "confusion_matrix": cm_plot,
        })
        scheduler.step()

    run.finish()


# Sweep config
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "fall/f1",
        "goal": "maximize"
    },
    "parameters": {
        "lr_head": {
            "distribution": "log_uniform_values",
            "min": 1e-5, "max": 1e-2
        },
        "hidden_dim": {
            "values": [16, 32, 64, 128]
        },
        "num_hidden_layers": {
            "values": [1, 2, 3]
        },
        "dropout_p": {
            "distribution": "uniform",
            "min": 0.0, "max": 0.5
        },
        "pruning_lag": {
            "distribution": "int_uniform",
            "min": 0, "max": 50
        },
        "speed_frames": {
            "distribution": "int_uniform",
            "min": 1, "max": 50
        },
        "conf_threshold": {
            "distribution": "uniform",
            "min": 0.3, "max": 0.9
        },
        "match_thresh": {
            "distribution": "uniform",
            "min": 0.3, "max": 0.9
        },
        "loss_iou_thresh": {
            "distribution": "uniform",
            "min": 0.3, "max": 0.7
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6, "max": 1e-2
        },
        "fall_weight": {
            "distribution": "log_uniform_values",
            "min": 0.2, "max": 5.0
        },
        "batch_size": {
            "values": [1, 2, 3, 4]
        },
        "epochs": {
            "values": [1, 2, 5, 10, 15]
        }
    }
}

# launch the sweep
sweep_id = wandb.sweep(sweep_config, project="fall-detection-speed-fusion")

wandb.agent(sweep_id, function=train, count=40)