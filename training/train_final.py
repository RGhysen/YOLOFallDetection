import os
import json
import random
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
import torch.nn.functional as F

from data_splitter import train_ds, val_ds, test_ds, collate_fn
from fusion_modelv2 import SpeedClassificationModule
from integrator_onnx import YoloNASSpeedFusion, ByteTrackArgument
from loss import classification_loss


def main():
    with open("consensus_config.json") as f:
        cfg = json.load(f)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Running on", device)

    seed = cfg.get("seed", 35)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    class_weights = torch.tensor([cfg.get("fall_weight", 1.0), 1.0], device=device)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=4, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                               num_workers=4, collate_fn=collate_fn)

    tracker_args = ByteTrackArgument()
    tracker_args.track_thresh = cfg["conf_threshold"]
    tracker_args.match_thresh = cfg["match_thresh"]

    fusion_module = SpeedClassificationModule(
        feat_dim=5,
        hidden_dim=cfg["hidden_dim"],
        num_hidden_layers=cfg["num_hidden_layers"],
        dropout_p=cfg["dropout_p"],
        pruning_lag=cfg["pruning_lag"]
    ).to(device)

    model = YoloNASSpeedFusion(
        fusion_module,
        tracker_args,
        conf_threshold=cfg["conf_threshold"],
        speed_average_frames=cfg["speed_frames"],
        visualize=False
    ).to(device)

    optimizer = torch.optim.AdamW(
        fusion_module.parameters(),
        lr=cfg["lr_head"],
        weight_decay=cfg["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
    )

    run = wandb.init(project="fall-detection-final-training", config=cfg)
    wandb.watch(model, log="all", log_freq=83)

    num_epochs = cfg.get("epochs")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses, train_matches = [], 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_matches = 0
            for frames, targets in zip(batch["images"], batch["targets"]):
                frames = frames.to(device, dtype=torch.float32)
                fusion_module.reset()
                bboxes_list, logits_list = model(frames, targets)
                loss_tensor, matches = classification_loss(
                    bboxes_list, logits_list, targets,
                    iou_thresh=cfg["loss_iou_thresh"],
                    weight=class_weights,
                    device=device
                )
                batch_loss = batch_loss + loss_tensor
                batch_matches += matches

            batch_loss = batch_loss / len(batch["images"])
            batch_loss.backward()
            optimizer.step()

            train_losses.append(batch_loss.item())
            train_matches += batch_matches

        avg_train_loss = sum(train_losses) / len(train_losses)
        wandb.log({
            "train/loss": avg_train_loss,
            "train/matches": train_matches,
            "lr": scheduler.get_last_lr()[0]
        }, step=epoch)

        scheduler.step()

        ckpt_filename = f"7fps_checkpoint_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "fusion_state_dict": fusion_module.state_dict(),
            "tracker_args": vars(tracker_args),
            "cfg": cfg
        }, ckpt_filename)
        artifact = wandb.Artifact(
            name=f"7fps-fall-detector-epoch-{epoch}",
            type="model",
            description=f"Checkpoint after epoch {epoch}"
        )
        artifact.add_file(ckpt_filename)
        run.log_artifact(artifact, aliases=["latest"])

    

    # Test evaluation
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            for frames, targets in zip(batch["images"], batch["targets"]):
                frames = frames.to(device, dtype=torch.float32)
                fusion_module.reset()
                bboxes_list, logits_list = model(frames, targets)
                for pred_boxes, logits, tgt in zip(bboxes_list, logits_list, targets):
                    gt_boxes = tgt["boxes"].to(device)
                    gt_labels = tgt["labels"].to(device)
                    if logits.numel() == 0:
                        all_labels.extend(gt_labels.tolist())
                        all_preds.extend([1] * len(gt_labels))
                    else:
                        ious = box_iou(pred_boxes, gt_boxes)
                        best_pred = ious.argmax(dim=0)
                        mask = ious.max(dim=0).values >= cfg["loss_iou_thresh"]
                        if mask.any():
                            probs = F.softmax(logits, dim=1)
                            all_labels.extend(gt_labels[mask].tolist())
                            all_preds.extend(probs[best_pred[mask]].argmax(1).tolist())
                        n_un = (~mask).sum().item()
                        if n_un:
                            all_labels.extend(gt_labels[~mask].tolist())
                            all_preds.extend([1] * n_un)

    rpt_cm = wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=["fall", "stand"],
    )

    from sklearn.metrics import classification_report
    rpt_dict = classification_report(
        all_labels, all_preds,
        labels=[0,1],
        target_names=["fall","stand"],
        output_dict=True, zero_division=0
    )

    final_step = cfg.get("epochs", 0)
    wandb.log({
        "test/precision_fall": rpt_dict["fall"]["precision"],
        "test/recall_fall":    rpt_dict["fall"]["recall"],
        "test/f1_fall":        rpt_dict["fall"]["f1-score"],
        "test/confusion_matrix": rpt_cm
    }, step=final_step, commit=True)

    wandb.run.summary["test/precision_fall"] = rpt_dict["fall"]["precision"]
    wandb.run.summary["test/recall_fall"]    = rpt_dict["fall"]["recall"]
    wandb.run.summary["test/f1_fall"]        = rpt_dict["fall"]["f1-score"]

    wandb.finish()


if __name__ == "__main__":
    main()
