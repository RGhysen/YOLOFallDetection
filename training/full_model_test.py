import json
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from sklearn.metrics import classification_report
import wandb
from fusion_modelv2 import SpeedClassificationModule
from integrator_onnx import YoloNASSpeedFusion, ByteTrackArgument

def evaluate_full_model(test_loader, model, fusion_module, iou_thresh, fall_thresh, device, wandb_run):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        # batch_size=1: one video per batch
        for batch in test_loader:
            for frames, targets_list in zip(batch["images"], batch["targets"]):
                frames = frames.to(device, dtype=torch.float32)
                fusion_module.reset()

                # run the full pipeline: get bboxes_list & logits_list per frame
                bboxes_list, logits_list = model(frames, targets=None)

                # now match per-frame predictions to GT
                for pred_boxes, logits, tgt in zip(bboxes_list, logits_list, targets_list):
                    gt_boxes  = tgt["boxes"].to(device)
                    gt_labels = tgt["labels"].to(device)

                    # no detections -> predict “stand” for all GT
                    if logits.numel() == 0:
                        all_labels.extend(gt_labels.tolist())
                        all_preds.extend([1] * len(gt_labels))
                        continue

                    # compute fall‐vs‐stand probs and threshold
                    probs       = F.softmax(logits, dim=1)
                    fall_scores = probs[:, 0]
                    # pred = 0 if score>fall_thresh else 1
                    pred_labels = torch.where(
                        fall_scores > fall_thresh,
                        torch.zeros_like(fall_scores, dtype=torch.long),
                        torch.ones_like(fall_scores,  dtype=torch.long)
                    )

                    # match pred_boxes -> gt_boxes by IoU
                    ious      = box_iou(pred_boxes, gt_boxes)
                    best_pred = ious.argmax(dim=0)
                    max_ious  = ious.max(dim=0).values
                    mask      = max_ious >= iou_thresh

                    # matched GT: collect both GT and predicted labels
                    if mask.any():
                        matched_idxs   = best_pred[mask]
                        matched_preds  = pred_labels[matched_idxs].tolist()
                        matched_gts    = gt_labels[mask].tolist()
                        all_preds .extend(matched_preds)
                        all_labels.extend(matched_gts)

                    # unmatched GT → default “stand”
                    n_un = (~mask).sum().item()
                    if n_un:
                        all_labels.extend(gt_labels[~mask].tolist())
                        all_preds.extend([1] * n_un)

    # log confusion matrix
    cm = wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=["fall","stand"]
    )

    # classification report (we care about the “fall” class)
    rpt = classification_report(
        all_labels,
        all_preds,
        labels=[0,1],
        target_names=["fall","stand"],
        output_dict=True,
        zero_division=0
    )

    wandb_run.log({
        "full_test/precision_fall": rpt["fall"]["precision"],
        "full_test/recall_fall":    rpt["fall"]["recall"],
        "full_test/f1_fall":        rpt["fall"]["f1-score"],
        "full_test/confusion_matrix": cm
    }, commit=True)

    wandb_run.summary["full_test/precision_fall"] = rpt["fall"]["precision"]
    wandb_run.summary["full_test/recall_fall"]    = rpt["fall"]["recall"]
    wandb_run.summary["full_test/f1_fall"]        = rpt["fall"]["f1-score"]


if __name__ == "__main__":
    with open("consensus_config.json") as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusion_module = SpeedClassificationModule(
        feat_dim=5,
        hidden_dim=cfg["hidden_dim"],
        num_hidden_layers=cfg["num_hidden_layers"],
        dropout_p=cfg["dropout_p"],
        pruning_lag=cfg["pruning_lag"]
    ).to(device)

    tracker_args = ByteTrackArgument()
    tracker_args.track_thresh   = cfg["conf_threshold"]
    tracker_args.match_thresh   = cfg["match_thresh"]

    model = YoloNASSpeedFusion(
        fusion_module,
        tracker_args,
        conf_threshold=cfg["conf_threshold"],
        speed_average_frames=cfg["speed_frames"],
        visualize=False
    ).to(device)

    ckpt = torch.load("7fps_checkpoint_epoch_10.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    fusion_module.load_state_dict(ckpt["fusion_state_dict"])

    run = wandb.init(project="fall-detection-final-training", config=cfg)
    from data_splitter import test_loader

    evaluate_full_model(
        test_loader=test_loader,
        model=model,
        fusion_module=fusion_module,
        iou_thresh=cfg["loss_iou_thresh"],
        fall_thresh=0.5,   
        device=device,
        wandb_run=run
    )

    run.finish()
