import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from detectionv2 import frame_predict
from sklearn.metrics import classification_report
import wandb

def evaluate_base_with_frame_predict(test_loader, iou_thresh, conf_thresh, wandb_run):
    all_labels, all_preds = [], []

    with torch.no_grad():
        # batch here is one video per item (batch_size=1)
        for batch in test_loader:
            # batch["images"] is a list of T×C×H×W tensors (one per video in the batch)
            # batch["targets"] is a list of length-T lists of dicts
            for frames, targets_list in zip(batch["images"], batch["targets"]):
                # frames: Tensor[T, C, H, W], targets_list: List[dict] of length T
                T = frames.shape[0]
                for t in range(T):
                    frame = frames[t]
                    tgt   = targets_list[t]

                    # Convert to H×W×3 uint8
                    img_np = frame.permute(1,2,0).cpu().numpy().astype(np.uint8)

                    # Run the ONNX‐exported YOLO‐NAS
                    dets_np = frame_predict(
                        input_frame=img_np,
                        conf_threshold=conf_thresh,
                        iou_threshold=iou_thresh
                    )  # [N,6] → x1,y1,x2,y2,label,score

                    # Pull out GT boxes/labels for this frame
                    gt_boxes  = tgt["boxes"]
                    gt_labels = tgt["labels"]

                    # No detections -> predict “stand” for every GT
                    if dets_np.size == 0:
                        all_labels.extend(gt_labels.tolist())
                        all_preds.extend([1]*len(gt_labels))
                        continue

                    # Match preds to GT by IoU
                    pred_boxes = torch.from_numpy(dets_np[:,:4])
                    ious = box_iou(pred_boxes, gt_boxes)
                    best_pred = ious.argmax(dim=0)
                    max_ious  = ious.max(dim=0).values
                    mask      = max_ious >= iou_thresh

                    # For matched GT, use the predicted label
                    if mask.any():
                        matched_idxs   = best_pred[mask].cpu().numpy()
                        matched_labels = dets_np[matched_idxs, 4].astype(int).tolist()
                        all_labels.extend(gt_labels[mask].tolist())
                        all_preds.extend(matched_labels)

                    # Unmatched GT -> default “stand”
                    n_un = (~mask).sum().item()
                    if n_un:
                        all_labels.extend(gt_labels[~mask].tolist())
                        all_preds.extend([1]*n_un)

    # Log confusion matrix
    cm = wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=["fall","stand"]
    )

    # Classification report
    rpt = classification_report(
        all_labels, all_preds,
        labels=[0,1], target_names=["fall","stand"],
        output_dict=True, zero_division=0
    )

    wandb_run.log({
        "base_test/precision_fall": rpt["fall"]["precision"],
        "base_test/recall_fall":    rpt["fall"]["recall"],
        "base_test/f1_fall":        rpt["fall"]["f1-score"],
        "base_test/confusion_matrix": cm
    }, step=0, commit=True)

    wandb_run.summary["base_test/precision_fall"] = rpt["fall"]["precision"]
    wandb_run.summary["base_test/recall_fall"]    = rpt["fall"]["recall"]
    wandb_run.summary["base_test/f1_fall"]        = rpt["fall"]["f1-score"]

run = wandb.init(project="fall-detection-final-training")

from data_splitter import test_loader
evaluate_base_with_frame_predict(
    test_loader=test_loader,
    iou_thresh=0.3698499318033875,
    conf_thresh=0.3276128537002468,
    wandb_run=run,
)