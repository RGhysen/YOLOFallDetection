import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

def classification_loss(bboxes_list, logits_list, targets, iou_thresh=0.5, weight: torch.Tensor = None, device=None):
    """
    Now returns (loss, total_matches).
    """
    losses = []
    match_counts = []

    if weight is not None:
        # assume at least one logits exists
        device = logits_list[0].device  
        weight = weight.to(device)

    for pred_boxes, logits, tgt in zip(bboxes_list, logits_list, targets):
        gt_boxes  = tgt['boxes'].to(logits.device)
        gt_labels = tgt['labels'].to(logits.device)

        if logits.numel() == 0 or gt_labels.numel() == 0:
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        best_pred_idx = ious.argmax(dim=0)
        max_ious = ious.max(dim=0).values
        mask = max_ious >= iou_thresh

        M = mask.sum().item()
        if M == 0:
            continue

        matched_logits = logits[best_pred_idx[mask]]
        matched_labels = gt_labels[mask]

        w = weight.to(device) if weight is not None else None

        losses.append(F.cross_entropy(matched_logits, matched_labels, weight=w, reduction='mean'))
        match_counts.append(M)

    if not losses:
        # no matches anywhere -> return zero but keep grad on tensor
        return torch.tensor(0., requires_grad=True, device=logits.device), 0

    loss = torch.stack(losses).mean()
    total_matches = sum(match_counts)
    return loss, total_matches