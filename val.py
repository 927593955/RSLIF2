"""
val.py  (rewritten for grounding)

Validates the grounding model.

Key fixes vs old val.py:
1. Removed class-match requirement from Acc@0.5  ← was artificially halving accuracy
2. Uses model's direct pred_box output instead of NMS on detection proposals
3. IoU computed directly between pred_box and GT box (no top-1 confidence ranking needed)
4. Supports optional @0.25 / @0.5 / @0.75 Acc thresholds
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou_diag(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """
    Per-sample IoU between matched pred/gt boxes.
    pred_xyxy, gt_xyxy: [B, 4]
    Returns: [B] IoU values
    """
    lt = torch.max(pred_xyxy[:, :2], gt_xyxy[:, :2])
    rb = torch.min(pred_xyxy[:, 2:], gt_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    a1 = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    a2 = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
    union = (a1 + a2 - inter).clamp(min=1e-6)
    return inter / union


@torch.no_grad()
def validate(model, val_loader, device,
             iou_thresholds=(0.25, 0.5, 0.75), verbose=True):
    """
    Args:
        model:          RemoteSensingVLM (grounding variant)
        val_loader:     DataLoader yielding (imgs, input_ids, masks, targets, spa_gt)
        device:         torch.device
        iou_thresholds: tuple of IoU thresholds for Acc@k
        verbose:        print progress bar

    Returns:
        metrics: dict  { 'acc@0.25': float, 'acc@0.5': float, 'acc@0.75': float,
                         'mean_iou': float }
    """
    model.eval()

    total = 0
    correct = {t: 0 for t in iou_thresholds}
    sum_iou = 0.0

    pbar = tqdm(val_loader, desc='Validating', leave=False) if verbose else val_loader

    for batch in pbar:
        imgs, input_ids, masks, targets, _ = [x.to(device) for x in batch]
        B = imgs.shape[0]

        output = model(imgs, [input_ids, masks])
        pred_box = output['pred_box']  # [B, 4] normalized cx,cy,w,h

        # Extract GT box per sample (first GT if multiple)
        h, w = imgs.shape[2:]
        for i in range(B):
            t = targets[targets[:, 0].long() == i]
            if len(t) == 0:
                continue

            total += 1

            # GT: convert normalized cxcywh → pixel xyxy
            gt_norm = t[0, 2:6]                           # [4] cx,cy,w,h normalized
            gt_xyxy_norm = box_cxcywh_to_xyxy(gt_norm.unsqueeze(0))   # [1,4]

            # Pred: also normalized
            pb = pred_box[i].unsqueeze(0).clamp(0, 1)     # [1, 4]
            pb_xyxy = box_cxcywh_to_xyxy(pb)              # [1, 4]

            iou = box_iou_diag(pb_xyxy, gt_xyxy_norm).item()
            sum_iou += iou

            for t_thresh in iou_thresholds:
                if iou >= t_thresh:
                    correct[t_thresh] += 1

        if verbose and isinstance(pbar, tqdm):
            acc5 = correct[0.5] / max(total, 1)
            pbar.set_postfix(Acc5=f'{acc5:.4f}')

    metrics = {f'acc@{t}': correct[t] / max(total, 1) for t in iou_thresholds}
    metrics['mean_iou'] = sum_iou / max(total, 1)
    metrics['total'] = total

    if verbose:
        print(
            f'📊 Val Results | '
            + ' | '.join(f"Acc@{t}={metrics[f'acc@{t}']:.4f}" for t in iou_thresholds)
            + f' | mIoU={metrics["mean_iou"]:.4f}'
            + f' | n={total}'
        )

    return metrics