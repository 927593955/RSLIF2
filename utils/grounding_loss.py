"""
utils/grounding_loss.py

Loss function for Visual Grounding (single-box REC task).

Replaces RSVLMLoss (YOLO detection loss) with a task-appropriate loss:
    L = λ_giou * L_GIoU + λ_l1 * L_L1 + λ_spa * L_spa + λ_orth * L_orth

- L_GIoU  : Generalized IoU on predicted vs GT box  (primary localization signal)
- L_L1    : L1 on normalized (cx,cy,w,h)             (fast coordinate convergence)
- L_spa   : Text-spatial-branch box prediction MSE    (auxiliary constraint)
- L_orth  : Orthogonality of semantic vs attribute vectors (representation quality)

No TaskAlignedAssigner, no DFL, no class prediction → much simpler.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Box utilities
# ──────────────────────────────────────────────────────────────────────────────

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2). Inputs in [0,1]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU between two sets of xyxy boxes.
    Both inputs: [N, 4]
    Returns: [N, N] matrix of GIoU values.
    """
    assert boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # [N, M, 2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]                      # [N, M]

    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter                # [N, M]
    iou = inter / (union + 1e-6)

    # Enclosing box
    enc_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enc_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[..., 0] * enc_wh[..., 1]

    giou = iou - (enc_area - union) / (enc_area + 1e-6)
    return giou


def _get_per_sample_gt(batch_targets: torch.Tensor, batch_size: int, device) -> torch.Tensor:
    """
    Extract one GT box (cx,cy,w,h) per sample from YOLO-style targets.
    batch_targets: [N, 6] with columns [img_idx, cls, cx, cy, w, h]
    Returns: [B, 4] normalized (cx,cy,w,h); zero-row if no GT for that sample.
    """
    gt = torch.zeros(batch_size, 4, device=device)
    for i in range(batch_size):
        mask = batch_targets[:, 0].long() == i
        if mask.any():
            gt[i] = batch_targets[mask][0, 2:6]  # take first GT box
    return gt


# ──────────────────────────────────────────────────────────────────────────────
# Main loss
# ──────────────────────────────────────────────────────────────────────────────

class GroundingLoss(nn.Module):
    def __init__(
        self,
        lambda_giou: float = 2.0,
        lambda_l1:   float = 5.0,
        lambda_spa:  float = 1.0,
        lambda_orth: float = 0.3,
    ):
        super().__init__()
        self.lambda_giou = lambda_giou
        self.lambda_l1   = lambda_l1
        self.lambda_spa  = lambda_spa
        self.lambda_orth = lambda_orth

    def forward(self, preds: dict, batch_targets: torch.Tensor, lambdas: dict = None):
        """
        preds:         output dict from RemoteSensingVLM
        batch_targets: [N, 6] YOLO-format [img_idx, cls, cx, cy, w, h] normalized
        lambdas:       optional dict to override default weights at runtime

        Returns: (total_loss, loss_dict)
        """
        device = batch_targets.device
        B = preds['pred_box'].shape[0]

        # ── Override lambdas if provided ──────────────────────────────────────
        lw = {
            'giou': self.lambda_giou,
            'l1':   self.lambda_l1,
            'spa':  self.lambda_spa,
            'orth': self.lambda_orth,
        }
        if lambdas:
            lw.update({k: float(v) for k, v in lambdas.items()})

        # ── Ground-truth boxes per sample ─────────────────────────────────────
        gt_boxes = _get_per_sample_gt(batch_targets, B, device)  # [B, 4] cx,cy,w,h
        valid = (gt_boxes.sum(-1) > 0)                           # [B] mask

        if not valid.any():
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {k: 0.0 for k in ('giou', 'l1', 'spa', 'orth', 'total')}

        # ── Predicted box ─────────────────────────────────────────────────────
        pred = preds['pred_box']                   # [B, 4] cx,cy,w,h in [0,1]
        pred_v = pred[valid]
        gt_v   = gt_boxes[valid]

        # ── L_GIoU ───────────────────────────────────────────────────────────
        pred_xyxy = box_cxcywh_to_xyxy(pred_v)    # [n, 4]
        gt_xyxy   = box_cxcywh_to_xyxy(gt_v)      # [n, 4]
        giou_mat  = generalized_box_iou(pred_xyxy, gt_xyxy)  # [n, n]
        loss_giou = (1 - giou_mat.diag()).mean()

        # ── L_L1 ─────────────────────────────────────────────────────────────
        loss_l1 = F.l1_loss(pred_v, gt_v)

        # ── L_spa (text spatial branch constraint) ────────────────────────────
        loss_spa = torch.tensor(0.0, device=device)
        if 'txt_pred_box' in preds and lw['spa'] > 0:
            txt_box = preds['txt_pred_box'][valid]  # [n, 4]
            loss_spa = F.smooth_l1_loss(txt_box, gt_v, beta=0.1)

        # ── L_orth (semantic ⊥ attribute representation) ─────────────────────
        loss_orth = torch.tensor(0.0, device=device)
        if 'sem_vec' in preds and 'attr_vec' in preds and lw['orth'] > 0:
            s = F.normalize(preds['sem_vec'], p=2, dim=1, eps=1e-8)
            a = F.normalize(preds['attr_vec'], p=2, dim=1, eps=1e-8)
            loss_orth = (s * a).sum(-1).pow(2).mean()
            if torch.isnan(loss_orth):
                loss_orth = torch.tensor(0.0, device=device)

        # ── Total ─────────────────────────────────────────────────────────────
        total = (
            lw['giou'] * loss_giou
            + lw['l1']   * loss_l1
            + lw['spa']  * loss_spa
            + lw['orth'] * loss_orth
        )

        if torch.isnan(total) or torch.isinf(total):
            total = torch.tensor(0.0, device=device)

        loss_dict = {
            'giou':  (lw['giou'] * loss_giou).item(),
            'l1':    (lw['l1']   * loss_l1).item(),
            'spa':   (lw['spa']  * loss_spa).item(),
            'orth':  (lw['orth'] * loss_orth).item(),
            'total': total.item(),
        }

        return total, loss_dict


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    loss_fn = GroundingLoss()
    B = 4
    preds = {
        'pred_box':     torch.rand(B, 4).clamp(0.05, 0.95),
        'txt_pred_box': torch.rand(B, 4).clamp(0.05, 0.95),
        'sem_vec':      torch.randn(B, 1024),
        'attr_vec':     torch.randn(B, 1024),
    }
    # YOLO-format targets: [img_idx, cls, cx, cy, w, h]
    targets = torch.tensor([
        [0, 1, 0.5, 0.5, 0.3, 0.3],
        [1, 2, 0.4, 0.6, 0.2, 0.4],
        [2, 0, 0.7, 0.3, 0.4, 0.2],
        [3, 1, 0.3, 0.4, 0.3, 0.3],
    ])
    loss, items = loss_fn(preds, targets)
    print('Loss:', loss.item())
    print('Items:', items)