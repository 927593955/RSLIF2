"""
utils/grounding_loss.py  (v2 - Proper Per-Branch Supervision)

Loss redesign summary:

PROBLEM FIXED: Supervision asymmetry
  OLD: semantic=nothing, spatial=smooth_l1, attribute=unused gate
  NEW: All three branches have direct, specific, correctly-motivated losses.

NEW LOSS TERMS:

  L_sem_cls  : CrossEntropy(sem_cls_logits, class_id)
    - class_id comes from targets[:, 1] which is ALREADY in the dataloader
    - Forces semantic branch to encode WHAT the object is
    - Natural supervision: "airplane" descriptions should cluster near class=0

  L_spa_quad : CrossEntropy(spa_quadrant_logits, quadrant_id)
    - quadrant_id = 3x3 grid cell containing the GT box center
    - Complements existing L_spa (box regression) with discrete location signal
    - Forces spatial branch to encode WHERE (coarse location = fast convergence)

  L_reconstruct : MSE(reconstructed, shared_pooled.detach())
    - reconstructed = MLP(concat[sem_vec, spa_vec, attr_vec])
    - shared_pooled = mean-pool of shared_encoder output (the full text meaning)
    - Forces the 3 branches to be COMPLEMENTARY: together they must recover all info
    - Stops attribute branch from collapsing to zero (if it did, reconstruction fails)
    - This is what makes attr_vec encode what semantic and spatial DON'T encode

  L_orth: (s·a)² + (s·p)²  [kept from v1, now alongside other losses]
    - Still useful as a push signal, but no longer the only attribute constraint

TOTAL LOSS:
  L = λ_giou*L_giou + λ_l1*L_l1
    + λ_spa*L_spa + λ_spa_quad*L_spa_quad
    + λ_sem_cls*L_sem_cls
    + λ_reconstruct*L_reconstruct
    + λ_orth*L_orth

TRAINING SCHEDULE (used with get_lambdas in train.py):
  Stage 1 (0..stage1_epochs):       giou + l1 only  -- learn to localize first
  Stage 2 (stage1..stage1+warmup2): + spa + spa_quad + sem_cls  -- branch signals
  Stage 3 (stage1+warmup2..end):    + reconstruct + orth  -- complementarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Box utilities (unchanged)
# =============================================================================

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    assert boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4
    lt   = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb   = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh   = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    iou   = inter / (union + 1e-6)
    enc_lt   = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enc_rb   = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enc_wh   = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[..., 0] * enc_wh[..., 1]
    giou = iou - (enc_area - union) / (enc_area + 1e-6)
    return giou


def _get_per_sample_gt(batch_targets: torch.Tensor, batch_size: int, device):
    """
    Extract per-sample GT from YOLO-format targets [N, 6]: [img_idx, cls, cx, cy, w, h].
    Returns:
        gt_boxes:    [B, 4]  normalized cx,cy,w,h  (zero if no GT)
        class_ids:   [B]     long class index        (0 if no GT)
        valid_mask:  [B]     bool, True if GT exists
    """
    gt_boxes  = torch.zeros(batch_size, 4, device=device)
    class_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
    valid     = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(batch_size):
        rows = batch_targets[batch_targets[:, 0].long() == i]
        if len(rows) > 0:
            gt_boxes[i]  = rows[0, 2:6]
            class_ids[i] = rows[0, 1].long()
            valid[i]     = True

    return gt_boxes, class_ids, valid


def _gt_to_quadrant(cx: torch.Tensor, cy: torch.Tensor, grid: int = 3) -> torch.Tensor:
    """
    Map GT box centers to grid-cell index.
    cx, cy: [B] normalized floats
    Returns: [B] long in [0, grid*grid-1]
    """
    col = (cx * grid).long().clamp(0, grid - 1)
    row = (cy * grid).long().clamp(0, grid - 1)
    return row * grid + col


# =============================================================================
# Main loss
# =============================================================================

class GroundingLoss(nn.Module):
    """
    Complete loss for RemoteSensingVLM grounding task.

    Default λ values are intentionally conservative for stability.
    Use get_lambdas() in train.py to apply staged scheduling.
    """

    def __init__(
        self,
        lambda_giou:        float = 2.0,
        lambda_l1:          float = 5.0,
        lambda_spa:         float = 1.0,
        lambda_spa_quad:    float = 0.5,  # NEW: spatial quadrant CE
        lambda_sem_cls:     float = 1.0,  # NEW: semantic class CE
        lambda_reconstruct: float = 0.5,  # NEW: complementarity reconstruction
        lambda_orth:        float = 0.3,
        spatial_grid:       int   = 3,
    ):
        super().__init__()
        self.lambda_giou        = lambda_giou
        self.lambda_l1          = lambda_l1
        self.lambda_spa         = lambda_spa
        self.lambda_spa_quad    = lambda_spa_quad
        self.lambda_sem_cls     = lambda_sem_cls
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_orth        = lambda_orth
        self.spatial_grid       = spatial_grid

    def forward(self, preds: dict, batch_targets: torch.Tensor, lambdas: dict = None):
        """
        preds:         output dict from RemoteSensingVLM (see vlm_grounding.py)
        batch_targets: [N, 6]  YOLO-format [img_idx, cls, cx, cy, w, h]
        lambdas:       optional runtime override of loss weights (from get_lambdas)

        Returns: (total_loss, loss_dict)
        """
        device = batch_targets.device
        B = preds['pred_box'].shape[0]

        # ── Merge runtime lambdas ─────────────────────────────────────────────
        lw = {
            'giou':        self.lambda_giou,
            'l1':          self.lambda_l1,
            'spa':         self.lambda_spa,
            'spa_quad':    self.lambda_spa_quad,
            'sem_cls':     self.lambda_sem_cls,
            'reconstruct': self.lambda_reconstruct,
            'orth':        self.lambda_orth,
        }
        if lambdas:
            lw.update({k: float(v) for k, v in lambdas.items() if k in lw})

        # ── Extract GT ────────────────────────────────────────────────────────
        gt_boxes, class_ids, valid = _get_per_sample_gt(batch_targets, B, device)

        zero = torch.tensor(0.0, device=device, requires_grad=True)
        if not valid.any():
            return zero, {k: 0.0 for k in ('giou','l1','spa','spa_quad',
                                             'sem_cls','reconstruct','orth','total')}

        pred    = preds['pred_box']          # [B, 4]
        pred_v  = pred[valid]
        gt_v    = gt_boxes[valid]

        # ── L_GIoU ───────────────────────────────────────────────────────────
        loss_giou = (1 - generalized_box_iou(
            box_cxcywh_to_xyxy(pred_v),
            box_cxcywh_to_xyxy(gt_v)
        ).diag()).mean()

        # ── L_L1 ─────────────────────────────────────────────────────────────
        loss_l1 = F.l1_loss(pred_v, gt_v)

        # ── L_spa (text spatial branch box regression) ────────────────────────
        loss_spa = torch.tensor(0.0, device=device)
        if 'txt_pred_box' in preds and lw['spa'] > 0:
            txt_box = preds['txt_pred_box'][valid]
            loss_spa = F.smooth_l1_loss(txt_box, gt_v, beta=0.1)

        # ── L_spa_quad (spatial quadrant classification) ──────────────────────
        # Supervision: which 3x3 grid cell contains the GT box center
        # Forces spatial branch to discretely learn WHERE objects appear
        loss_spa_quad = torch.tensor(0.0, device=device)
        if 'spa_quadrant_logits' in preds and lw['spa_quad'] > 0:
            quad_logits = preds['spa_quadrant_logits'][valid]        # [n, 9]
            cx_v = gt_v[:, 0]
            cy_v = gt_v[:, 1]
            quad_targets = _gt_to_quadrant(cx_v, cy_v, self.spatial_grid)  # [n]
            loss_spa_quad = F.cross_entropy(quad_logits, quad_targets)

        # ── L_sem_cls (semantic class discrimination) ─────────────────────────
        # Supervision: which DIOR class is the referred object
        # class_id is targets[:, 1], already stored in batch_targets
        # Forces semantic branch to encode WHAT the object is (class-level)
        loss_sem_cls = torch.tensor(0.0, device=device)
        if 'sem_cls_logits' in preds and lw['sem_cls'] > 0:
            cls_logits = preds['sem_cls_logits'][valid]              # [n, 20]
            cls_targets = class_ids[valid]                           # [n]
            loss_sem_cls = F.cross_entropy(cls_logits, cls_targets)

        # ── L_reconstruct (attribute branch complementarity) ─────────────────
        # Forces the 3 branches to together encode all information.
        # Since sem and spa already have strong direct losses,
        # the attr branch must fill in what they miss → learns attributes.
        # shared_pooled is detached: we don't want to pull sem/spa toward it,
        # only pull attr_vec to fill the gap.
        loss_reconstruct = torch.tensor(0.0, device=device)
        if ('reconstructed' in preds and 'shared_pooled' in preds
                and lw['reconstruct'] > 0):
            rec   = preds['reconstructed'][valid]                    # [n, D]
            target = preds['shared_pooled'][valid].detach()          # [n, D]
            loss_reconstruct = F.mse_loss(rec, target)
            # Guard against instability
            if torch.isnan(loss_reconstruct):
                loss_reconstruct = torch.tensor(0.0, device=device)

        # ── L_orth (semantic ⊥ attribute, kept from v1) ───────────────────────
        loss_orth = torch.tensor(0.0, device=device)
        if 'sem_vec' in preds and 'attr_vec' in preds and lw['orth'] > 0:
            s = F.normalize(preds['sem_vec'], p=2, dim=1, eps=1e-8)
            a = F.normalize(preds['attr_vec'], p=2, dim=1, eps=1e-8)
            loss_orth = (s * a).sum(-1).pow(2).mean()
            if torch.isnan(loss_orth):
                loss_orth = torch.tensor(0.0, device=device)

        # ── Total ─────────────────────────────────────────────────────────────
        total = (
            lw['giou']        * loss_giou        +
            lw['l1']          * loss_l1           +
            lw['spa']         * loss_spa          +
            lw['spa_quad']    * loss_spa_quad     +
            lw['sem_cls']     * loss_sem_cls      +
            lw['reconstruct'] * loss_reconstruct  +
            lw['orth']        * loss_orth
        )

        if torch.isnan(total) or torch.isinf(total):
            total = torch.tensor(0.0, device=device)

        loss_dict = {
            'giou':        (lw['giou']        * loss_giou).item(),
            'l1':          (lw['l1']          * loss_l1).item(),
            'spa':         (lw['spa']         * loss_spa).item(),
            'spa_quad':    (lw['spa_quad']    * loss_spa_quad).item(),
            'sem_cls':     (lw['sem_cls']     * loss_sem_cls).item(),
            'reconstruct': (lw['reconstruct'] * loss_reconstruct).item(),
            'orth':        (lw['orth']        * loss_orth).item(),
            'total':       total.item(),
        }

        return total, loss_dict