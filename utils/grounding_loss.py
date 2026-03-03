"""
utils/grounding_loss.py  (v3 — adds L_div to prevent branch collapse)

NEW LOSS: L_div — Branch Diversity Regularisation
  Purpose: Force sem_vec, spa_vec, attr_vec to be different from each other.
  Formula: cosine_sim(sem, spa)² + cosine_sim(sem, attr)² + cosine_sim(spa, attr)²
  Active:  From epoch 0 (lambda_div default = 0.5)

  Without this, all three branches converge to the same representation in Stage 1
  because they start from identical weights and have no branch-specific loss signal
  until Stage 2. By the time Stage 2 starts (epoch 10), the collapse is already
  entrenched and sem_cls / spa_quad losses can't un-collapse it.

  The diversity loss provides a constant push to keep branches different, so that
  when the branch-specific losses arrive in Stage 2, there is already some
  specialisation to build on.

All other losses unchanged from v2.
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
    lt    = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb    = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh    = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    iou   = inter / (union + 1e-6)
    enc_lt   = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enc_rb   = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enc_wh   = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[..., 0] * enc_wh[..., 1]
    giou     = iou - (enc_area - union) / (enc_area + 1e-6)
    return giou


def _get_per_sample_gt(batch_targets: torch.Tensor, batch_size: int, device):
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
    col = (cx * grid).long().clamp(0, grid - 1)
    row = (cy * grid).long().clamp(0, grid - 1)
    return row * grid + col


# =============================================================================
# Main loss
# =============================================================================

class GroundingLoss(nn.Module):
    """
    v3: added L_div (branch diversity loss, active from epoch 0).

    LOSS TERMS:
      L_giou        — GIoU between pred_box and GT box
      L_l1          — L1 between pred_box and GT box
      L_spa         — smooth-L1 between txt_pred_box and GT box (spatial text branch)
      L_spa_quad    — cross-entropy: spatial branch → 3×3 grid cell
      L_sem_cls     — cross-entropy: semantic branch → DIOR class id
      L_reconstruct — MSE: MLP(sem+spa+attr) → shared_pooled (complementarity)
      L_orth        — (sem·attr)²  (orthogonality push)
      L_div  [NEW]  — (sem·spa)² + (sem·attr)² + (spa·attr)²  (diversity, active from epoch 0)

    DEFAULT LAMBDA SCHEDULE (via get_lambdas in train.py):
      Stage 1 (0→stage1): giou + l1 + div + small sem_cls + small spa_quad
      Stage 2 (+offset):  + spa + spa_quad + sem_cls at full weight
      Stage 3 (→end):     + reconstruct (ramped) + orth
    """

    def __init__(
        self,
        lambda_giou:        float = 2.0,
        lambda_l1:          float = 5.0,
        lambda_spa:         float = 1.0,
        lambda_spa_quad:    float = 0.5,
        lambda_sem_cls:     float = 1.0,
        lambda_reconstruct: float = 0.5,
        lambda_orth:        float = 0.3,
        lambda_div:         float = 0.5,   # NEW: branch diversity
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
        self.lambda_div         = lambda_div
        self.spatial_grid       = spatial_grid

    def forward(self, preds: dict, batch_targets: torch.Tensor, lambdas: dict = None):
        device = batch_targets.device
        B = preds['pred_box'].shape[0]

        lw = {
            'giou':        self.lambda_giou,
            'l1':          self.lambda_l1,
            'spa':         self.lambda_spa,
            'spa_quad':    self.lambda_spa_quad,
            'sem_cls':     self.lambda_sem_cls,
            'reconstruct': self.lambda_reconstruct,
            'orth':        self.lambda_orth,
            'div':         self.lambda_div,
        }
        if lambdas:
            lw.update({k: float(v) for k, v in lambdas.items() if k in lw})

        gt_boxes, class_ids, valid = _get_per_sample_gt(batch_targets, B, device)

        zero = torch.tensor(0.0, device=device, requires_grad=True)
        if not valid.any():
            return zero, {k: 0.0 for k in ('giou','l1','spa','spa_quad',
                                             'sem_cls','reconstruct','orth','div','total')}

        pred   = preds['pred_box']
        pred_v = pred[valid]
        gt_v   = gt_boxes[valid]

        # ── L_GIoU ──────────────────────────────────────────────────────────
        loss_giou = (1 - generalized_box_iou(
            box_cxcywh_to_xyxy(pred_v),
            box_cxcywh_to_xyxy(gt_v)
        ).diag()).mean()

        # ── L_L1 ────────────────────────────────────────────────────────────
        loss_l1 = F.l1_loss(pred_v, gt_v)

        # ── L_spa ───────────────────────────────────────────────────────────
        loss_spa = torch.tensor(0.0, device=device)
        if 'txt_pred_box' in preds and lw['spa'] > 0:
            loss_spa = F.smooth_l1_loss(preds['txt_pred_box'][valid], gt_v, beta=0.1)

        # ── L_spa_quad ──────────────────────────────────────────────────────
        loss_spa_quad = torch.tensor(0.0, device=device)
        if 'spa_quadrant_logits' in preds and lw['spa_quad'] > 0:
            quad_logits  = preds['spa_quadrant_logits'][valid]
            quad_targets = _gt_to_quadrant(gt_v[:, 0], gt_v[:, 1], self.spatial_grid)
            loss_spa_quad = F.cross_entropy(quad_logits, quad_targets)

        # ── L_sem_cls ───────────────────────────────────────────────────────
        loss_sem_cls = torch.tensor(0.0, device=device)
        if 'sem_cls_logits' in preds and lw['sem_cls'] > 0:
            loss_sem_cls = F.cross_entropy(
                preds['sem_cls_logits'][valid], class_ids[valid]
            )

        # ── L_reconstruct ───────────────────────────────────────────────────
        loss_reconstruct = torch.tensor(0.0, device=device)
        if ('reconstructed' in preds and 'shared_pooled' in preds
                and lw['reconstruct'] > 0):
            rec    = preds['reconstructed'][valid]
            target = preds['shared_pooled'][valid].detach()
            loss_reconstruct = F.mse_loss(rec, target)
            if torch.isnan(loss_reconstruct):
                loss_reconstruct = torch.tensor(0.0, device=device)

        # ── L_orth ──────────────────────────────────────────────────────────
        loss_orth = torch.tensor(0.0, device=device)
        if 'sem_vec' in preds and 'attr_vec' in preds and lw['orth'] > 0:
            s = F.normalize(preds['sem_vec'], p=2, dim=1, eps=1e-8)
            a = F.normalize(preds['attr_vec'], p=2, dim=1, eps=1e-8)
            loss_orth = (s * a).sum(-1).pow(2).mean()
            if torch.isnan(loss_orth):
                loss_orth = torch.tensor(0.0, device=device)

        # ── L_div (NEW) ─────────────────────────────────────────────────────
        # Forces three branches to maintain different representations.
        # Active from epoch 0 — this is the key fix for branch collapse.
        # By penalising pairwise cosine similarity, it provides a constant
        # gradient signal that differentiates the three branch encoders even
        # before the branch-specific losses (sem_cls, spa_quad) kick in.
        loss_div = torch.tensor(0.0, device=device)
        if ('sem_vec' in preds and 'spa_vec' in preds and 'attr_vec' in preds
                and lw['div'] > 0):
            s  = F.normalize(preds['sem_vec'],  p=2, dim=1, eps=1e-8)
            sp = F.normalize(preds['spa_vec'],  p=2, dim=1, eps=1e-8)
            a  = F.normalize(preds['attr_vec'], p=2, dim=1, eps=1e-8)
            # Penalise high pairwise cosine similarity (squared for smooth gradient)
            sim_ss = (s  * sp).sum(-1).pow(2).mean()   # sem vs spa
            sim_sa = (s  * a ).sum(-1).pow(2).mean()   # sem vs attr
            sim_pa = (sp * a ).sum(-1).pow(2).mean()   # spa vs attr
            loss_div = (sim_ss + sim_sa + sim_pa) / 3.0
            if torch.isnan(loss_div):
                loss_div = torch.tensor(0.0, device=device)

        # ── Total ────────────────────────────────────────────────────────────
        total = (
            lw['giou']        * loss_giou        +
            lw['l1']          * loss_l1           +
            lw['spa']         * loss_spa          +
            lw['spa_quad']    * loss_spa_quad     +
            lw['sem_cls']     * loss_sem_cls      +
            lw['reconstruct'] * loss_reconstruct  +
            lw['orth']        * loss_orth         +
            lw['div']         * loss_div
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
            'div':         (lw['div']         * loss_div).item(),
            'total':       total.item(),
        }

        return total, loss_dict