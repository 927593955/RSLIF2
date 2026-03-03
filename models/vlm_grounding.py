"""
models/vlm_grounding.py  (v3 — fixes KeyError on 'gate' key)

BUG FIX:
  OLD: 'gate': txt_out['gate']  ← KeyError
  The new text_encoder.py (v2, with TokenTypeGatedPooler) removed attr_gate_head
  entirely. It no longer outputs a 'gate' key. Instead it outputs:
    'sem_gate':  [B, L]  gate weights for semantic branch
    'spa_gate':  [B, L]  gate weights for spatial branch
    'attr_gate': [B, L]  gate weights for attribute branch
  These are per-token weights for analysis via branch_probe.py.
  They are NOT used in grounding_loss.py so removing 'gate' from the return
  dict of vlm_grounding.py has no effect on training.

  Also added explicit forwarding of the gate weight outputs so branch_probe.py
  can read them from the model output dict without needing to call text_encoder
  a second time.

  Also added 'sem_cls_logits', 'spa_quadrant_logits', 'reconstructed',
  'shared_pooled' to the return dict so grounding_loss.py can compute
  all branch supervision losses from the top-level model output.
"""

import torch
import torch.nn as nn

from models.yolo_backbone import YOLOv8Backbone
from models.text_encoder import TriStreamDeBERTa
from models.fusion_neck import FusionNeck
from models.grounding_head import GroundingHead


class RelationBoxHead(nn.Module):
    """
    Lightweight MLP that predicts a box from the relation vector.
    Used for the auxiliary L_rel loss only.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RemoteSensingVLM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        # ── Text encoder ──────────────────────────────────────────────────────
        self.text_encoder = TriStreamDeBERTa(model_path=cfg['deberta_path'])
        text_dim = self.text_encoder.hidden_dim

        # ── Visual backbone ───────────────────────────────────────────────────
        self.backbone = YOLOv8Backbone(
            width=cfg.get('backbone_width', 0.5),
            depth=cfg.get('backbone_depth', 0.33),
        )
        if cfg.get('yolo_weight'):
            self.backbone.load_pretrained_weights(cfg['yolo_weight'])

        # ── Multi-modal fusion neck ───────────────────────────────────────────
        self.neck = FusionNeck(
            in_channels=self.backbone.out_channels,
            text_dim=text_dim,
        )

        # ── Grounding head ────────────────────────────────────────────────────
        hidden_dim = cfg.get('hidden_dim', 256)
        self.grounding_head = GroundingHead(
            in_channels  = self.backbone.out_channels,
            text_dim     = text_dim,
            hidden_dim   = hidden_dim,
            num_heads    = 8,
            dropout      = 0.1,
            pool_size    = 8,
            n_dec_layers = cfg.get('n_dec_layers', 3),
        )

        # ── Auxiliary relation box head ────────────────────────────────────────
        self.relation_box_head = RelationBoxHead(hidden_dim)
        self.relation_vec_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self._init_custom_modules()

    def _init_custom_modules(self):
        for module in [self.neck, self.grounding_head,
                       self.relation_box_head, self.relation_vec_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')

    def forward(self, imgs: torch.Tensor, texts):
        """
        Args:
            imgs  : [B, 3, H, W]
            texts : (input_ids [B,L], attention_mask [B,L])  OR list of strings

        Returns dict with ALL keys needed by:
            - grounding_loss.py  (pred_box, txt_pred_box, sem_cls_logits, ...)
            - branch_probe.py    (sem_vec, spa_vec, attr_vec, sem_gate, spa_gate, attr_gate)
            - val.py             (pred_box)
        """
        # ── Text encoding ─────────────────────────────────────────────────────
        txt_out = self.text_encoder(texts)

        # ── Visual backbone ───────────────────────────────────────────────────
        raw_feats = self.backbone(imgs)      # [P3, P4, P5]

        # ── Multi-modal fusion neck ───────────────────────────────────────────
        fused_feats = self.neck(raw_feats, txt_out)  # [F3, N4, N5]

        # ── Relation auxiliary prediction ─────────────────────────────────────
        rel_vec_proj      = self.relation_vec_proj(txt_out['relation_vec'])
        relation_pred_box = self.relation_box_head(rel_vec_proj)

        # ── Grounding head ────────────────────────────────────────────────────
        pred_box, attn_map = self.grounding_head(
            feats       = fused_feats,
            text_vec    = txt_out['sem_vec'],
            text_seq    = txt_out['sem'],
            text_mask   = txt_out['mask'],
            attr_seq    = txt_out['attr'],
            attr_mask   = txt_out['mask'],
            rel_tokens  = txt_out['relation_tokens'],
        )

        # ── Build return dict ─────────────────────────────────────────────────
        # BUG FIX: removed 'gate': txt_out['gate']  (key doesn't exist in v2 text encoder)
        # Added: sem_gate, spa_gate, attr_gate (the per-token gate weights)
        # Added: sem_cls_logits, spa_quadrant_logits, reconstructed, shared_pooled
        #        so grounding_loss.py can compute all branch supervision losses
        #        directly from the top-level model output dict.
        return {
            # PRIMARY output
            'pred_box':             pred_box,                       # [B, 4]
            # Auxiliary box predictions (for branch supervision losses)
            'txt_pred_box':         txt_out['pred_box'],            # [B, 4] spatial branch
            'relation_pred_box':    relation_pred_box,              # [B, 4] relation branch
            # Branch supervision logits (consumed by grounding_loss.py)
            'sem_cls_logits':       txt_out['sem_cls_logits'],      # [B, 20]
            'spa_quadrant_logits':  txt_out['spa_quadrant_logits'], # [B, 9]
            'reconstructed':        txt_out['reconstructed'],       # [B, D]
            'shared_pooled':        txt_out['shared_pooled'],       # [B, D]
            # Branch vectors (for grounding_loss orth + for branch_probe.py)
            'sem_vec':              txt_out['sem_vec'],             # [B, D]
            'spa_vec':              txt_out['spa_vec'],             # [B, D]
            'attr_vec':             txt_out['attr_vec'],            # [B, D]
            # Gate weights (for branch_probe.py analysis only)
            'sem_gate':             txt_out['sem_gate'],            # [B, L]
            'spa_gate':             txt_out['spa_gate'],            # [B, L]
            'attr_gate':            txt_out['attr_gate'],           # [B, L]
            # Spatial relation
            'relation_vec':         txt_out['relation_vec'],        # [B, D]
            # Mask and sequences
            'mask':                 txt_out['mask'],                # [B, L]
            'sem':                  txt_out['sem'],                 # [B, L, D]
            # Attention map from grounding head (for visualization)
            'attn_map':             attn_map,                       # [B, N]
        }