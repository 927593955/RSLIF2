"""
models/vlm_grounding.py  (v2)

Key change vs v1:
  The grounding head now receives THREE text streams instead of one:
    - sem (semantic) sequence  [existing]
    - attr (attribute) sequence  [NEW — fixes issue 3]
    - relation_tokens from StructuredPositionPointerEncoder  [NEW — fixes issue 4]

  This change alone is responsible for fixing issues 3 and 4:
    - "the longest bridge" fails because the grounding head never saw the
      attribute branch output — attr_film in FusionNeck modulates feature maps
      at a coarse level but cannot perform token-level comparative attention.
    - "B on the right side of A" fails because relation_tokens (already computed
      by StructuredPositionPointerEncoder) were never forwarded to the grounding
      head in v1 — they were silently discarded.

  Also adds an optional relation_pred_box auxiliary output for L_rel loss.
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
    Used only for the auxiliary L_rel loss; not used during inference.
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
        """
        cfg keys (same as v1 plus):
            n_dec_layers (int) : grounding decoder depth, default 3
        """
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

        # ── Auxiliary relation box head (for L_rel loss) ──────────────────────
        self.relation_box_head = RelationBoxHead(hidden_dim)
        # Project relation_vec (text_dim) → hidden_dim before the MLP
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
            texts : (input_ids [B,L], attention_mask [B,L])
                    OR list of raw strings

        Returns dict:
            pred_box          : [B, 4]  PRIMARY output  (cx,cy,w,h normalised)
            txt_pred_box      : [B, 4]  text-spatial auxiliary
            relation_pred_box : [B, 4]  relation auxiliary (for L_rel)
            sem_vec / spa_vec / attr_vec : [B, D]
            gate              : [B, 1]
            mask              : [B, L]
            sem               : [B, L, D]  semantic sequence
            attn_map          : [B, N] pooling weights from grounding head
        """
        # ── Text encoding ─────────────────────────────────────────────────────
        txt_out = self.text_encoder(texts)

        # ── Visual backbone ───────────────────────────────────────────────────
        raw_feats = self.backbone(imgs)      # [P3, P4, P5]

        # ── Multi-modal fusion neck ───────────────────────────────────────────
        fused_feats = self.neck(raw_feats, txt_out)  # [F3, N4, N5]

        # ── Relation auxiliary prediction (before grounding head) ─────────────
        # relation_vec: [B, text_dim] from StructuredPositionPointerEncoder
        rel_vec_proj = self.relation_vec_proj(txt_out['relation_vec'])  # [B, hidden_dim]
        relation_pred_box = self.relation_box_head(rel_vec_proj)         # [B, 4]

        # ── Grounding head (multi-source iterative decoder) ───────────────────
        # relation_tokens: [B, K, text_dim]  (K = num_pointers - 1 = 3)
        pred_box, attn_map = self.grounding_head(
            feats       = fused_feats,
            text_vec    = txt_out['sem_vec'],
            text_seq    = txt_out['sem'],           # semantic sequence
            text_mask   = txt_out['mask'],
            attr_seq    = txt_out['attr'],          # ← attribute sequence  [FIX issue 3]
            attr_mask   = txt_out['mask'],          # same tokenisation → same mask
            rel_tokens  = txt_out['relation_tokens'],  # ← spatial relation tokens [FIX issue 4]
        )

        return {
            'pred_box':          pred_box,                  # PRIMARY
            'txt_pred_box':      txt_out['pred_box'],       # text-spatial branch
            'relation_pred_box': relation_pred_box,         # relation branch
            'sem_vec':           txt_out['sem_vec'],
            'spa_vec':           txt_out['spa_vec'],
            'attr_vec':          txt_out['attr_vec'],
            'relation_vec':      txt_out['relation_vec'],
            'gate':              txt_out['gate'],
            'mask':              txt_out['mask'],
            'sem':               txt_out['sem'],
            'attn_map':          attn_map,
        }