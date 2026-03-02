"""
models/vlm_grounding.py

Clean RemoteSensingVLM for Visual Grounding / REC.

Key changes from old fusion_module_*.py versions:
- Replaces YOLO Detect head with GroundingHead (direct box regression)
- Uses unified FusionNeck instead of 4 redundant implementations
- Forward output is a plain dict; no more YOLO-format outputs
- Cleaner initialization; backbone pretrained weights loaded once
"""

import torch
import torch.nn as nn

from models.yolo_backbone import YOLOv8Backbone
from models.text_encoder import TriStreamDeBERTa
from models.fusion_neck import FusionNeck
from models.grounding_head import GroundingHead


class RemoteSensingVLM(nn.Module):
    def __init__(self, cfg: dict):
        """
        cfg keys:
            deberta_path  (str)  : path to DeBERTa-v3 checkpoint
            yolo_weight   (str)  : path to yolov8s.pt, or None to skip
            hidden_dim    (int)  : grounding head hidden dim, default 256
            backbone_width(float): YOLOv8 width multiplier, default 0.5
            backbone_depth(float): YOLOv8 depth multiplier, default 0.33
        """
        super().__init__()

        # ── Text encoder ──────────────────────────────────────────────────────
        self.text_encoder = TriStreamDeBERTa(model_path=cfg['deberta_path'])
        text_dim = self.text_encoder.hidden_dim

        # ── Visual backbone ───────────────────────────────────────────────────
        width = cfg.get('backbone_width', 0.5)
        depth = cfg.get('backbone_depth', 0.33)
        self.backbone = YOLOv8Backbone(width=width, depth=depth)
        if cfg.get('yolo_weight'):
            self.backbone.load_pretrained_weights(cfg['yolo_weight'])

        # ── Multi-modal fusion neck ───────────────────────────────────────────
        self.neck = FusionNeck(
            in_channels=self.backbone.out_channels,
            text_dim=text_dim
        )

        # ── Grounding head (replaces YOLO Detect) ────────────────────────────
        hidden_dim = cfg.get('hidden_dim', 256)
        self.grounding_head = GroundingHead(
            in_channels=self.backbone.out_channels,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        # ── Initialize non-pretrained modules ────────────────────────────────
        self._init_custom_modules()

    def _init_custom_modules(self):
        for module in [self.neck, self.grounding_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, imgs, texts):
        """
        Args:
            imgs:  [B, 3, H, W] normalized image tensor
            texts: (input_ids, attention_mask) tuple of [B, L] tensors
                   OR list of raw strings (tokenized internally by text_encoder)
        Returns dict:
            pred_box    : [B, 4] normalized (cx, cy, w, h) in [0, 1]  ← primary output
            txt_pred_box: [B, 4] box predicted from text spatial branch (auxiliary)
            sem_vec     : [B, D] semantic text vector
            spa_vec     : [B, D] spatial text vector
            attr_vec    : [B, D] attribute text vector
            gate        : [B, 1] attribute gate score
            mask        : [B, L] text attention mask
            sem         : [B, L, D] semantic text sequence (for loss / visualization)
            attn_map    : attention weights from grounding head
        """
        # ── Text encoding ─────────────────────────────────────────────────────
        txt_out = self.text_encoder(texts)

        # ── Visual backbone ───────────────────────────────────────────────────
        raw_feats = self.backbone(imgs)  # [P3, P4, P5]

        # ── Multi-modal fusion ────────────────────────────────────────────────
        fused_feats = self.neck(raw_feats, txt_out)  # [F3, N4, N5]

        # ── Grounding prediction ──────────────────────────────────────────────
        # Use semantic sequence for rich text conditioning in cross-attention
        pred_box, attn_map = self.grounding_head(
            feats=fused_feats,
            text_vec=txt_out['sem_vec'],
            text_seq=txt_out['sem'],
            text_mask=txt_out['mask']
        )

        return {
            'pred_box':     pred_box,                  # [B, 4]  PRIMARY OUTPUT
            'txt_pred_box': txt_out['pred_box'],        # [B, 4]  auxiliary
            'sem_vec':      txt_out['sem_vec'],
            'spa_vec':      txt_out['spa_vec'],
            'attr_vec':     txt_out['attr_vec'],
            'relation_vec': txt_out.get('relation_vec', txt_out['spa_vec']),
            'gate':         txt_out['gate'],
            'mask':         txt_out['mask'],
            'sem':          txt_out['sem'],
            'attn_map':     attn_map
        }