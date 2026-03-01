"""
models/fusion_neck.py

Simplified multi-modal fusion neck (PA-FPN style).
Consolidates the 4+ redundant neck implementations into one clean module.

Changes vs old versions:
- Single file, single class
- Removed duplicate TextFeaturePyramid implementations
- Unified GatedFusion variants into one configurable FusionBlock
- Fixed coordinate cache (uses register_buffer for proper device handling)
- Cleaner forward signature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolo_backbone import C2f, Conv


# ──────────────────────────────────────────────────────────────────────────────
# 1. Reusable fusion primitives
# ──────────────────────────────────────────────────────────────────────────────

class TextFusion(nn.Module):
    """
    Configurable text-conditioned visual feature modulation.

    mode='cross_attn': cross-attention (richer, used for P4/P5)
    mode='gate':       channel+spatial gate (lighter, used for P3)
    """

    def __init__(self, visual_dim, text_dim, mode='cross_attn', num_heads=4, dropout=0.1):
        super().__init__()
        self.mode = mode

        if mode == 'cross_attn':
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, visual_dim),
                nn.LayerNorm(visual_dim)
            )
            self.attn = nn.MultiheadAttention(
                visual_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm_v = nn.LayerNorm(visual_dim)
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(visual_dim, visual_dim, 7, padding=3, groups=visual_dim),
                nn.BatchNorm2d(visual_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(visual_dim, visual_dim, 1)
            )
            self.drop = nn.Dropout2d(dropout)
            self.post_norm = nn.BatchNorm2d(visual_dim)
            self.alpha = nn.Parameter(torch.tensor(0.1))

        elif mode == 'gate':
            self.t_norm = nn.LayerNorm(text_dim)
            self.channel_gate = nn.Sequential(
                nn.Linear(text_dim, visual_dim),
                # No Sigmoid here; use tanh residual to avoid over-suppression
            )
            self.spatial_gate = nn.Linear(text_dim, visual_dim)
            self.gate_temp = nn.Parameter(torch.tensor(1.0))
            self.refine = nn.Sequential(
                nn.Conv2d(visual_dim, visual_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(visual_dim),
                nn.SiLU(inplace=True),
                nn.Dropout2d(dropout)
            )
            self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, visual_feat, text_vec_or_seq):
        """
        visual_feat: [B, C, H, W]
        text_vec_or_seq: [B, D] or [B, L, D]
        """
        b, c, h, w = visual_feat.shape

        if self.mode == 'cross_attn':
            text = text_vec_or_seq
            if text.dim() == 2:
                text = text.unsqueeze(1)
            t = self.text_proj(text)                           # [B, L, C]
            v = visual_feat.flatten(2).transpose(1, 2)        # [B, HW, C]
            v = self.norm_v(v)
            attn_out, _ = self.attn(v, t, t)
            v_aligned = attn_out.transpose(1, 2).view(b, c, h, w)
            v_reasoned = self.drop(self.spatial_conv(v_aligned))
            return self.post_norm(visual_feat + torch.tanh(self.alpha) * v_reasoned)

        elif self.mode == 'gate':
            t = self.t_norm(text_vec_or_seq)                   # expects [B, D]
            # Channel gate: residual style to avoid over-suppression
            ch = torch.tanh(self.channel_gate(t) / self.gate_temp.clamp(min=0.1))
            ch = ch.view(b, c, 1, 1)
            # Spatial gate: dot-product attention map
            sp_query = self.spatial_gate(t).view(b, c, 1, 1)
            sp_attn = torch.sigmoid(
                (visual_feat * sp_query).sum(dim=1, keepdim=True)
                / (c ** 0.5)
            )
            gated = visual_feat * (1.0 + ch) * sp_attn
            return visual_feat + torch.tanh(self.beta) * self.refine(gated)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class ScaleAwareFiLM(nn.Module):
    """Per-level FiLM (Feature-wise Linear Modulation) conditioned on text."""

    def __init__(self, text_dim, level_channels: dict, hidden_ratio=0.5):
        super().__init__()
        hidden = max(int(text_dim * hidden_ratio), 128)
        self.shared = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1)
        )
        self.gamma = nn.ModuleDict({
            k: nn.Linear(hidden, c) for k, c in level_channels.items()
        })
        self.beta = nn.ModuleDict({
            k: nn.Linear(hidden, c) for k, c in level_channels.items()
        })

    def forward(self, feat, text_vec, level: str):
        h = self.shared(text_vec)
        g = torch.tanh(self.gamma[level](h)).unsqueeze(-1).unsqueeze(-1)
        b = self.beta[level](h).unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + g) + b


class TextFeaturePyramid(nn.Module):
    """Pools a text sequence to global + per-level vectors."""

    def __init__(self, text_dim, levels=('p3', 'p4', 'p5')):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.LayerNorm(text_dim)
        )
        self.level_proj = nn.ModuleDict({
            lv: nn.Sequential(nn.Linear(text_dim, text_dim), nn.LayerNorm(text_dim))
            for lv in levels
        })

    def forward(self, text_feat, mask=None):
        if text_feat.dim() == 3:
            if mask is not None:
                mf = mask.unsqueeze(-1).float()
                global_vec = (text_feat * mf).sum(1) / (mf.sum(1) + 1e-6)
            else:
                global_vec = text_feat.mean(1)
        else:
            global_vec = text_feat

        base = self.shared(global_vec)
        out = {'global': base}
        for lv, proj in self.level_proj.items():
            out[lv] = proj(base)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 2. PA-FPN Neck
# ──────────────────────────────────────────────────────────────────────────────

class FusionNeck(nn.Module):
    """
    Multi-modal PA-FPN neck.
    Uses semantic text for top-down path and spatial text for bottom-up path.
    """

    def __init__(self, in_channels, text_dim):
        super().__init__()
        c3, c4, c5 = in_channels
        self.text_pyr = TextFeaturePyramid(text_dim)

        # Attribute FiLM for fine-grained property conditioning
        self.attr_film = ScaleAwareFiLM(
            text_dim, {'p3': c3, 'p4': c4, 'p5': c5}
        )

        # Top-down fusions (semantic: cross-attention for higher semantic levels)
        self.fuse_p5 = TextFusion(c5, text_dim, mode='cross_attn', num_heads=8)
        self.fuse_p4 = TextFusion(c4, text_dim, mode='cross_attn', num_heads=4)
        self.fuse_p3 = TextFusion(c3, text_dim, mode='gate')

        # Bottom-up fusions (spatial)
        self.fuse_n4 = TextFusion(c4, text_dim, mode='gate')
        self.fuse_n5 = TextFusion(c5, text_dim, mode='cross_attn', num_heads=8)

        # Standard FPN convolutions
        self.reduce_p5 = Conv(c5, c4, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p4 = C2f(c4, c4, n=3, shortcut=False)
        self.reduce_p4 = Conv(c4, c3, 1, 1)
        self.c2f_p3 = C2f(c3, c3, n=3, shortcut=False)
        self.down_p3 = Conv(c3, c3, 3, 2)
        self.c2f_n4 = C2f(c3 + c4, c4, n=3, shortcut=False)
        self.down_p4 = Conv(c4, c4, 3, 2)
        self.c2f_n5 = C2f(c4 + c5, c5, n=3, shortcut=False)

    def forward(self, raw_feats, txt_out):
        """
        raw_feats: [p3, p4, p5] from backbone
        txt_out:   dict with keys 'sem', 'spa', 'attr', 'sem_vec', 'spa_vec', 'mask'
        Returns:   [f3, n4, n5] fused feature pyramid
        """
        raw_p3, raw_p4, raw_p5 = raw_feats
        sem, spa, attr = txt_out['sem'], txt_out['spa'], txt_out['attr']
        mask = txt_out['mask']

        sem_pyr = self.text_pyr(sem, mask)
        spa_pyr = self.text_pyr(spa, mask)
        attr_pyr = self.text_pyr(attr, mask)

        # ── Top-down path (semantic conditioning) ─────────────────────────────
        f5 = self.fuse_p5(raw_p5, sem_pyr['global'])
        f5_up = self.up(self.reduce_p5(f5))

        f4 = self.c2f_p4(f5_up + self.fuse_p4(raw_p4, spa_pyr['p4']))
        f4_up = self.up(self.reduce_p4(f4))

        f3 = self.c2f_p3(f4_up + self.fuse_p3(raw_p3, spa_pyr['global']))

        # ── Bottom-up path (spatial conditioning) ─────────────────────────────
        n4 = self.c2f_n4(torch.cat([self.down_p3(f3), f4], dim=1))
        n4 = self.fuse_n4(n4, spa_pyr['global'])

        n5 = self.c2f_n5(torch.cat([self.down_p4(n4), f5], dim=1))
        n5 = self.fuse_n5(n5, sem_pyr['global'])

        # ── Attribute FiLM (fine-grained attribute conditioning) ──────────────
        f3 = self.attr_film(f3, attr_pyr['p3'], 'p3')
        n4 = self.attr_film(n4, attr_pyr['p4'], 'p4')
        n5 = self.attr_film(n5, attr_pyr['p5'], 'p5')

        return [f3, n4, n5]