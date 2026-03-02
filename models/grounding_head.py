"""
models/grounding_head.py  (v2 - optimized for speed)

Key change vs v1:
  - Removed expensive O(N²) self-attention on all 8400 spatial tokens.
    (80²+40²+20² = 8400 tokens → 8400² attention matrix = ~70M pairs per sample)
  - Each scale is now avg-pooled to pool_size×pool_size (default 8×8=64) tokens
    before cross-attention.  3 scales × 64 = 192 tokens total.
  - Cross-attention: 192 queries × 77 text keys  (was 8400 × 77)
  - ~44× fewer tokens in attention → ~5-10× faster head.
  - Kept text-guided spatial pooling for final box regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingHead(nn.Module):
    def __init__(
        self,
        in_channels,
        text_dim,
        hidden_dim=256,
        num_heads=8,
        dropout=0.1,
        pool_size=8,        # Each scale pooled to pool_size×pool_size before attn
    ):
        """
        Args:
            in_channels : list of ints e.g. [128, 256, 512]
            text_dim    : text encoder output dim (e.g. 1024 for DeBERTa-large)
            hidden_dim  : internal working dimension
            num_heads   : attention heads
            dropout     : dropout probability
            pool_size   : spatial pool before cross-attention (reduces 8400→192)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_size  = pool_size
        num_scales = len(in_channels)

        # ── 1. Per-scale projection ───────────────────────────────────────────
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, hidden_dim // 8), hidden_dim),
                nn.GELU(),
            ) for c in in_channels
        ])
        self.level_embed = nn.Parameter(
            torch.randn(num_scales, hidden_dim) * 0.02
        )

        # ── 2. Text projection ────────────────────────────────────────────────
        self.text_seq_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.text_vec_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ── 3. Cross-attention (pooled visual tokens attend to text) ──────────
        #   Input size: [B, num_scales×pool_size², D]  e.g. [B, 192, 256]
        #   Much cheaper than the original [B, 8400, D]
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # ── 4. Lightweight FFN ────────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # ── 5. Text-guided spatial pooling ────────────────────────────────────
        self.pool_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pool_key_proj   = nn.Linear(hidden_dim, hidden_dim)

        # ── 6. Box regression MLP ─────────────────────────────────────────────
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),   # normalized (cx, cy, w, h) in [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Start predicting centered boxes
        nn.init.constant_(self.box_head[-2].bias, 0.5)

    def forward(self, feats, text_vec, text_seq=None, text_mask=None):
        """
        Args:
            feats     : list of [B, C_i, H_i, W_i]
            text_vec  : [B, text_dim]  global text vector
            text_seq  : [B, L, text_dim]  full token sequence (optional)
            text_mask : [B, L]  1=valid, 0=padding

        Returns:
            pred_box    : [B, 4]  normalized (cx, cy, w, h) in [0, 1]
            attn_weights: [B×heads, N_vis, L]  cross-attn map for visualization
        """
        B = feats[0].shape[0]
        P = self.pool_size

        # ── Step 1: Project + pool each scale to P×P ──────────────────────────
        vis_tokens_list = []
        for i, (feat, proj) in enumerate(zip(feats, self.scale_projs)):
            t = proj(feat)                                     # [B, D, H, W]
            t = F.adaptive_avg_pool2d(t, (P, P))              # [B, D, P, P]
            t = t.flatten(2).transpose(1, 2)                   # [B, P², D]
            t = t + self.level_embed[i]                        # scale identity
            vis_tokens_list.append(t)

        vis_tokens = torch.cat(vis_tokens_list, dim=1)         # [B, 3×P², D]

        # ── Step 2: Project text ───────────────────────────────────────────────
        if text_seq is not None:
            text_feats = self.text_seq_proj(text_seq)          # [B, L, D]
        else:
            text_feats = self.text_vec_proj(text_vec).unsqueeze(1)   # [B, 1, D]

        key_padding_mask = None
        if text_mask is not None and text_seq is not None:
            key_padding_mask = (text_mask == 0).bool()         # True = ignore

        # ── Step 3: Cross-attention ────────────────────────────────────────────
        ca_out, attn_weights = self.cross_attn(
            query=vis_tokens,
            key=text_feats,
            value=text_feats,
            key_padding_mask=key_padding_mask,
        )
        vis_tokens = self.cross_norm(vis_tokens + ca_out)

        # ── Step 4: FFN ────────────────────────────────────────────────────────
        vis_tokens = self.ffn_norm(vis_tokens + self.ffn(vis_tokens))

        # ── Step 5: Text-guided spatial pooling ───────────────────────────────
        q = self.pool_query_proj(self.text_vec_proj(text_vec))  # [B, D]
        k = self.pool_key_proj(vis_tokens)                       # [B, N, D]
        pool_scores  = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)
        pool_scores  = pool_scores / (self.hidden_dim ** 0.5)   # [B, N]
        pool_weights = F.softmax(pool_scores, dim=-1)
        pooled = torch.bmm(
            pool_weights.unsqueeze(1), vis_tokens
        ).squeeze(1)                                             # [B, D]

        # ── Step 6: Predict box ────────────────────────────────────────────────
        pred_box = self.box_head(pooled)                         # [B, 4]

        return pred_box, attn_weights