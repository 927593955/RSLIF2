"""
models/grounding_head.py

Text-conditioned single-box grounding head for Visual Grounding / REC tasks.

Replaces the YOLO Detect head. Given fused multi-scale visual features and
text features, directly predicts ONE normalized bounding box [cx, cy, w, h].

Design:
  1. Project multi-scale features to unified hidden_dim with level embeddings
  2. Cross-attention: visual tokens attend to text sequence (text conditions vision)
  3. Text-guided spatial pooling: compute attention weights over spatial locations
  4. Lightweight MLP → predict [cx, cy, w, h] in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingHead(nn.Module):
    def __init__(self, in_channels, text_dim, hidden_dim=256, num_heads=8, dropout=0.1):
        """
        Args:
            in_channels: list of channel counts for each feature scale, e.g. [128, 256, 512]
            text_dim:    dimension of text encoder output (e.g. 1024 for DeBERTa-large)
            hidden_dim:  internal working dimension
            num_heads:   number of attention heads
            dropout:     dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── 1. Multi-scale visual feature projection ──────────────────────────
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, hidden_dim // 8), hidden_dim),
                nn.GELU()
            ) for c in in_channels
        ])
        # Learnable level embeddings to encode scale identity
        self.level_embed = nn.Parameter(torch.randn(len(in_channels), hidden_dim) * 0.02)

        # ── 2. Text projection ─────────────────────────────────────────────────
        self.text_seq_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.text_vec_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # ── 3. Cross-attention block (visual attends to text) ──────────────────
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # ── 4. Self-attention for spatial context propagation ──────────────────
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(hidden_dim)

        # ── 5. FFN ─────────────────────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # ── 6. Text-guided spatial pooling ────────────────────────────────────
        # Uses text_vec as query to compute a soft attention mask over spatial locations
        self.pool_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pool_key_proj = nn.Linear(hidden_dim, hidden_dim)

        # ── 7. Box prediction MLP ─────────────────────────────────────────────
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()  # output in [0, 1] for normalized cx, cy, w, h
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Initialize box head final layer to predict near-center boxes
        nn.init.constant_(self.box_head[-2].bias, 0.5)  # cx, cy, w, h → 0.5

    def forward(self, feats, text_vec, text_seq=None, text_mask=None):
        """
        Args:
            feats:     list of [B, C_i, H_i, W_i] multi-scale visual features
            text_vec:  [B, text_dim] global text vector (e.g. sem_vec from DeBERTa)
            text_seq:  [B, L, text_dim] full text token sequence for rich cross-attention
                       If None, uses text_vec expanded to [B, 1, text_dim]
            text_mask: [B, L] binary mask (1=valid token, 0=padding)
        Returns:
            pred_box:    [B, 4] normalized (cx, cy, w, h)
            attn_weights:[B, L, sum(HW)] cross-attention map for visualization
        """
        B = feats[0].shape[0]

        # ── Step 1: Project and flatten multi-scale features ──────────────────
        vis_tokens_list = []
        for i, (feat, proj) in enumerate(zip(feats, self.scale_projs)):
            t = proj(feat)                           # [B, hidden_dim, H, W]
            t = t.flatten(2).transpose(1, 2)         # [B, H*W, hidden_dim]
            t = t + self.level_embed[i]              # add level embedding
            vis_tokens_list.append(t)
        vis_tokens = torch.cat(vis_tokens_list, dim=1)  # [B, sum(H*W), hidden_dim]

        # ── Step 2: Project text features ─────────────────────────────────────
        if text_seq is not None:
            text_feats = self.text_seq_proj(text_seq)   # [B, L, hidden_dim]
        else:
            text_feats = self.text_vec_proj(text_vec).unsqueeze(1)  # [B, 1, hidden_dim]

        # Key padding mask: True positions are IGNORED by attention
        key_padding_mask = None
        if text_mask is not None and text_seq is not None:
            key_padding_mask = (text_mask == 0).bool()  # [B, L]

        # ── Step 3: Cross-attention (visual tokens attend to text) ────────────
        ca_out, attn_weights = self.cross_attn(
            query=vis_tokens,
            key=text_feats,
            value=text_feats,
            key_padding_mask=key_padding_mask
        )
        vis_tokens = self.cross_norm(vis_tokens + ca_out)

        # ── Step 4: Self-attention (propagate spatial context) ─────────────────
        sa_out, _ = self.self_attn(vis_tokens, vis_tokens, vis_tokens)
        vis_tokens = self.self_norm(vis_tokens + sa_out)

        # ── Step 5: FFN ────────────────────────────────────────────────────────
        vis_tokens = self.ffn_norm(vis_tokens + self.ffn(vis_tokens))

        # ── Step 6: Text-guided spatial pooling ───────────────────────────────
        # Compute a soft attention mask over spatial tokens using text as query
        q = self.pool_query_proj(self.text_vec_proj(text_vec))  # [B, hidden_dim]
        k = self.pool_key_proj(vis_tokens)                       # [B, sum(HW), hidden_dim]
        pool_scores = torch.bmm(
            k, q.unsqueeze(-1)
        ).squeeze(-1) / (self.hidden_dim ** 0.5)                 # [B, sum(HW)]
        pool_weights = F.softmax(pool_scores, dim=-1)            # [B, sum(HW)]
        pooled = torch.bmm(
            pool_weights.unsqueeze(1), vis_tokens
        ).squeeze(1)                                             # [B, hidden_dim]

        # ── Step 7: Predict box ────────────────────────────────────────────────
        pred_box = self.box_head(pooled)  # [B, 4]

        return pred_box, attn_weights