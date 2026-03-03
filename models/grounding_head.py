"""
models/grounding_head.py  (v4 — fix box head bias initialisation)

BUG FIX: Box head bias was set to 0.5 for ALL four outputs, causing the model
to start predicting boxes of size (62%, 62%) centred at (62%, 62%).

OLD:
    nn.init.constant_(self.box_head[-2].bias, 0.5)
    # → sigmoid(0.5) ≈ 0.62 for cx, cy, w, h
    # → 62%×62% box in the lower-right quadrant
    # → GIoU loss must fight this 130+ epochs before reasonable localisation

NEW:
    nn.init.constant_(self.box_head[-2].bias[:2], 0.0)   # cx, cy → 0.5 (image centre)
    nn.init.constant_(self.box_head[-2].bias[2:], -2.0)  # w, h   → 0.12 (small prior)
    # → starts predicting a small 12%×12% box at image centre
    # → this is a much better neutral prior for remote sensing objects

Everything else is unchanged from v3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Single decoder layer
# ─────────────────────────────────────────────────────────────────────────────

class MultiSourceDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        mha  = lambda: nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        norm = lambda: nn.LayerNorm(d_model)

        self.self_attn = mha();  self.norm1 = norm()
        self.sem_attn  = mha();  self.norm2 = norm()
        self.attr_attn = mha();  self.norm3 = norm()
        self.rel_attn  = mha();  self.norm4 = norm()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm5 = norm()

        self.attr_gate = nn.Parameter(torch.full((1,), -4.0))
        self.rel_gate  = nn.Parameter(torch.full((1,), -4.0))

    def forward(self, vis, sem_kv, attr_kv, rel_kv,
                sem_mask=None, attr_mask=None):
        v2, _ = self.self_attn(vis, vis, vis)
        vis = self.norm1(vis + v2)

        s2, _ = self.sem_attn(vis, sem_kv, sem_kv, key_padding_mask=sem_mask)
        vis = self.norm2(vis + s2)

        a2, _ = self.attr_attn(vis, attr_kv, attr_kv, key_padding_mask=attr_mask)
        vis = self.norm3(vis + torch.sigmoid(self.attr_gate) * a2)

        r2, _ = self.rel_attn(vis, rel_kv, rel_kv)
        vis = self.norm4(vis + torch.sigmoid(self.rel_gate) * r2)

        vis = self.norm5(vis + self.ffn(vis))
        return vis


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Main grounding head
# ─────────────────────────────────────────────────────────────────────────────

class GroundingHead(nn.Module):
    def __init__(
        self,
        in_channels,
        text_dim,
        hidden_dim: int   = 256,
        num_heads:  int   = 8,
        dropout:    float = 0.1,
        pool_size:  int   = 8,
        n_dec_layers: int = 3,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.pool_size    = pool_size
        self.n_dec_layers = n_dec_layers
        num_scales        = len(in_channels)

        # ── 1. Per-scale visual projection ────────────────────────────────────
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, 1, bias=False),
                nn.GroupNorm(min(8, hidden_dim // 8), hidden_dim),
                nn.GELU(),
            ) for c in in_channels
        ])
        self.level_embed = nn.Parameter(torch.randn(num_scales, hidden_dim) * 0.02)

        # ── 2. Text projections ───────────────────────────────────────────────
        proj = lambda: nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.sem_proj  = proj()
        self.attr_proj = proj()
        self.rel_proj  = proj()
        self.sem_vec_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )

        # ── 3. Iterative decoder ──────────────────────────────────────────────
        self.decoder_layers = nn.ModuleList([
            MultiSourceDecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(n_dec_layers)
        ])

        # ── 4. Text-guided attention pooling ─────────────────────────────────
        self.pool_q = nn.Linear(hidden_dim, hidden_dim)
        self.pool_k = nn.Linear(hidden_dim, hidden_dim)

        # ── 5. Box regression head ────────────────────────────────────────────
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # ── FIX: correct box head bias initialisation ─────────────────────────
        # self.box_head[-2] is the Linear(hidden_dim, 4) before Sigmoid.
        #
        # OLD (wrong):
        #   nn.init.constant_(self.box_head[-2].bias, 0.5)
        #   → sigmoid(0.5) ≈ 0.62 for ALL four outputs
        #   → model starts predicting a 62%×62% box at (62%, 62%) — very wrong
        #
        # NEW (correct):
        #   cx, cy bias = 0.0  → sigmoid(0.0) = 0.5  (predicts image centre)
        #   w,  h  bias = -2.0 → sigmoid(-2.0) ≈ 0.12 (predicts small 12% box)
        #
        # A small neutral prior is much better than a large off-centre prior.
        # The GIoU + L1 losses can then efficiently move from this good prior.
        nn.init.constant_(self.box_head[-2].bias[:2], 0.0)   # cx, cy → 0.50
        nn.init.constant_(self.box_head[-2].bias[2:], -2.0)  # w, h   → 0.12

    def forward(
        self,
        feats,
        text_vec,
        text_seq=None,
        text_mask=None,
        attr_seq=None,
        attr_mask=None,
        rel_tokens=None,
    ):
        B = feats[0].shape[0]
        P = self.pool_size

        # ── Step 1: Visual tokens ─────────────────────────────────────────────
        vis_list = []
        for i, (feat, proj) in enumerate(zip(feats, self.scale_projs)):
            t = proj(feat)
            t = F.adaptive_avg_pool2d(t, (P, P))
            t = t.flatten(2).transpose(1, 2)
            t = t + self.level_embed[i]
            vis_list.append(t)
        vis = torch.cat(vis_list, dim=1)

        # ── Step 2: Text projections ──────────────────────────────────────────
        if text_seq is not None:
            sem_kv = self.sem_proj(text_seq)
        else:
            sem_kv = self.sem_vec_proj(text_vec).unsqueeze(1)
        sem_mask_key = (text_mask == 0).bool() if text_mask is not None else None

        if attr_seq is not None:
            attr_kv      = self.attr_proj(attr_seq)
            attr_mask_key = (attr_mask == 0).bool() if attr_mask is not None else None
        else:
            attr_kv       = sem_kv
            attr_mask_key = sem_mask_key

        if rel_tokens is not None:
            rel_kv = self.rel_proj(rel_tokens)
        else:
            rel_kv = self.sem_vec_proj(text_vec).unsqueeze(1)

        # ── Step 3: Iterative decoding ────────────────────────────────────────
        for layer in self.decoder_layers:
            vis = layer(vis, sem_kv, attr_kv, rel_kv,
                        sem_mask=sem_mask_key, attr_mask=attr_mask_key)

        # ── Step 4: Text-guided attention pooling ─────────────────────────────
        q_vec        = self.pool_q(self.sem_vec_proj(text_vec))
        k_mat        = self.pool_k(vis)
        scores       = torch.bmm(k_mat, q_vec.unsqueeze(-1)).squeeze(-1)
        scores       = scores / (self.hidden_dim ** 0.5)
        pool_weights = F.softmax(scores, dim=-1)
        pooled       = torch.bmm(pool_weights.unsqueeze(1), vis).squeeze(1)

        # ── Step 5: Box regression ────────────────────────────────────────────
        pred_box = self.box_head(pooled)

        return pred_box, pool_weights