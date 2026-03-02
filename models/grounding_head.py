"""
models/grounding_head.py  (v3 - multi-source iterative decoder)

Root-cause analysis of the four observed failure modes:

Issue 1  ── Accuracy plateau / regression at epoch ~40
  Root cause: Single-pass cross-attention saturates early because the visual
  token pool (192 tokens, avg-pooled) provides no mechanism for inter-token
  comparison.  After the grounding loss has pulled the easy cases in,
  the remaining hard cases require comparative spatial reasoning that a
  single cross-attention layer cannot express.  The abrupt introduction
  of L_orth (λ=2.0) at epoch 10 also destabilises the optimiser.
  Fix: Iterative decoder (N_dec=3 transformer layers) with visual self-attention
  in every layer.  Each pass refines the previous estimate, so gradients keep
  flowing after the easy basin is exhausted.  The gated auxiliary streams
  (attr / relation) are initialised to zero contribution and learn gradually.

Issue 2  ── Bounding-box offset (correctly-classified IoU ~0.7)
  Root cause: Predicting absolute (cx,cy,w,h) from a single softmax-pooled
  vector forces the model to encode both "where to look" and "how big the box
  is" in the same representation.  The Sigmoid output saturates near 0/1,
  biasing small-object predictions toward the centre of the image.
  Fix: CIoU loss explicitly penalises centre distance (rho²/c²) and aspect-ratio
  mismatch (alpha·v) rather than just overlap area.  The iterative decoder
  lets the model progressively refine the centroid before committing to the
  final scale, reducing systematic offset.

Issue 3  ── Attribute queries fail ("the longest bridge")
  Root cause: vlm_grounding.py only passes sem_seq/sem_vec to the grounding
  head; the attr branch (TriStreamDeBERTa.attribute_encoder) outputs are used
  only inside FusionNeck via ScaleAwareFiLM, which modulates feature maps but
  cannot perform token-level comparative attention ("longest bridge" requires
  comparing bridge tokens across the spatial extent of the image).
  Fix: Dedicated attribute cross-attention sublayer in every decoder layer,
  fed with the attr branch token sequence from the text encoder.  An attention
  gate (sigmoid, initialised to 0) lets the model learn when to activate it.

Issue 4  ── Relational queries fail ("B on the right side of A")
  Root cause: StructuredPositionPointerEncoder already produces relation_tokens
  [B, K, D] encoding pairwise spatial relations, but vlm_grounding.py never
  passes these to the grounding head — they are silently discarded.
  Fix: Dedicated relation cross-attention sublayer fed with relation_tokens,
  again gated and initialised to zero contribution.

SOTA references:
  TransVG (ICCV 2021) — iterative visual-linguistic decoder for grounding
  MDETR  (ICCV 2021) — modulated cross-attention on multi-source text
  SimVG  (NeurIPS 2024) — simplified iterative visual grounding
  UNINEXT (CVPR 2023) — universal prompts for instance perception
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Single decoder layer  (visual self-attn + 3 cross-attn streams + FFN)
# ─────────────────────────────────────────────────────────────────────────────

class MultiSourceDecoderLayer(nn.Module):
    """
    One decoder layer with four attention sublayers:
      1. Visual self-attention  → inter-token comparison ("which is LONGEST")
      2. Semantic cross-attn   → category grounding ("bridge", "airplane")
      3. Attribute cross-attn  → discriminative attribute ("red", "largest")
      4. Relation cross-attn   → spatial relation ("right of A", "near B")
      5. FFN

    Attribute and relation streams are initialised with zero contribution
    (sigmoid gate starts at 0.0) so the model falls back to semantic-only
    grounding on early epochs and gradually opens the auxiliary streams.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        mha = lambda: nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        norm = lambda: nn.LayerNorm(d_model)

        # 1. Visual self-attention
        self.self_attn = mha();  self.norm1 = norm()
        # 2. Semantic cross-attention
        self.sem_attn  = mha();  self.norm2 = norm()
        # 3. Attribute cross-attention
        self.attr_attn = mha();  self.norm3 = norm()
        # 4. Relation cross-attention
        self.rel_attn  = mha();  self.norm4 = norm()
        # 5. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm5 = norm()

        # Learned gates — init=0 → sigmoid(0)=0.5; but we explicitly set to
        # a large negative so the gate starts near 0 contribution.
        self.attr_gate = nn.Parameter(torch.full((1,), -4.0))
        self.rel_gate  = nn.Parameter(torch.full((1,), -4.0))

    def forward(self, vis, sem_kv, attr_kv, rel_kv,
                sem_mask=None, attr_mask=None):
        """
        vis     : [B, N, D]
        *_kv    : [B, L*, D]  key/value sequences for each stream
        *_mask  : [B, L*]  True = ignore (key_padding_mask convention)
        """
        # 1. Visual self-attention
        v2, _ = self.self_attn(vis, vis, vis)
        vis = self.norm1(vis + v2)

        # 2. Semantic cross-attention
        s2, _ = self.sem_attn(vis, sem_kv, sem_kv, key_padding_mask=sem_mask)
        vis = self.norm2(vis + s2)

        # 3. Attribute cross-attention (gated)
        a2, _ = self.attr_attn(vis, attr_kv, attr_kv, key_padding_mask=attr_mask)
        vis = self.norm3(vis + torch.sigmoid(self.attr_gate) * a2)

        # 4. Relation cross-attention (gated; rel_kv is short — usually 3–4 tokens)
        r2, _ = self.rel_attn(vis, rel_kv, rel_kv)
        vis = self.norm4(vis + torch.sigmoid(self.rel_gate) * r2)

        # 5. FFN
        vis = self.norm5(vis + self.ffn(vis))
        return vis


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Main grounding head
# ─────────────────────────────────────────────────────────────────────────────

class GroundingHead(nn.Module):
    """
    Multi-source iterative grounding head.

    Args:
        in_channels  : [C3, C4, C5] visual feature channels
        text_dim     : text encoder hidden dim (e.g. 768 for DeBERTa-v3-small)
        hidden_dim   : internal working dimension (default 256)
        num_heads    : number of attention heads
        dropout      : dropout probability
        pool_size    : spatial pool size per scale (default 8 → 3×64=192 tokens)
        n_dec_layers : number of iterative decoder layers (default 3)
    """

    def __init__(
        self,
        in_channels,
        text_dim,
        hidden_dim: int = 256,
        num_heads:  int = 8,
        dropout:    float = 0.1,
        pool_size:  int = 8,
        n_dec_layers: int = 3,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.pool_size    = pool_size
        self.n_dec_layers = n_dec_layers
        num_scales        = len(in_channels)

        # ── 1. Per-scale visual projection ───────────────────────────────────
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, 1, bias=False),
                nn.GroupNorm(min(8, hidden_dim // 8), hidden_dim),
                nn.GELU(),
            ) for c in in_channels
        ])
        self.level_embed = nn.Parameter(torch.randn(num_scales, hidden_dim) * 0.02)

        # ── 2. Text projections (one per stream) ──────────────────────────────
        proj = lambda: nn.Sequential(nn.Linear(text_dim, hidden_dim),
                                     nn.LayerNorm(hidden_dim))
        self.sem_proj  = proj()   # semantic sequence
        self.attr_proj = proj()   # attribute sequence
        self.rel_proj  = proj()   # relation pointer tokens

        # Global vector for pooling query (semantic branch)
        self.sem_vec_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )

        # ── 3. Iterative multi-source decoder ─────────────────────────────────
        self.decoder_layers = nn.ModuleList([
            MultiSourceDecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(n_dec_layers)
        ])

        # ── 4. Text-guided attention pooling ──────────────────────────────────
        self.pool_q = nn.Linear(hidden_dim, hidden_dim)
        self.pool_k = nn.Linear(hidden_dim, hidden_dim)

        # ── 5. Box regression head ────────────────────────────────────────────
        # Predicts normalised (cx, cy, w, h) ∈ [0,1]
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Start predicting the centre of the image; the model quickly learns
        # to deviate from this prior.
        nn.init.constant_(self.box_head[-2].bias, 0.5)

    # ── Forward ───────────────────────────────────────────────────────────────

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
        """
        feats      : [P3, P4, P5]  visual feature maps
        text_vec   : [B, text_dim]  semantic global vector
        text_seq   : [B, L, text_dim]  semantic token sequence
        text_mask  : [B, L]  1=valid token
        attr_seq   : [B, L, text_dim]  attribute token sequence
        attr_mask  : [B, L]  1=valid token
        rel_tokens : [B, K, text_dim]  relation pointer tokens (K≈3–4)

        Returns:
            pred_box   : [B, 4]  normalised (cx, cy, w, h) in [0, 1]
            pool_weights: [B, N] attention weights for visualisation
        """
        B = feats[0].shape[0]
        P = self.pool_size

        # ── Step 1: Visual tokens ─────────────────────────────────────────────
        vis_list = []
        for i, (feat, proj) in enumerate(zip(feats, self.scale_projs)):
            t = proj(feat)                            # [B, D, H, W]
            t = F.adaptive_avg_pool2d(t, (P, P))     # [B, D, P, P]
            t = t.flatten(2).transpose(1, 2)          # [B, P², D]
            t = t + self.level_embed[i]               # scale identity
            vis_list.append(t)
        vis = torch.cat(vis_list, dim=1)              # [B, 3P², D]

        # ── Step 2: Text projections ──────────────────────────────────────────
        # Semantic stream
        if text_seq is not None:
            sem_kv = self.sem_proj(text_seq)           # [B, L, D]
        else:
            sem_kv = self.sem_vec_proj(text_vec).unsqueeze(1)  # [B, 1, D]
        sem_mask_key = (text_mask == 0).bool() if text_mask is not None else None

        # Attribute stream (fall back to semantic if not provided)
        if attr_seq is not None:
            attr_kv = self.attr_proj(attr_seq)         # [B, L, D]
            attr_mask_key = (attr_mask == 0).bool() if attr_mask is not None else None
        else:
            attr_kv       = sem_kv
            attr_mask_key = sem_mask_key

        # Relation stream (fall back to singleton semantic vec if not provided)
        if rel_tokens is not None:
            # rel_tokens may come from StructuredPositionPointerEncoder:
            # shape [B, K, text_dim] (text_dim from spa branch)
            rel_kv = self.rel_proj(rel_tokens)         # [B, K, D]
        else:
            rel_kv = self.sem_vec_proj(text_vec).unsqueeze(1)  # [B, 1, D]

        # ── Step 3: Iterative decoding ────────────────────────────────────────
        for layer in self.decoder_layers:
            vis = layer(
                vis, sem_kv, attr_kv, rel_kv,
                sem_mask=sem_mask_key,
                attr_mask=attr_mask_key,
            )

        # ── Step 4: Text-guided attention pooling ─────────────────────────────
        q_vec       = self.pool_q(self.sem_vec_proj(text_vec))  # [B, D]
        k_mat       = self.pool_k(vis)                           # [B, N, D]
        scores      = torch.bmm(k_mat, q_vec.unsqueeze(-1)).squeeze(-1)
        scores      = scores / (self.hidden_dim ** 0.5)          # [B, N]
        pool_weights = F.softmax(scores, dim=-1)
        pooled      = torch.bmm(pool_weights.unsqueeze(1), vis).squeeze(1)  # [B, D]

        # ── Step 5: Box regression ────────────────────────────────────────────
        pred_box = self.box_head(pooled)    # [B, 4]  sigmoid → [0, 1]

        return pred_box, pool_weights