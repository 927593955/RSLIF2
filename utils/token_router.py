"""
models/token_router.py  (v2)

Changes vs v1:
  - REMOVED: routing_entropy_loss, routing_diversity_loss
    These had no target and caused router collapse to uniform weights.
  - ADDED: router_vocab_loss() — simple CE against TokenLabeler pseudo-labels.
    This is the only supervision the router needs.
  - FIXED: SoftTokenMaskedPooler routing bias now uses log(3 * weight) instead
    of log(weight). At uniform (0.33), bias = log(1.0) = 0. At high
    confidence (0.9), bias = log(2.7) ≈ 1.0. At low (0.05), bias ≈ -1.9.
    Previously log(0.33) = -1.1 for all branches → identical poolers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LABEL_SEMANTIC  = 0
LABEL_SPATIAL   = 1
LABEL_ATTRIBUTE = 2
LABEL_IGNORE    = -1
NUM_TOKEN_TYPES = 3


class TokenTypeClassifier(nn.Module):
    """
    Lightweight per-token classifier: hidden_state → [sem, spa, attr] logits.
    Supervised by cross-entropy against TokenLabeler pseudo-labels.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        mid = hidden_size // 4
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, NUM_TOKEN_TYPES),
        )
        # Initialize to near-uniform — vocabulary supervision will pull it apart
        nn.init.normal_(self.net[0].weight, 0, 0.01)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[3].weight, 0, 0.01)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        """
        hidden: [B, L, D]
        mask:   [B, L]
        Returns:
            weights: [B, L, 3]  softmax probabilities
            logits:  [B, L, 3]  raw (for CE loss)
        """
        logits = self.net(hidden)                        # [B, L, 3]
        if mask is not None:
            logits = logits.masked_fill(
                (mask == 0).unsqueeze(-1), -1e4
            )
        weights = F.softmax(logits, dim=-1)
        return weights, logits

    def vocab_loss(self, logits: torch.Tensor,
                   token_type_labels: torch.Tensor) -> torch.Tensor:
        """
        Direct CE supervision from TokenLabeler pseudo-labels.

        logits:             [B, L, 3]
        token_type_labels:  [B, L]  int64 with -1 = ignore

        ignore_index=-1 excludes special tokens and padding automatically.
        """
        B, L, C = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * L, C),
            token_type_labels.reshape(B * L),
            ignore_index=LABEL_IGNORE,
        )
        return loss


class SoftTokenMaskedPooler(nn.Module):
    """
    Attention pooler that biases toward tokens of a specific type.

    FIX vs v1: routing bias is now log(NUM_TOKEN_TYPES * weight) instead of
    log(weight). This centers the bias at zero for uniform weights and creates
    a proper relative log-odds contribution.
    
    At uniform weight (1/3): bias = log(3 * 1/3) = log(1) = 0  → no effect
    At high confidence  (0.9): bias = log(2.7) ≈ +1.0          → boosts token
    At low confidence  (0.05): bias = log(0.15) ≈ -1.9          → suppresses
    """

    def __init__(self, hidden_size: int, branch_idx: int):
        super().__init__()
        self.branch_idx = branch_idx
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        # Learned scale for how much to trust the routing weights
        # Initialized to 1.0; grows as router gains confidence
        self.routing_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                token_weights: torch.Tensor):
        """
        x:             [B, L, D]
        mask:          [B, L]
        token_weights: [B, L, 3]

        Returns: pooled [B, D], attn_weights [B, L]
        """
        scores = self.attn(x).squeeze(-1)               # [B, L]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        # Routing bias: centered log-odds relative to uniform
        w = token_weights[:, :, self.branch_idx].clamp(1e-8, 1.0)
        routing_bias = torch.log(NUM_TOKEN_TYPES * w)   # [B, L]
        scale = self.routing_scale.clamp(0.0, 5.0)
        scores = scores + scale * routing_bias

        # Numerical stability
        scores = scores - scores.max(dim=1, keepdim=True)[0].detach()
        scores = scores.clamp(-50, 50)
        attn_w = F.softmax(scores, dim=-1)              # [B, L]

        pooled = (x * attn_w.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return pooled, attn_w