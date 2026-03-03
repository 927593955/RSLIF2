"""
models/token_router.py

Token-Type Router: The missing piece that makes tri-stream text encoding real.

Core idea:
  After the shared encoder produces contextual embeddings, each token is SOFTLY
  classified into one of three types: [semantic, spatial, attribute].
  Each branch then receives a TOKEN-MASKED view of the sequence — only the tokens
  most relevant to that branch get high attention weights.

This replaces the uniform scalar modulation (w_content, w_pos) in disentangled_attention.py
which applied the SAME weight to ALL tokens regardless of their type.

Design:
  1. TokenTypeClassifier: lightweight 2-layer MLP → 3-class softmax per token
     Outputs token_weights: [B, L, 3] — how much each token belongs to each branch
  2. SoftTokenMask: applies token_weights as attention bias to each branch
     Tokens classified as "not for this branch" get pushed toward -inf
  3. BranchSupervisor: provides direct supervision for ALL three branches
     - Semantic: classification probe (which of 20 DIOR classes)
     - Spatial:  box regression (already exists, now improved)
     - Attribute: attribute presence prediction (size, color, shape, state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Token Type Constants
# ─────────────────────────────────────────────────────────────────────────────
TOKEN_SEMANTIC  = 0
TOKEN_SPATIAL   = 1
TOKEN_ATTRIBUTE = 2
NUM_TOKEN_TYPES = 3


class TokenTypeClassifier(nn.Module):
    """
    Classifies each token in the sequence into [semantic, spatial, attribute].
    
    Architecture: 2-layer MLP with residual connection.
    Input:  hidden_states [B, L, D]
    Output: token_logits  [B, L, 3]  — raw logits (use softmax outside)
            token_weights [B, L, 3]  — softmax probabilities
    
    Why this works:
    - DeBERTa's shared encoder already produces rich contextual embeddings
    - A 2-layer MLP can easily learn "left"/"right" → spatial, "large"/"small" → attribute
    - The soft classification (not hard argmax) allows gradient to flow back to
      the shared encoder, teaching it to produce more separable token representations
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        mid = hidden_size // 4   # small: 256 for deberta-large (1024 hidden)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, NUM_TOKEN_TYPES),
        )
        # Initialize to near-uniform (avoid premature specialization)
        nn.init.normal_(self.classifier[0].weight, 0, 0.01)
        nn.init.zeros_(self.classifier[0].bias)
        nn.init.normal_(self.classifier[3].weight, 0, 0.01)
        nn.init.constant_(self.classifier[3].bias, 0.0)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """
        hidden_states: [B, L, D]
        attention_mask: [B, L]  1=valid, 0=padding
        
        Returns:
            token_weights: [B, L, 3]  soft per-token branch assignments
            token_logits:  [B, L, 3]  raw logits for loss computation
        """
        logits = self.classifier(hidden_states)   # [B, L, 3]
        
        # Mask padding tokens — they should not contribute to any branch
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(-1)   # [B, L, 1]
            logits = logits.masked_fill(pad_mask, -1e4)
        
        weights = F.softmax(logits, dim=-1)   # [B, L, 3]
        return weights, logits


class SoftTokenMaskedPooler(nn.Module):
    """
    Branch-specific pooler that uses token type weights to create a 
    SELECTIVELY ATTENDED sequence vector.
    
    For branch k:
        - Computes standard attention scores over the sequence
        - Multiplies attention scores by token_weights[:, :, k]
        - This pushes tokens of other types toward zero contribution
    
    This is fundamentally different from the current TextAttentionPooler which
    treats ALL tokens equally regardless of their type.
    
    Input:  sequence [B, L, D], token_weights [B, L, 3], branch_idx int
    Output: pooled_vec [B, D]
    """
    
    def __init__(self, hidden_size: int, branch_idx: int):
        super().__init__()
        self.branch_idx = branch_idx
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        # Per-branch learned temperature — let each branch decide how sharp to be
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                token_weights: torch.Tensor):
        """
        x:             [B, L, D]
        mask:          [B, L]  1=valid
        token_weights: [B, L, 3]
        """
        # Standard content attention scores
        scores = self.attn(x).squeeze(-1)   # [B, L]
        
        # Mask padding
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # KEY CHANGE: multiply by branch-specific token routing weights
        branch_weights = token_weights[:, :, self.branch_idx]   # [B, L]
        
        # Apply temperature-scaled routing
        temp = self.temperature.clamp(min=0.1, max=10.0)
        # Shift scores by log of routing weight (additive in log-space = multiplicative)
        # This is equivalent to: softmax(score + log(branch_weight + eps))
        routing_bias = torch.log(branch_weights + 1e-8) * temp
        scores = scores + routing_bias
        
        # Stability
        scores_max = scores.max(dim=1, keepdim=True)[0].detach()
        scores = scores - scores_max
        scores = torch.clamp(scores, min=-50, max=50)
        
        attn_weights = F.softmax(scores, dim=-1)   # [B, L]
        
        if torch.isnan(attn_weights).any():
            attn_weights = (mask.float() / mask.float().sum(-1, keepdim=True).clamp(min=1))
        
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)   # [B, D]
        return pooled, attn_weights


class TokenTypeSupervisionLoss(nn.Module):
    """
    Provides direct supervision for ALL three branches using pseudo-labels
    derived from the input text itself (no external annotation needed).
    
    Supervision sources:
    
    SEMANTIC branch:
        - Class ID from YOLO target (which of 20 DIOR categories)
        - Probe: linear layer on sem_vec → 20-class cross-entropy
        - This forces sem_vec to encode WHAT the object is
    
    SPATIAL branch:
        - GT box coordinates (already exists as loss_spa)
        - NEW: spatial token presence signal — if text contains spatial words,
          spatial branch should produce high-variance routing weights for those tokens
        - Probe: linear layer on spa_vec → box regression (smooth L1)
    
    ATTRIBUTE branch:
        - Attribute gate: does this description contain size/color/shape words?
          Computed heuristically from input text as pseudo-label
        - Probe: attribute richness score per text
        - This forces attr_vec to encode HOW the object looks
    
    Token routing regularization:
        - Entropy loss: encourage routing weights to be SHARP (not uniform)
          Sharp routing = tokens actually being classified, not ignored
        - Diversity loss: different branches should get different token subsets
          Prevents collapse where all tokens go to one branch
    """
    
    # Words that strongly indicate spatial content
    SPATIAL_WORDS = {
        'left', 'right', 'top', 'bottom', 'above', 'below', 'upper', 'lower',
        'center', 'middle', 'beside', 'near', 'far', 'between', 'adjacent',
        'north', 'south', 'east', 'west', 'next', 'corner', 'edge', 'side',
        'cx', 'cy',  # spatial hint tokens
    }
    
    # Words that strongly indicate attribute content
    ATTRIBUTE_WORDS = {
        'large', 'small', 'tiny', 'big', 'long', 'short', 'wide', 'narrow',
        'red', 'white', 'black', 'dark', 'bright', 'gray', 'blue', 'green',
        'circular', 'rectangular', 'elongated', 'square', 'round',
        'parked', 'moving', 'docked', 'empty', 'full',
        'single', 'multiple', 'several',
        'metallic', 'concrete', 'grass', 'water',
    }
    
    def __init__(self, hidden_dim: int, num_classes: int = 20):
        super().__init__()
        self.num_classes = num_classes
        
        # Semantic probe: sem_vec → class ID
        self.sem_probe = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Attribute probe: attr_vec → attribute richness score [0, 1]
        self.attr_probe = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize probes with small weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def compute_attribute_pseudo_labels(self, input_ids, tokenizer):
        """
        Heuristic: decode tokens, check if any attribute words are present.
        Returns: [B] float tensor, 1.0 if description has attributes, 0.0 otherwise.
        This is a pseudo-label — no external annotation needed.
        """
        batch_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        labels = []
        for text in batch_texts:
            words = set(text.lower().split())
            has_attr = bool(words & self.ATTRIBUTE_WORDS)
            labels.append(1.0 if has_attr else 0.0)
        return torch.tensor(labels, dtype=torch.float32, device=input_ids.device)
    
    def routing_entropy_loss(self, token_weights: torch.Tensor,
                             attention_mask: torch.Tensor):
        """
        Minimize per-token entropy of routing weights.
        High entropy = token doesn't know which branch it belongs to.
        Low entropy = token is clearly classified → branches see different tokens.
        
        token_weights: [B, L, 3]
        """
        # Entropy: -sum(p * log(p)) per token, per sample
        eps = 1e-8
        entropy = -(token_weights * (token_weights + eps).log()).sum(-1)   # [B, L]
        
        # Only count valid (non-padding) tokens
        if attention_mask is not None:
            entropy = entropy * attention_mask.float()
            n_valid = attention_mask.float().sum(-1).clamp(min=1)
            entropy = entropy.sum(-1) / n_valid   # [B]
        else:
            entropy = entropy.mean(-1)
        
        # We want LOW entropy → minimize mean entropy
        return entropy.mean()
    
    def routing_diversity_loss(self, token_weights: torch.Tensor,
                               attention_mask: torch.Tensor):
        """
        Maximize difference between branch routing distributions.
        If all branches get the same tokens → branches can't specialize.
        
        We want the three branch distributions (column vectors of token_weights)
        to be DIFFERENT from each other.
        
        Measure: KL divergence between branch distributions should be HIGH.
        Proxy: minimize correlation between branch assignment vectors.
        """
        if attention_mask is not None:
            mask_f = attention_mask.float().unsqueeze(-1)   # [B, L, 1]
            # Masked average routing per branch: [B, 3]
            branch_dist = (token_weights * mask_f).sum(1)   # [B, 3]
            branch_dist = branch_dist / mask_f.sum(1).clamp(min=1)
        else:
            branch_dist = token_weights.mean(1)   # [B, 3]
        
        # Branch distributions should be peaked on different branches
        # Ideal: sem_branch sees mostly type-0, spa sees type-1, attr sees type-2
        # Measure spread: we want branch_dist to be close to identity-like
        # Simple proxy: penalize when any two branches have similar distributions
        
        sem_dist  = token_weights[:, :, 0]   # [B, L]
        spa_dist  = token_weights[:, :, 1]   # [B, L]
        attr_dist = token_weights[:, :, 2]   # [B, L]
        
        # Cosine similarity between branch routing vectors (want LOW similarity)
        def branch_cos_sim(a, b):
            a_n = F.normalize(a, dim=-1, eps=1e-8)
            b_n = F.normalize(b, dim=-1, eps=1e-8)
            return (a_n * b_n).sum(-1).mean()   # scalar
        
        overlap_01 = branch_cos_sim(sem_dist, spa_dist)
        overlap_02 = branch_cos_sim(sem_dist, attr_dist)
        overlap_12 = branch_cos_sim(spa_dist, attr_dist)
        
        # Penalize high overlap — we want branches to see different tokens
        diversity_loss = overlap_01 + overlap_02 + overlap_12
        return diversity_loss
    
    def forward(
        self,
        sem_vec:       torch.Tensor,    # [B, D]
        attr_vec:      torch.Tensor,    # [B, D]
        token_weights: torch.Tensor,    # [B, L, 3]
        attention_mask: torch.Tensor,   # [B, L]
        class_ids:     torch.Tensor,    # [B] long — from YOLO targets
        attr_labels:   torch.Tensor,    # [B] float — pseudo-labels
    ):
        """
        Returns dict of individual loss components (all differentiable).
        """
        losses = {}
        
        # ── Semantic classification probe ─────────────────────────────────
        # Forces sem_vec to encode class identity
        sem_logits = self.sem_probe(sem_vec)           # [B, num_classes]
        valid = (class_ids >= 0) & (class_ids < self.num_classes)
        if valid.any():
            losses['sem_cls'] = F.cross_entropy(
                sem_logits[valid], class_ids[valid].long()
            )
        else:
            losses['sem_cls'] = torch.tensor(0.0, device=sem_vec.device)
        
        # ── Attribute richness probe ──────────────────────────────────────
        # Forces attr_vec to encode PRESENCE of attribute words
        attr_pred = self.attr_probe(attr_vec).squeeze(-1)   # [B]
        losses['attr_presence'] = F.binary_cross_entropy(attr_pred, attr_labels)
        
        # ── Token routing losses ──────────────────────────────────────────
        losses['routing_entropy'] = self.routing_entropy_loss(
            token_weights, attention_mask
        )
        losses['routing_diversity'] = self.routing_diversity_loss(
            token_weights, attention_mask
        )
        
        return losses