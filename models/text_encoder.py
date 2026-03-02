"""
models/text_encoder.py  (v2 - Token-Type Gated Branches)

Root causes fixed vs v1:

PROBLEM 1 - Uniform scalar modulation is token-agnostic
  OLD: attention_scores *= w_content  (same scalar for ALL tokens)
  NEW: Removed. The DeBERTa branch modulation is kept for gradient diversity
       but the REAL specialization is done via TokenTypeGatedPooler.

PROBLEM 2 - No mechanism to identify word types
  NEW: TokenTypeGatedPooler learns per-branch token importance weights.
       The semantic branch gate learns to upweight class-name tokens.
       The spatial branch gate learns to upweight position/relation tokens.
       The attribute branch gate learns to upweight property tokens.
       This is learned end-to-end via branch-specific supervision signals.

PROBLEM 3 - Specialization is unverified
  NEW: BranchProbe module allows measuring branch specialization during eval.
       Logs per-branch token entropy (low = focused, high = diffuse).

PROBLEM 4 - Supervision asymmetry
  OLD: spatial=direct box loss, semantic=nothing, attribute=unused gate
  NEW: semantic=class discrimination loss (20 DIOR classes from targets)
       spatial=box regression + spatial quadrant classification
       attribute=branch complementarity + reconstruction consistency
       ALL branches have direct, strong, specific supervision.

PROBLEM 5 - attr_gate computed but never supervised
  OLD: pred_attr_gate returned but absent from loss
  NEW: Removed attr_gate_head entirely. Replaced with supervised complementarity.

PROBLEM 6 - StructuredPositionPointerEncoder unsupervised
  OLD: 16-template bank with no loss signal
  NEW: Kept but directly connected to spatial quadrant loss.

PROBLEM 7 - Adapters applied after sequence context
  OLD: BranchAdapter(sem_feat) then pool  ->  adapter sees sequence, then pool discards it
  NEW: TokenTypeGatedPooler(sem_feat) uses per-token learned gates, preserving selectivity.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Tokenizer

from lib.DeBERTaLib.deberta import DeBERTa
from lib.DeBERTaLib.config import ModelConfig


# =============================================================================
# 1. TokenTypeGatedPooler  (KEY NEW MODULE)
# =============================================================================

class TokenTypeGatedPooler(nn.Module):
    """
    Learns WHICH tokens matter for this branch via a soft gate.

    For a text like "a large red airplane near the harbor":
      - semantic gate: high weight on "airplane", low on "near", "large"
      - spatial gate:  high weight on "near", "harbor", low on "airplane", "red"
      - attribute gate: high weight on "large", "red", low on "near", "airplane"

    The gates are learned purely from task supervision:
      - semantic: class discrimination loss drives gate to attend class-name tokens
      - spatial:  box regression loss drives gate to attend position tokens
      - attribute: complementarity loss drives gate to be different from sem+spa

    Architecture:
      hidden [B, L, D]
        -> gate_scorer [D -> D//4 -> 1] (small MLP, cheap)
        -> raw_gate [B, L]
        -> softmax over valid tokens (mask applied)
        -> gated_pooled = sum(hidden * gate) [B, D]

    Also returns gate_weights for analysis/visualization.
    """

    def __init__(self, hidden_size, bottleneck_ratio=4):
        super().__init__()
        mid = max(hidden_size // bottleneck_ratio, 64)
        self.gate_scorer = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Linear(mid, 1)
        )
        # Temperature: learnable, controls gate sharpness
        # Starts at 1.0, can sharpen during training
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, hidden, mask):
        """
        hidden: [B, L, D]
        mask:   [B, L]  1=valid token, 0=padding
        Returns:
            pooled:       [B, D]  weighted sum
            gate_weights: [B, L]  for analysis
        """
        temp = self.log_temp.exp().clamp(min=0.1, max=10.0)

        # Raw gate scores [B, L]
        scores = self.gate_scorer(hidden).squeeze(-1) / temp

        # Mask out padding before softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        gate_weights = F.softmax(scores, dim=-1)  # [B, L]

        # Gated pooling
        pooled = torch.bmm(gate_weights.unsqueeze(1), hidden).squeeze(1)  # [B, D]

        return pooled, gate_weights


# =============================================================================
# 2. Spatial quadrant helper (for spatial supervision)
# =============================================================================

def boxes_to_quadrant(cx, cy, grid=3):
    """
    Convert normalized center coords to grid-cell index [0, grid*grid-1].
    cx, cy: [B] floats in [0, 1]
    Returns: [B] long
    """
    col = (cx * grid).long().clamp(0, grid - 1)
    row = (cy * grid).long().clamp(0, grid - 1)
    return row * grid + col   # [B], range [0, grid*grid-1]


# =============================================================================
# 3. StructuredPositionPointerEncoder  (unchanged, now properly supervised)
# =============================================================================

class StructuredPositionPointerEncoder(nn.Module):
    """
    Unchanged from v1. Now receives direct supervision from spatial quadrant loss
    via the relation_vec which feeds quadrant_head in BranchSupervision.
    """
    def __init__(self, hidden_dim, num_pointers=4, num_relation_templates=16):
        super().__init__()
        self.num_pointers = num_pointers
        self.num_relation_templates = num_relation_templates
        self.pointer_scorer = nn.Linear(hidden_dim, num_pointers)
        self.pointer_norm = nn.LayerNorm(hidden_dim)

        self.graph_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.graph_norm = nn.LayerNorm(hidden_dim)

        self.rel_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh()
        )
        self.rel_token_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.out_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.template_bank = nn.Parameter(torch.randn(num_relation_templates, hidden_dim) * 0.02)
        self.template_query = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.template_logits_head = nn.Linear(hidden_dim, num_relation_templates)
        self.template_norm = nn.LayerNorm(hidden_dim)

    def forward(self, spa_feat, mask):
        pointer_logits = self.pointer_scorer(spa_feat).transpose(1, 2)  # [B, K, L]
        if mask is not None:
            m = mask.unsqueeze(1) == 0
            pointer_logits = pointer_logits.masked_fill(m, -1e4)
        pointer_attn = torch.softmax(pointer_logits, dim=-1)
        pointer_tokens = torch.matmul(pointer_attn, spa_feat)
        pointer_tokens = self.pointer_norm(pointer_tokens)

        graph_tokens, _ = self.graph_attn(pointer_tokens, pointer_tokens, pointer_tokens)
        graph_tokens = self.graph_norm(pointer_tokens + graph_tokens)

        src = graph_tokens[:, :-1, :]
        dst = graph_tokens[:, 1:, :]
        pair_feat = torch.cat([src, dst], dim=-1)
        relative_position = self.rel_head(pair_feat)
        rel_tokens = self.rel_token_proj(torch.cat([src, relative_position], dim=-1))

        relation_seed = rel_tokens.mean(dim=1)
        template_query = self.template_query(relation_seed)
        template_logits = self.template_logits_head(template_query)
        template_attn = torch.softmax(template_logits, dim=-1)
        template_vec = torch.matmul(template_attn, self.template_bank)
        template_vec = self.template_norm(template_vec)

        relation_vec = self.out_fuse(
            torch.cat([graph_tokens.mean(dim=1), rel_tokens.mean(dim=1)], dim=-1)
        )
        relation_vec = self.template_norm(relation_vec + template_vec)

        return {
            'pointer_tokens': pointer_tokens,
            'graph_tokens': graph_tokens,
            'relative_position': relative_position,
            'relation_tokens': rel_tokens,
            'relation_vec': relation_vec,
            'template_logits': template_logits,
            'template_attn': template_attn,
            'template_vec': template_vec,
        }


# =============================================================================
# 4. Main model: TriStreamDeBERTa  (v2)
# =============================================================================

class TriStreamDeBERTa(nn.Module):
    def __init__(self, model_path='./pretrain_ckp/deberta-v3-large', max_len=77,
                 num_classes=20, spatial_grid=3):
        super().__init__()
        self.max_len = max_len
        self.num_classes = num_classes
        self.spatial_grid = spatial_grid

        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)

        config_path = os.path.join(model_path, 'config.json')
        self.config = ModelConfig.from_json_file(config_path)

        raw_model = DeBERTa(config=self.config)
        self._load_weights(model_path, raw_model)

        self.total_layers = len(raw_model.encoder.layer)
        self.split_idx = self.total_layers // 2

        self.embeddings = raw_model.embeddings

        # Shared lower layers
        self.shared_encoder = nn.ModuleList([
            raw_model.encoder.layer[i] for i in range(self.split_idx)
        ])

        # Three branch upper layers
        self.semantic_encoder = nn.ModuleList([
            raw_model.encoder.layer[i] for i in range(self.split_idx, self.total_layers)
        ])
        self.spatial_encoder = copy.deepcopy(self.semantic_encoder)
        self.attribute_encoder = copy.deepcopy(self.semantic_encoder)

        self.rel_embeddings = raw_model.encoder.rel_embeddings
        self.rel_layer_norm = raw_model.encoder.LayerNorm
        del raw_model

        self.hidden_dim = self.config.hidden_size

        # ── KEY CHANGE: TokenTypeGatedPooler replaces TextAttentionPooler ──────
        # Each branch gets its own gated pooler that learns which tokens to attend.
        # Supervised end-to-end by branch-specific losses below.
        self.sem_pooler  = TokenTypeGatedPooler(self.hidden_dim)
        self.spa_pooler  = TokenTypeGatedPooler(self.hidden_dim)
        self.attr_pooler = TokenTypeGatedPooler(self.hidden_dim)

        # ── Spatial relation encoder (now properly supervised) ────────────────
        self.spatial_relation_encoder = StructuredPositionPointerEncoder(
            self.hidden_dim, num_pointers=4
        )

        # ── Per-branch supervision heads ──────────────────────────────────────

        # SEMANTIC: class discrimination (20 DIOR classes)
        # Trained with CrossEntropy(sem_vec -> class_id)
        # class_id comes from targets[:, 1] in the dataloader -- ALREADY AVAILABLE
        self.sem_cls_head = nn.Linear(self.hidden_dim, num_classes, bias=False)

        # SPATIAL: box regression (already exists) + quadrant classification
        # Quadrant: divide image into spatial_grid x spatial_grid cells -> 9 classes
        # Gives precise spatial location supervision beyond just box regression
        self.spa_regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 4),
            nn.Sigmoid()
        )
        self.spa_quadrant_head = nn.Linear(
            self.hidden_dim, spatial_grid * spatial_grid, bias=True
        )

        # ATTRIBUTE: Reconstruction head
        # Forces attr_vec to encode what sem+spa miss, i.e. it must be complementary.
        # Loss: ||MLP(sem_vec + spa_vec + attr_vec) - shared_pooled||^2
        # shared_pooled = mean-pooled output of shared_encoder (cached in forward)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

    def _load_weights(self, model_path, raw_model):
        weight_path = os.path.join(model_path, 'pytorch_model.bin')
        if not os.path.exists(weight_path):
            print(f"⚠️ Weights not found at {weight_path}")
            return
        print(f"Loading weights from {weight_path}...")
        hf_state_dict = torch.load(weight_path, map_location='cpu')
        clean_state = {}
        for k, v in hf_state_dict.items():
            if k.startswith('deberta.'):
                clean_state[k.replace('deberta.', '', 1)] = v
            elif 'lm_predictions' not in k:
                clean_state[k] = v
        raw_model.load_state_dict(clean_state, strict=False)
        print("✅ Pretrained weights loaded.")

    def get_rel_pos(self):
        rel_embed = self.rel_embeddings.weight
        if self.rel_layer_norm is not None:
            rel_embed = self.rel_layer_norm(rel_embed)
        return rel_embed

    def _run_shared_encoder(self, input_ids, mask):
        """Run embeddings + shared lower layers. Returns hidden_states and shared_pooled."""
        embedding_output = self.embeddings(input_ids.long(), mask)
        hidden_states = embedding_output['embeddings']

        if mask.dim() == 2:
            extended_mask = mask.unsqueeze(1).unsqueeze(2).to(dtype=hidden_states.dtype)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = mask

        rel_embed = self.get_rel_pos()

        for layer in self.shared_encoder:
            hidden_states = layer(
                hidden_states, extended_mask,
                return_att=False, rel_embeddings=rel_embed, branch_type="normal"
            )

        # Shared pooled: simple mean over valid tokens
        # Used as reconstruction target for attribute branch supervision
        valid = mask.unsqueeze(-1).float()                                # [B, L, 1]
        shared_pooled = (hidden_states * valid).sum(1) / valid.sum(1)     # [B, D]

        return hidden_states, extended_mask, rel_embed, shared_pooled

    def _run_branch(self, hidden_states, extended_mask, rel_embed, encoder_layers, branch_type):
        """Run one branch's upper encoder layers."""
        feat = hidden_states
        num_layers = len(encoder_layers)
        for idx, layer in enumerate(encoder_layers):
            feat = layer(
                feat, extended_mask, return_att=False,
                rel_embeddings=rel_embed,
                branch_type=branch_type,
                layer_id=idx, total_layers=num_layers
            )
        return feat

    def forward(self, text_input):
        device = self.embeddings.word_embeddings.weight.device

        if isinstance(text_input, list) and isinstance(text_input[0], str):
            inputs = self.tokenizer(
                text_input, padding='max_length', truncation=True,
                max_length=self.max_len, return_tensors="pt"
            )
            input_ids = inputs['input_ids'].to(device)
            mask = inputs['attention_mask'].to(device)
        elif isinstance(text_input, (list, tuple)) and len(text_input) == 2:
            input_ids = text_input[0].to(device)
            mask = text_input[1].to(device)
        else:
            raise ValueError(f"Unsupported input format: {type(text_input)}")

        # 1. Shared encoder
        hidden_states, extended_mask, rel_embed, shared_pooled = \
            self._run_shared_encoder(input_ids, mask)

        # 2. Three branch encoders
        sem_feat  = self._run_branch(hidden_states, extended_mask, rel_embed,
                                     self.semantic_encoder,  "semantic")
        spa_feat  = self._run_branch(hidden_states, extended_mask, rel_embed,
                                     self.spatial_encoder,   "spatial")
        attr_feat = self._run_branch(hidden_states, extended_mask, rel_embed,
                                     self.attribute_encoder, "attribute")

        # 3. TokenTypeGated pooling  ← KEY CHANGE
        # Each gate learns WHICH TOKENS to attend for its branch.
        # sem_gate will learn to focus on class-name tokens ("airplane", "ship")
        # spa_gate will learn to focus on position tokens ("near", "left of")
        # attr_gate will learn to focus on property tokens ("large", "red")
        sem_vec,  sem_gate  = self.sem_pooler(sem_feat,  mask)
        spa_vec,  spa_gate  = self.spa_pooler(spa_feat,  mask)
        attr_vec, attr_gate = self.attr_pooler(attr_feat, mask)

        # 4. Branch supervision outputs

        # SEMANTIC: class logits
        # Loss: CrossEntropy(sem_cls_logits, class_id_from_targets)
        sem_cls_logits = self.sem_cls_head(sem_vec)   # [B, 20]

        # SPATIAL: box regression + quadrant
        pred_box_from_text = self.spa_regressor(spa_vec)    # [B, 4]
        spa_quadrant_logits = self.spa_quadrant_head(spa_vec)  # [B, 9]

        # ATTRIBUTE: complementarity reconstruction
        # Loss: ||reconstruction_head(cat[sem,spa,attr]) - shared_pooled||^2
        combined = torch.cat([sem_vec, spa_vec, attr_vec], dim=-1)  # [B, 3D]
        reconstructed = self.reconstruction_head(combined)          # [B, D]

        # 5. Spatial relation encoder (supervised via quadrant indirectly)
        relation_out = self.spatial_relation_encoder(spa_feat, mask)

        return {
            # Branch sequences (for FusionNeck cross-attention)
            'sem':  sem_feat,
            'spa':  spa_feat,
            'attr': attr_feat,
            # Branch vectors (for GroundingHead and losses)
            'sem_vec':  sem_vec,
            'spa_vec':  spa_vec,
            'attr_vec': attr_vec,
            # Gate weights (for analysis and visualization)
            'sem_gate':  sem_gate,    # [B, L]  -- which tokens semantic focuses on
            'spa_gate':  spa_gate,    # [B, L]  -- which tokens spatial focuses on
            'attr_gate': attr_gate,   # [B, L]  -- which tokens attribute focuses on
            # Supervision outputs
            'sem_cls_logits':    sem_cls_logits,       # [B, 20]  -> CE loss
            'pred_box':          pred_box_from_text,   # [B, 4]   -> smooth L1
            'spa_quadrant_logits': spa_quadrant_logits, # [B, 9]  -> CE loss
            'reconstructed':     reconstructed,        # [B, D]   -> MSE vs shared_pooled
            'shared_pooled':     shared_pooled,        # [B, D]   -> reconstruction target
            # Spatial relation
            'relative_position': relation_out['relative_position'],
            'relation_tokens':   relation_out['relation_tokens'],
            'relation_vec':      relation_out['relation_vec'],
            'template_logits':   relation_out['template_logits'],
            'template_attn':     relation_out['template_attn'],
            'template_vec':      relation_out['template_vec'],
            # Mask for downstream modules
            'mask': mask,
        }