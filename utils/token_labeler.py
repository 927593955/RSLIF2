"""
utils/token_labeler.py

Vocabulary-based token type labeler. Generates [B, L] integer labels for
each input token:
    0 = SEMANTIC  (class names, determiners, most words)
    1 = SPATIAL   (left, right, above, near, north, cx, ...)
    2 = ATTRIBUTE (large, small, red, circular, parked, ...)
   -1 = IGNORE    (special tokens [CLS]/[SEP], padding)

These labels directly supervise the token router via cross-entropy loss,
replacing the entropy+diversity losses that had no target and collapsed
to the uniform-weight degenerate solution.
"""

from __future__ import annotations
from typing import List
import torch

LABEL_SEMANTIC  = 0
LABEL_SPATIAL   = 1
LABEL_ATTRIBUTE = 2
LABEL_IGNORE    = -1

SPATIAL_VOCAB: frozenset = frozenset({
    # Absolute positions
    "left", "right", "top", "bottom", "center", "middle",
    "upper", "lower", "above", "below",
    # Compass
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    # Relations
    "near", "beside", "next", "adjacent", "between", "among",
    "behind", "front", "inside", "outside", "along",
    "corner", "edge", "side",
    # Spatial hint tokens
    "cx", "cy",
})

ATTRIBUTE_VOCAB: frozenset = frozenset({
    # Size
    "large", "small", "tiny", "big", "long", "short", "wide",
    "narrow", "huge", "massive", "little",
    # Color
    "red", "white", "black", "dark", "bright", "gray", "grey",
    "blue", "green", "yellow", "brown",
    # Shape
    "circular", "rectangular", "elongated", "square", "round",
    "curved", "straight", "oval",
    # Material
    "metallic", "concrete", "grassy", "sandy", "paved",
    # State
    "parked", "moving", "docked", "empty", "full",
    # Count/density
    "multiple", "several", "dense", "sparse",
    # Orientation
    "horizontal", "vertical", "diagonal",
})


class TokenLabeler:
    """
    Pre-computes token_id → label mapping so per-batch labeling is O(L)
    with no tokenizer calls at training time.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._id_to_label: dict[int, int] = {}
        self._special_ids: set[int] = set()
        self._build()

    def _build(self):
        tok = self.tokenizer

        # Special token IDs → always IGNORE
        for attr in ['cls_token_id', 'sep_token_id', 'pad_token_id',
                     'unk_token_id', 'mask_token_id']:
            tid = getattr(tok, attr, None)
            if tid is not None:
                self._special_ids.add(int(tid))

        def register(word: str, label: int):
            # Tokenize without special tokens; label every sub-word token
            ids = tok.encode(word, add_special_tokens=False)
            for tid in ids:
                # Higher label wins: spatial (1) > attribute (2)?
                # Priority: SPATIAL > ATTRIBUTE so spatial words aren't
                # mislabeled as attribute if they share sub-tokens
                existing = self._id_to_label.get(tid, LABEL_SEMANTIC)
                if label > existing:
                    self._id_to_label[tid] = label

        for w in ATTRIBUTE_VOCAB:
            register(w, LABEL_ATTRIBUTE)
            register(w.capitalize(), LABEL_ATTRIBUTE)

        for w in SPATIAL_VOCAB:
            # Spatial overwrites attribute for shared sub-tokens
            register(w, LABEL_SPATIAL)
            register(w.capitalize(), LABEL_SPATIAL)
            register(w.upper(), LABEL_SPATIAL)

    def label_ids(self, input_ids: torch.Tensor,
                  attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids:      [B, L]
        attention_mask: [B, L]
        Returns:        [B, L] int64 labels
        """
        flat = input_ids.reshape(-1).tolist()
        out = [
            LABEL_IGNORE if tid in self._special_ids
            else self._id_to_label.get(tid, LABEL_SEMANTIC)
            for tid in flat
        ]
        labels = torch.tensor(out, dtype=torch.long,
                              device=input_ids.device).reshape(input_ids.shape)
        # Padding → IGNORE
        labels = labels.masked_fill(attention_mask == 0, LABEL_IGNORE)
        return labels

    def tokenize_with_labels(
        self,
        texts: List[str],
        max_length: int = 77,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize texts and return (input_ids, attention_mask, token_type_labels).
        Call this in the dataset __getitem__ or collate_fn so labels are
        pre-computed on CPU and moved to GPU with the rest of the batch.
        """
        enc = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        ids  = enc['input_ids']
        mask = enc['attention_mask']
        lbls = self.label_ids(ids, mask)
        return ids, mask, lbls

    def coverage_stats(self, token_type_labels: torch.Tensor,
                       attention_mask: torch.Tensor) -> dict:
        """Diagnostic: fraction of real tokens per type."""
        valid = (attention_mask == 1) & (token_type_labels != LABEL_IGNORE)
        lv = token_type_labels[valid]
        n = lv.numel()
        if n == 0:
            return {}
        return {
            'sem_frac':  (lv == LABEL_SEMANTIC).float().mean().item(),
            'spa_frac':  (lv == LABEL_SPATIAL).float().mean().item(),
            'attr_frac': (lv == LABEL_ATTRIBUTE).float().mean().item(),
            'n_tokens':  n,
        }