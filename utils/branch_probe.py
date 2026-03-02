"""
utils/branch_probe.py

PROBLEM 3 FIX: Specialization is unverified.

This module provides concrete measurements of branch specialization:

1. GateEntropyAnalyzer:
   - Measures entropy of each branch's token gate weights
   - Low entropy = focused on few tokens (specialized)
   - High entropy = diffuse attention (not specialized)
   - Reports which token POSITIONS each branch focuses on
   - Lets you verify: does the spatial gate really focus on position tokens?

2. BranchProbe:
   - Trains lightweight linear probes on frozen branch vectors
   - Measures: can sem_vec predict class? (should be YES)
   - Measures: can spa_vec predict quadrant? (should be YES)
   - Measures: can attr_vec predict class? (should be LOW)
   - Measures: can sem_vec predict quadrant? (should be LOW)
   - Decoupled accuracy = specialization confirmed

3. run_branch_analysis():
   - Run after each validation epoch to log specialization metrics
   - Call from val.py or train.py

Usage:
    from utils.branch_probe import run_branch_analysis
    # In your validation loop, after validate():
    run_branch_analysis(model, val_loader, device, epoch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict


# =============================================================================
# 1. Gate Entropy Analyzer
# =============================================================================

class GateEntropyAnalyzer:
    """
    Collects gate weight tensors from a forward pass and computes entropy.
    
    Usage:
        analyzer = GateEntropyAnalyzer()
        # Run a batch through the model
        output = model(imgs, texts)
        analyzer.update(output)
        stats = analyzer.compute()
        print(stats)
    """

    def __init__(self):
        self.sem_gates  = []
        self.spa_gates  = []
        self.attr_gates = []

    @torch.no_grad()
    def update(self, model_output: dict):
        """Accumulate gate weights from a model forward pass."""
        if 'sem_gate' in model_output:
            self.sem_gates.append(model_output['sem_gate'].cpu())   # [B, L]
        if 'spa_gate' in model_output:
            self.spa_gates.append(model_output['spa_gate'].cpu())
        if 'attr_gate' in model_output:
            self.attr_gates.append(model_output['attr_gate'].cpu())

    def _gate_entropy(self, gates_list):
        """Mean entropy of gate distributions. Lower = more focused."""
        if not gates_list:
            return float('nan')
        gates = torch.cat(gates_list, dim=0)      # [N, L]
        # Entropy: -sum(p * log(p))
        # Gates are already softmax, so they sum to 1
        eps = 1e-10
        entropy = -(gates * (gates + eps).log()).sum(dim=-1)  # [N]
        return entropy.mean().item()

    def _peak_position_bias(self, gates_list, seq_len):
        """
        For each position in the sequence, how often is it the argmax gate?
        Returns normalized position preference histogram.
        """
        if not gates_list:
            return None
        gates = torch.cat(gates_list, dim=0)   # [N, L]
        argmax = gates.argmax(dim=-1)           # [N]
        hist = torch.zeros(seq_len)
        for pos in argmax:
            if pos < seq_len:
                hist[pos] += 1
        return (hist / hist.sum()).tolist()

    def compute(self, seq_len=77):
        """
        Returns dict of entropy and positional statistics.
        
        Interpretation:
          - sem entropy LOW (<2.0): semantic gate is focusing on specific tokens ✓
          - spa entropy LOW (<2.0): spatial gate is focusing on position tokens ✓
          - If all entropies are equal and high: branches not specializing ✗
        """
        # Maximum possible entropy for L tokens = log(L)
        max_entropy = np.log(seq_len)

        sem_ent  = self._gate_entropy(self.sem_gates)
        spa_ent  = self._gate_entropy(self.spa_gates)
        attr_ent = self._gate_entropy(self.attr_gates)

        results = {
            'gate_entropy': {
                'semantic':  sem_ent,
                'spatial':   spa_ent,
                'attribute': attr_ent,
                'max_possible': max_entropy,
                # Normalized to [0,1]: 0=perfectly focused, 1=fully uniform
                'sem_norm':  sem_ent  / max_entropy if not np.isnan(sem_ent)  else None,
                'spa_norm':  spa_ent  / max_entropy if not np.isnan(spa_ent)  else None,
                'attr_norm': attr_ent / max_entropy if not np.isnan(attr_ent) else None,
            }
        }

        # Position histograms (first 20 positions for display)
        sem_hist  = self._peak_position_bias(self.sem_gates,  seq_len)
        spa_hist  = self._peak_position_bias(self.spa_gates,  seq_len)
        attr_hist = self._peak_position_bias(self.attr_gates, seq_len)

        if sem_hist:
            results['peak_position_top5'] = {
                'semantic':  sorted(range(len(sem_hist)),  key=lambda i: -sem_hist[i])[:5],
                'spatial':   sorted(range(len(spa_hist)),  key=lambda i: -spa_hist[i])[:5],
                'attribute': sorted(range(len(attr_hist)), key=lambda i: -attr_hist[i])[:5],
            }

        return results

    def reset(self):
        self.sem_gates  = []
        self.spa_gates  = []
        self.attr_gates = []


# =============================================================================
# 2. Branch Probe (linear probing for specialization measurement)
# =============================================================================

class BranchProbe(nn.Module):
    """
    Lightweight linear probes that measure what each branch has learned.

    Train these probes on the FROZEN branch vectors.
    High probe accuracy = branch encodes that information.
    Low probe accuracy = branch does NOT encode that information.

    Expected results if branches are specialized:
        sem_vec -> class_id:  HIGH accuracy (~DIOR 20-class accuracy)
        spa_vec -> quadrant:  HIGH accuracy (~9-class grid cell)
        attr_vec -> class_id: LOW accuracy (attribute ≠ class category)
        sem_vec -> quadrant:  LOW accuracy (semantic ≠ location)
    """

    def __init__(self, hidden_dim, num_classes=20, spatial_grid=3):
        super().__init__()
        num_quads = spatial_grid * spatial_grid

        # One probe per (branch, task) combination
        self.sem_cls_probe   = nn.Linear(hidden_dim, num_classes)
        self.sem_quad_probe  = nn.Linear(hidden_dim, num_quads)
        self.spa_cls_probe   = nn.Linear(hidden_dim, num_classes)
        self.spa_quad_probe  = nn.Linear(hidden_dim, num_quads)
        self.attr_cls_probe  = nn.Linear(hidden_dim, num_classes)
        self.attr_quad_probe = nn.Linear(hidden_dim, num_quads)

    def forward(self, sem_vec, spa_vec, attr_vec):
        return {
            'sem_cls':   self.sem_cls_probe(sem_vec.detach()),
            'sem_quad':  self.sem_quad_probe(sem_vec.detach()),
            'spa_cls':   self.spa_cls_probe(spa_vec.detach()),
            'spa_quad':  self.spa_quad_probe(spa_vec.detach()),
            'attr_cls':  self.attr_cls_probe(attr_vec.detach()),
            'attr_quad': self.attr_quad_probe(attr_vec.detach()),
        }


# =============================================================================
# 3. run_branch_analysis: plug into your training loop
# =============================================================================

@torch.no_grad()
def run_branch_analysis(model, val_loader, device, epoch,
                        num_batches=50, spatial_grid=3):
    """
    Run after validation to log branch specialization metrics.

    Outputs:
      1. Gate entropy per branch (lower is more specialized)
      2. Token position histogram peaks
      3. Inline classification accuracy report

    Typical usage in train.py, after validate():
        if (epoch + 1) % 10 == 0:
            run_branch_analysis(model, val_loader, device, epoch)
    """
    model.eval()
    analyzer = GateEntropyAnalyzer()

    # Collect branch vectors and targets
    sem_vecs, spa_vecs, attr_vecs = [], [], []
    cls_ids, quad_ids = [], []

    for i, batch in enumerate(tqdm(val_loader, desc='Branch analysis', leave=False)):
        if i >= num_batches:
            break

        imgs, input_ids, masks, targets, _ = [x.to(device) for x in batch]

        output = model(imgs, [input_ids, masks])

        analyzer.update(output)

        # Collect branch vectors
        sem_vecs.append(output['sem_vec'].cpu())
        spa_vecs.append(output['spa_vec'].cpu())
        attr_vecs.append(output['attr_vec'].cpu())

        # Collect GT class and quadrant per sample
        B = imgs.shape[0]
        batch_cls  = []
        batch_quad = []
        for si in range(B):
            rows = targets[targets[:, 0].long() == si]
            if len(rows) > 0:
                cls_id = int(rows[0, 1].item())
                cx = rows[0, 2].item()
                cy = rows[0, 3].item()
                col = min(int(cx * spatial_grid), spatial_grid - 1)
                row = min(int(cy * spatial_grid), spatial_grid - 1)
                quad_id = row * spatial_grid + col
            else:
                cls_id  = 0
                quad_id = 0
            batch_cls.append(cls_id)
            batch_quad.append(quad_id)

        cls_ids.extend(batch_cls)
        quad_ids.extend(batch_quad)

    # ── Gate entropy analysis ────────────────────────────────────────────────
    gate_stats = analyzer.compute()
    print(f'\n📊 Branch Analysis — Epoch {epoch + 1}')
    print('─' * 60)

    ent = gate_stats['gate_entropy']
    print('  Gate Entropy (lower = more focused/specialized):')
    print(f'    Semantic  : {ent["semantic"]:.3f}  (norm: {ent["sem_norm"]:.3f})')
    print(f'    Spatial   : {ent["spatial"]:.3f}  (norm: {ent["spa_norm"]:.3f})')
    print(f'    Attribute : {ent["attribute"]:.3f}  (norm: {ent["attr_norm"]:.3f})')
    print(f'    Max possible: {ent["max_possible"]:.3f}')

    if 'peak_position_top5' in gate_stats:
        pos = gate_stats['peak_position_top5']
        print('  Peak gate positions (token indices, 0=CLS):')
        print(f'    Semantic  : {pos["semantic"]}')
        print(f'    Spatial   : {pos["spatial"]}')
        print(f'    Attribute : {pos["attribute"]}')

    # ── Linear probing ────────────────────────────────────────────────────────
    if not sem_vecs:
        return gate_stats

    sem_v  = torch.cat(sem_vecs,  0)   # [N, D]
    spa_v  = torch.cat(spa_vecs,  0)
    attr_v = torch.cat(attr_vecs, 0)
    cls_t  = torch.tensor(cls_ids,  dtype=torch.long)
    quad_t = torch.tensor(quad_ids, dtype=torch.long)

    hidden_dim  = sem_v.shape[1]
    num_classes = 20
    num_quads   = spatial_grid * spatial_grid

    def probe_accuracy(features, targets, num_classes, steps=100, lr=0.01):
        """Train a quick linear probe and return accuracy."""
        features = features.to(device)
        targets  = targets.to(device)
        probe = nn.Linear(hidden_dim, num_classes).to(device)
        opt   = torch.optim.Adam(probe.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.cross_entropy(probe(features.detach()), targets)
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds = probe(features.detach()).argmax(dim=-1)
            acc   = (preds == targets).float().mean().item()
        return acc

    print('\n  Linear Probe Accuracy (specialization measurement):')
    print('  Expected: SEM→cls HIGH, SPA→quad HIGH, cross-probes LOW')

    sem_cls_acc  = probe_accuracy(sem_v,  cls_t,  num_classes)
    spa_quad_acc = probe_accuracy(spa_v,  quad_t, num_quads)
    attr_cls_acc = probe_accuracy(attr_v, cls_t,  num_classes)
    sem_quad_acc = probe_accuracy(sem_v,  quad_t, num_quads)

    print(f'    sem_vec  → class_id  : {sem_cls_acc:.3f}  '
          f'{"✓ HIGH (specialized)" if sem_cls_acc > 0.4 else "✗ LOW (not specialized)"}')
    print(f'    spa_vec  → quadrant  : {spa_quad_acc:.3f}  '
          f'{"✓ HIGH (specialized)" if spa_quad_acc > 0.3 else "✗ LOW (not specialized)"}')
    print(f'    attr_vec → class_id  : {attr_cls_acc:.3f}  '
          f'{"✓ LOW (correctly different)" if attr_cls_acc < sem_cls_acc * 0.7 else "✗ HIGH (not differentiated)"}')
    print(f'    sem_vec  → quadrant  : {sem_quad_acc:.3f}  '
          f'{"✓ LOW (correctly different)" if sem_quad_acc < spa_quad_acc * 0.7 else "✗ HIGH (not differentiated)"}')

    specialization_score = (
        (sem_cls_acc if sem_cls_acc > 0.4 else 0.0) +
        (spa_quad_acc if spa_quad_acc > 0.3 else 0.0) +
        (1.0 - attr_cls_acc if attr_cls_acc < sem_cls_acc * 0.7 else 0.0) +
        (1.0 - sem_quad_acc if sem_quad_acc < spa_quad_acc * 0.7 else 0.0)
    ) / 4.0

    print(f'\n  Specialization Score: {specialization_score:.3f} / 1.0')
    print('  (>0.5 = branches are genuinely specialized)')
    print('─' * 60)

    return {
        'gate_entropy':   ent,
        'sem_cls_acc':    sem_cls_acc,
        'spa_quad_acc':   spa_quad_acc,
        'attr_cls_acc':   attr_cls_acc,
        'sem_quad_acc':   sem_quad_acc,
        'specialization': specialization_score,
    }