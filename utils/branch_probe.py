"""
utils/branch_probe.py  (v2 — fixes gradient error in probe_accuracy)

BUG FIXED:
  run_branch_analysis is decorated @torch.no_grad(), which disables the
  gradient engine globally for the entire function body.
  Inside it, probe_accuracy creates a linear layer and calls loss.backward()
  — but no computation graph exists under no_grad, so backward raises:
    "element 0 of tensors does not require grad and does not have a grad_fn"

FIX:
  probe_accuracy now wraps the training loop in torch.enable_grad(), which
  temporarily re-enables gradients even inside the outer no_grad scope.
  The feature tensors are still detached (we only want to probe the frozen
  representations, not fine-tune the model), so the fix is:

    with torch.enable_grad():
        loss = F.cross_entropy(probe(features), targets)
        loss.backward()

  Note: remove .detach() from probe(features) inside the enable_grad block —
  features are already detached tensors collected under no_grad, so calling
  .detach() again is a no-op, but leaving it in makes the intent clearer.
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
    def __init__(self):
        self.sem_gates  = []
        self.spa_gates  = []
        self.attr_gates = []

    @torch.no_grad()
    def update(self, model_output: dict):
        if 'sem_gate' in model_output:
            self.sem_gates.append(model_output['sem_gate'].cpu())
        if 'spa_gate' in model_output:
            self.spa_gates.append(model_output['spa_gate'].cpu())
        if 'attr_gate' in model_output:
            self.attr_gates.append(model_output['attr_gate'].cpu())

    def _gate_entropy(self, gates_list):
        if not gates_list:
            return float('nan')
        gates = torch.cat(gates_list, dim=0)
        eps = 1e-10
        entropy = -(gates * (gates + eps).log()).sum(dim=-1)
        return entropy.mean().item()

    def _peak_position_bias(self, gates_list, seq_len):
        if not gates_list:
            return None
        gates = torch.cat(gates_list, dim=0)
        argmax = gates.argmax(dim=-1)
        hist = torch.zeros(seq_len)
        for pos in argmax:
            if pos < seq_len:
                hist[pos] += 1
        return (hist / hist.sum()).tolist()

    def compute(self, seq_len=77):
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
                'sem_norm':  sem_ent  / max_entropy if not np.isnan(sem_ent)  else None,
                'spa_norm':  spa_ent  / max_entropy if not np.isnan(spa_ent)  else None,
                'attr_norm': attr_ent / max_entropy if not np.isnan(attr_ent) else None,
            }
        }

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
# 2. Branch Probe
# =============================================================================

class BranchProbe(nn.Module):
    def __init__(self, hidden_dim, num_classes=20, spatial_grid=3):
        super().__init__()
        num_quads = spatial_grid * spatial_grid
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
# 3. run_branch_analysis
# =============================================================================

@torch.no_grad()
def run_branch_analysis(model, val_loader, device, epoch,
                        num_batches=50, spatial_grid=3):
    """
    Run after validation to log branch specialization metrics.
    The outer @torch.no_grad() covers feature collection.
    probe_accuracy uses torch.enable_grad() internally for probe training.
    """
    model.eval()
    analyzer = GateEntropyAnalyzer()

    sem_vecs, spa_vecs, attr_vecs = [], [], []
    cls_ids, quad_ids = [], []

    for i, batch in enumerate(tqdm(val_loader, desc='Branch analysis', leave=False)):
        if i >= num_batches:
            break

        imgs, input_ids, masks, targets, _ = [x.to(device) for x in batch]
        output = model(imgs, [input_ids, masks])
        analyzer.update(output)

        sem_vecs.append(output['sem_vec'].cpu())
        spa_vecs.append(output['spa_vec'].cpu())
        attr_vecs.append(output['attr_vec'].cpu())

        B = imgs.shape[0]
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
            cls_ids.append(cls_id)
            quad_ids.append(quad_id)

    # ── Gate entropy ─────────────────────────────────────────────────────────
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

    if not sem_vecs:
        return gate_stats

    sem_v  = torch.cat(sem_vecs,  0)
    spa_v  = torch.cat(spa_vecs,  0)
    attr_v = torch.cat(attr_vecs, 0)
    cls_t  = torch.tensor(cls_ids,  dtype=torch.long)
    quad_t = torch.tensor(quad_ids, dtype=torch.long)

    hidden_dim  = sem_v.shape[1]
    num_classes = 20
    num_quads   = spatial_grid * spatial_grid

    def probe_accuracy(features, targets, num_classes, steps=100, lr=0.01):
        """
        Train a linear probe on FROZEN branch vectors.

        FIX: run_branch_analysis is decorated @torch.no_grad(), which disables
        the gradient engine for the entire function.  We must re-enable it here
        with torch.enable_grad() so that the probe's parameters accumulate
        gradients and loss.backward() works.

        The branch vectors (features) are already detached — they were collected
        under no_grad — so this only adds gradients for the probe itself, not
        for the model being evaluated.
        """
        features_cpu = features.float()   # already detached (collected under no_grad)
        targets_cpu  = targets

        probe = nn.Linear(hidden_dim, num_classes)
        opt   = torch.optim.Adam(probe.parameters(), lr=lr)

        # ── Probe training requires gradients ─────────────────────────────────
        with torch.enable_grad():
            probe_device = device
            probe = probe.to(probe_device)
            feat_d = features_cpu.to(probe_device)
            tgt_d  = targets_cpu.to(probe_device)

            for _ in range(steps):
                opt.zero_grad()
                # features are detached; only probe.weight/bias get gradients
                loss = F.cross_entropy(probe(feat_d), tgt_d)
                loss.backward()
                opt.step()

        # ── Accuracy evaluation (no grad needed) ──────────────────────────────
        with torch.no_grad():
            preds = probe(feat_d).argmax(dim=-1)
            acc   = (preds == tgt_d).float().mean().item()

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