"""
train.py  (v4 — fixes for three bugs vs v3)

BUG FIXES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUG 1: GroundingLoss init used wrong parameter names.
  OLD: GroundingLoss(lambda_attr_pres=..., lambda_routing_ent=..., lambda_routing_div=...)
  NEW: GroundingLoss(lambda_spa_quad=..., lambda_sem_cls=..., lambda_reconstruct=...)
  These now match grounding_loss.py exactly.

BUG 2: get_lambdas() returned keys that don't exist in grounding_loss.py's lw dict.
  OLD keys: attr_pres, routing_ent, routing_div  (grounding_loss.py ignores them silently)
  NEW keys: spa_quad, reconstruct
  The lw.update() in grounding_loss only accepts keys it knows; wrong keys were no-ops,
  meaning the lambdas schedule was not actually being applied.

BUG 3: Optimizer had a 'token_router' param group that matched 0 parameters.
  No module named token_router exists in the model.
  Removed. Now only two groups: vision (full LR) and text (0.1× LR).

BUG 4: branch_probe.py existed but was never called.
  Added: run_branch_analysis() every 10 epochs after validate().
  This gives you the specialization verification that was the whole point of
  adding that module.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three-stage training schedule (unchanged logic, corrected keys):

Stage 1 (0 → stage1_epochs):
  Losses: giou + l1 only
  Goal: learn basic box localization before any branch-specific signals.

Stage 2 (stage1_epochs → stage1_epochs + stage2_offset):
  Losses: + spa + spa_quad + sem_cls
  Goal: separate spatial and semantic branches with direct supervision.

Stage 3 (stage1_epochs + stage2_offset → end):
  Losses: + reconstruct + orth
  Goal: enforce attribute branch complementarity.
  reconstruct ramps in over `reconstruct_ramp_epochs` to avoid destabilization.
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from warnings import simplefilter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

simplefilter(action='ignore', category=FutureWarning)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.dataset import DIORRSVGDataset, rsvlm_collate_fn
from models.vlm_grounding import RemoteSensingVLM
from utils.grounding_loss import GroundingLoss
from utils.ema import ModelEMA
from utils.branch_probe import run_branch_analysis   # BUG 4 FIX: now imported
from val import validate


# ─────────────────────────────────────────────────────────────────────────────
# BUG 1 + 2 FIX: Corrected three-stage lambda schedule
# Keys returned here must exactly match the 'lw' dict in GroundingLoss.forward()
# ─────────────────────────────────────────────────────────────────────────────

def get_lambdas(opt, epoch: int) -> dict:
    """
    Returns loss weight dict for the current epoch.

    Keys MUST match grounding_loss.py's lw dict:
        giou, l1, spa, spa_quad, sem_cls, reconstruct, orth

    Stage 1: giou + l1 only
    Stage 2: + spa + spa_quad + sem_cls
    Stage 3: + reconstruct + orth  (reconstruct ramps in slowly)
    """
    stage2_start = opt.stage1_epochs
    stage3_start = opt.stage1_epochs + opt.stage2_offset

    if epoch < stage2_start:
        # Stage 1: localization only
        return {
            'giou':        2.0,
            'l1':          5.0,
            'spa':         0.0,
            'spa_quad':    0.0,
            'sem_cls':     0.0,
            'reconstruct': 0.0,
            'orth':        0.0,
        }

    elif epoch < stage3_start:
        # Stage 2: direct branch supervision for spatial + semantic
        return {
            'giou':        2.0,
            'l1':          5.0,
            'spa':         opt.lambda_spa,
            'spa_quad':    opt.lambda_spa_quad,
            'sem_cls':     opt.lambda_sem_cls,
            'reconstruct': 0.0,   # attribute not yet — sem+spa must stabilize first
            'orth':        0.0,
        }

    else:
        # Stage 3: attribute complementarity + orthogonality
        # Ramp reconstruct in slowly: too sudden → training instability
        epochs_in_s3 = epoch - stage3_start
        ramp = min(1.0, epochs_in_s3 / max(opt.reconstruct_ramp_epochs, 1))
        return {
            'giou':        2.0,
            'l1':          5.0,
            'spa':         opt.lambda_spa,
            'spa_quad':    opt.lambda_spa_quad,
            'sem_cls':     opt.lambda_sem_cls,
            'reconstruct': opt.lambda_reconstruct * ramp,
            'orth':        opt.lambda_orth,
        }


def to_fp32(x):
    if torch.is_tensor(x):
        return x.float()
    if isinstance(x, dict):
        return {k: to_fp32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_fp32(v) for v in x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(opt):
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        f'cuda:{opt.device}'
        if torch.cuda.is_available() and opt.device != 'cpu'
        else 'cpu'
    )
    print(f'Training on {device} | batch_size={opt.batch_size} | Saving to {save_dir}')

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = DIORRSVGDataset(
        data_root=opt.data_root,
        xml_root=opt.xml_root,
        split_txt_path=opt.train_txt,
        tokenizer_path=opt.text_model,
        img_size=opt.imgsz,
        max_len=opt.max_len,
        min_objects_per_image=opt.min_objects_per_image,
        use_all_images=opt.use_all_images,
        spatial_hint_prob=opt.train_spatial_hint_prob,
    )
    val_dataset = DIORRSVGDataset(
        data_root=opt.data_root,
        xml_root=opt.xml_root,
        split_txt_path=opt.val_txt,
        tokenizer_path=opt.text_model,
        img_size=opt.imgsz,
        max_len=opt.max_len,
        min_objects_per_image=1,
        use_all_images=False,
        spatial_hint_prob=0.0,
    )

    nw = opt.workers
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=nw, collate_fn=rsvlm_collate_fn, pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=max(1, nw // 2), collate_fn=rsvlm_collate_fn, pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = {
        'deberta_path':   opt.text_model,
        'yolo_weight':    opt.yolo_weight,
        'hidden_dim':     opt.hidden_dim,
        'backbone_width': 0.5,
        'backbone_depth': 0.33,
    }
    model = RemoteSensingVLM(model_cfg).to(device)

    if opt.freeze_shared_encoder:
        frozen = 0
        for n, p in model.text_encoder.named_parameters():
            if 'shared_encoder' in n or 'embeddings' in n:
                p.requires_grad_(False)
                frozen += p.numel()
        print(f'  Frozen {frozen/1e6:.1f}M params in shared text encoder.')

    if opt.compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print('  torch.compile enabled.')
        except Exception as e:
            print(f'  torch.compile not available: {e}')

    # ── BUG 3 FIX: Optimizer — removed dead 'token_router' param group ────────
    # Previously the optimizer had a third group filtering by 'token_router' in
    # parameter names, which matched nothing and silently created an empty group.
    # Now: two groups only — vision (full LR) and text (0.1x LR).
    text_params = [
        p for n, p in model.named_parameters()
        if 'text_encoder' in n and p.requires_grad
    ]
    vision_params = [
        p for n, p in model.named_parameters()
        if 'text_encoder' not in n and p.requires_grad
    ]

    print(
        f'  Trainable: vision={sum(p.numel() for p in vision_params)/1e6:.1f}M  '
        f'text={sum(p.numel() for p in text_params)/1e6:.1f}M params'
    )

    optimizer = optim.AdamW([
        {'params': vision_params, 'lr': opt.lr0},
        {'params': text_params,   'lr': opt.lr0 * 0.1},
    ], weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.epochs, eta_min=1e-6
    )

    # ── BUG 1 FIX: GroundingLoss initialized with correct parameter names ─────
    # OLD (broken): GroundingLoss(lambda_attr_pres=..., lambda_routing_ent=..., lambda_routing_div=...)
    # NEW (correct): matches grounding_loss.py __init__ signature exactly
    loss_fn = GroundingLoss(
        lambda_giou=2.0,
        lambda_l1=5.0,
        lambda_spa=opt.lambda_spa,
        lambda_spa_quad=opt.lambda_spa_quad,     # was: lambda_attr_pres (wrong name)
        lambda_sem_cls=opt.lambda_sem_cls,
        lambda_reconstruct=opt.lambda_reconstruct, # was: lambda_routing_ent (wrong name)
        lambda_orth=opt.lambda_orth,
        spatial_grid=3,
    )

    ema    = ModelEMA(model)
    scaler = amp.GradScaler(enabled=True)

    nb         = len(train_loader)
    nw_warmup  = max(round(opt.warmup_epochs * nb), 100)
    accumulate = max(1, round(64 / opt.batch_size))

    # 7 tracked loss components (matching grounding_loss.py loss_dict keys)
    loss_keys = ['giou', 'l1', 'spa', 'spa_quad', 'sem_cls', 'reconstruct', 'orth']

    stage2_start = opt.stage1_epochs
    stage3_start = opt.stage1_epochs + opt.stage2_offset
    print(f'  Steps/epoch: {nb} | Warmup: {nw_warmup} | Accum: {accumulate}')
    print(f'  Stage schedule: S1=0→{stage2_start} | S2={stage2_start}→{stage3_start} | S3={stage3_start}→{opt.epochs}')

    best_acc = 0.0

    for epoch in range(opt.epochs):
        model.train()
        lambdas = get_lambdas(opt, epoch)

        # Stage transition logging
        if epoch == 0:
            tqdm.write(f'\n📍 Epoch 1: Stage 1 — localization only (giou + l1)')
        elif epoch == stage2_start:
            tqdm.write(f'\n📍 Epoch {epoch+1}: Stage 2 — +spa +spa_quad +sem_cls')
        elif epoch == stage3_start:
            tqdm.write(f'\n📍 Epoch {epoch+1}: Stage 3 — +reconstruct (ramping) +orth')

        mloss = torch.zeros(len(loss_keys) + 1, device=device)  # +1 for total

        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{opt.epochs}',
            bar_format='{l_bar}{bar:10}{r_bar}',
        )

        for batch_i, batch in enumerate(pbar):
            ni = batch_i + nb * epoch

            # Warmup LR
            if ni <= nw_warmup:
                for j, x in enumerate(optimizer.param_groups):
                    base = opt.lr0 if j == 0 else opt.lr0 * 0.1
                    x['lr'] = 0.1 * base + (base - 0.1 * base) * (ni / nw_warmup)

            imgs, input_ids, masks, yolo_targets, _ = [
                x.to(device, non_blocking=True) for x in batch
            ]

            with amp.autocast(enabled=True):
                preds = model(imgs, [input_ids, masks])

            with amp.autocast(enabled=False):
                preds_fp32 = to_fp32(preds)
                loss, loss_dict = loss_fn(preds_fp32, yolo_targets, lambdas)

            scaler.scale(loss).backward()

            if (batch_i + 1) % accumulate == 0 or (batch_i + 1) == nb:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            with torch.no_grad():
                vals = [loss_dict.get(k, 0.0) for k in loss_keys] + [loss_dict.get('total', 0.0)]
                items = torch.tensor(vals, device=device)
                mloss = (mloss * batch_i + items) / (batch_i + 1)

            # BUG 2 FIX: pbar now shows keys that actually exist in loss_dict
            pbar.set_postfix(
                GIoU=f'{mloss[0]:.3f}',
                L1=f'{mloss[1]:.3f}',
                Spa=f'{mloss[2]:.3f}',
                Quad=f'{mloss[3]:.3f}',
                SCls=f'{mloss[4]:.3f}',
                Rec=f'{mloss[5]:.3f}',
                Tot=f'{mloss[7]:.3f}',
                lr=f'{optimizer.param_groups[0]["lr"]:.5f}',
            )

            del imgs, input_ids, masks, loss

        # ── Validation ────────────────────────────────────────────────────────
        do_val = (
            (epoch + 1) % opt.val_interval == 0
            or (epoch + 1) == opt.epochs
            or epoch < 5
        )
        if do_val:
            torch.cuda.empty_cache()
            val_model = ema.ema if ema else model
            metrics   = validate(val_model, val_loader, device, verbose=True)
            acc5      = metrics['acc@0.5']

            if acc5 > best_acc:
                best_acc = acc5
                ckpt = {
                    'epoch':    epoch,
                    'model':    deepcopy(val_model).half().state_dict(),
                    'opt':      vars(opt),
                    'best_acc': best_acc,
                    'metrics':  metrics,
                }
                torch.save(ckpt, save_dir / 'best.pt')
                tqdm.write(f'✅ Best: Acc@0.5={best_acc:.4f}')

            # ── BUG 4 FIX: branch_probe.py now actually called ────────────────
            # Measures whether branches are genuinely specializing.
            # Every 10 epochs; output printed to console.
            # Skip epoch 0 — branches haven't had time to specialize yet.
            if epoch > 0 and (epoch + 1) % opt.probe_interval == 0:
                try:
                    run_branch_analysis(val_model, val_loader, device, epoch,
                                        num_batches=30, spatial_grid=3)
                except Exception as e:
                    tqdm.write(f'⚠️  branch_probe failed: {e}')

        if opt.save_period > 0 and (epoch + 1) % opt.save_period == 0:
            torch.save(
                {'epoch': epoch, 'model': model.state_dict()},
                save_dir / 'last.pt',
            )

        scheduler.step()

    print('✅ Training complete.')


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_opt():
    p = argparse.ArgumentParser()

    # Hardware
    p.add_argument('--device',  default='2')
    p.add_argument('--workers', type=int, default=4)

    # Training schedule
    p.add_argument('--epochs',        type=int,   default=150)
    p.add_argument('--batch-size',    type=int,   default=48)
    p.add_argument('--imgsz',         type=int,   default=640)
    p.add_argument('--warmup-epochs', type=int,   default=5)
    p.add_argument('--save-period',   type=int,   default=10)
    p.add_argument('--val-interval',  type=int,   default=5)

    # Paths
    p.add_argument('--project',    default=str(ROOT / 'runs/train'))
    p.add_argument('--name',       default='grounding_v4')
    p.add_argument('--data-root',  default='/data/bxc/DIOR-RSVG/JPEGImages')
    p.add_argument('--xml-root',   default='/data/bxc/DIOR-RSVG/Annotations')
    p.add_argument('--train-txt',  default='/data/bxc/DIOR-RSVG/train.txt')
    p.add_argument('--val-txt',    default='/data/bxc/DIOR-RSVG/val.txt')
    p.add_argument('--text-model', default='./pretrain_ckp/deberta-v3-small')
    p.add_argument('--yolo-weight',default='yolov8s.pt')

    # Model
    p.add_argument('--max-len',    type=int, default=77)
    p.add_argument('--hidden-dim', type=int, default=256)

    # Dataset
    p.add_argument('--min-objects-per-image',   type=int,   default=1)
    p.add_argument('--use-all-images',          action='store_true')
    p.add_argument('--train-spatial-hint-prob', type=float, default=0.0)

    # Optimizer
    p.add_argument('--lr0',          type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.05)

    # Stage schedule
    p.add_argument('--stage1-epochs',          type=int, default=10,
                   help='Epochs of localization-only training (Stage 1)')
    p.add_argument('--stage2-offset',          type=int, default=10,
                   help='Additional epochs after stage1 before attribute loss (Stage 2 length)')
    p.add_argument('--reconstruct-ramp-epochs',type=int, default=10,
                   help='Epochs to ramp reconstruct loss from 0 to full weight in Stage 3')

    # Loss weights — names now match grounding_loss.py exactly (BUG 1 FIX)
    p.add_argument('--lambda-spa',         type=float, default=1.0,
                   help='Spatial branch box regression weight')
    p.add_argument('--lambda-spa-quad',    type=float, default=0.5,
                   help='Spatial branch quadrant classification weight')
    p.add_argument('--lambda-sem-cls',     type=float, default=1.0,
                   help='Semantic branch class discrimination weight')
    p.add_argument('--lambda-reconstruct', type=float, default=0.5,
                   help='Attribute branch complementarity reconstruction weight')
    p.add_argument('--lambda-orth',        type=float, default=0.3,
                   help='Semantic-attribute orthogonality weight')

    # Probing (BUG 4 FIX)
    p.add_argument('--probe-interval', type=int, default=10,
                   help='Run branch specialization analysis every N epochs (0 = off)')

    # Speed
    p.add_argument('--compile',               action='store_true')
    p.add_argument('--freeze-shared-encoder', action='store_true')

    return p.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)