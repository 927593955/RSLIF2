"""
train.py  (v5 — fixes branch collapse by enabling supervision from epoch 0)

BUG FIX (v4→v5):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUG 5 (the core cause of zero training progress):
  Stage 1 had sem_cls=0 and spa_quad=0, meaning all three branch encoders
  received IDENTICAL gradient signals in the first 10 epochs.
  Since they were also initialised from identical weights (deepcopy), they
  converged to the same representation and stayed there permanently.
  Gate entropy = 2.174 for ALL branches at BOTH epoch 20 and epoch 150.

  FIX: Stage 1 now includes small sem_cls=0.2 and spa_quad=0.1.
  These give semantic_encoder and spatial_encoder DIFFERENT gradient signals
  from batch 1, breaking the symmetry before it solidifies.

BUG 6: L_div (branch diversity loss) was absent.
  Even with different stage-2 losses, once branches are fully collapsed, the
  reconstruction and orth losses cannot un-collapse them (see ROOT_CAUSE_ANALYSIS.md).
  FIX: L_div is now active in all stages (lambda_div=0.5 in Stage 1, full in S2/S3).
  L_div = mean pairwise cosine² penalty; added as a key to get_lambdas().

BUG 4 (from v4, pbar cosmetic):
  orth loss (index 6) was never displayed in the pbar, making it impossible
  to see if it was exploding or zero during training.
  FIX: Added Orth= to pbar display.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
from utils.branch_probe import run_branch_analysis
from val import validate


# ─────────────────────────────────────────────────────────────────────────────
# FIX 5 + 6: Three-stage lambda schedule with branch losses from epoch 0
# ─────────────────────────────────────────────────────────────────────────────

def get_lambdas(opt, epoch: int) -> dict:
    """
    Returns loss weight dict for the current epoch.

    Keys match grounding_loss.py's lw dict:
        giou, l1, spa, spa_quad, sem_cls, reconstruct, orth, div

    Stage 1: giou + l1 + div + SMALL sem_cls + SMALL spa_quad   ← KEY FIX
    Stage 2: all of Stage 1 at full weight + spa
    Stage 3: + reconstruct (ramped) + orth
    """
    stage2_start = opt.stage1_epochs
    stage3_start = opt.stage1_epochs + opt.stage2_offset

    if epoch < stage2_start:
        # Stage 1: localization + branch symmetry-breaking
        # sem_cls and spa_quad at SMALL but non-zero weight from epoch 0.
        # This is what was missing — without it, all three branches converge
        # to identical representations in the first 10 epochs.
        return {
            'giou':        2.0,
            'l1':          5.0,
            'spa':         0.0,
            'spa_quad':    opt.lambda_spa_quad_s1,  # small: 0.1
            'sem_cls':     opt.lambda_sem_cls_s1,   # small: 0.2
            'reconstruct': 0.0,
            'orth':        0.0,
            'div':         opt.lambda_div,           # diversity active from day 1
        }

    elif epoch < stage3_start:
        # Stage 2: full branch supervision
        return {
            'giou':        2.0,
            'l1':          5.0,
            'spa':         opt.lambda_spa,
            'spa_quad':    opt.lambda_spa_quad,
            'sem_cls':     opt.lambda_sem_cls,
            'reconstruct': 0.0,
            'orth':        0.0,
            'div':         opt.lambda_div,
        }

    else:
        # Stage 3: + attribute complementarity
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
            'div':         opt.lambda_div,
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
        data_root=opt.data_root, xml_root=opt.xml_root,
        split_txt_path=opt.train_txt, tokenizer_path=opt.text_model,
        img_size=opt.imgsz, max_len=opt.max_len,
        min_objects_per_image=opt.min_objects_per_image,
        use_all_images=opt.use_all_images,
        spatial_hint_prob=opt.train_spatial_hint_prob,
    )
    val_dataset = DIORRSVGDataset(
        data_root=opt.data_root, xml_root=opt.xml_root,
        split_txt_path=opt.val_txt, tokenizer_path=opt.text_model,
        img_size=opt.imgsz, max_len=opt.max_len,
        min_objects_per_image=1, use_all_images=False, spatial_hint_prob=0.0,
    )

    nw = opt.workers
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=nw, collate_fn=rsvlm_collate_fn, pin_memory=True,
        persistent_workers=(nw > 0), prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=max(1, nw // 2), collate_fn=rsvlm_collate_fn, pin_memory=True,
        persistent_workers=(nw > 0), prefetch_factor=2 if nw > 0 else None,
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

    # ── Optimizer ─────────────────────────────────────────────────────────────
    text_params   = [p for n, p in model.named_parameters() if 'text_encoder' in n and p.requires_grad]
    vision_params = [p for n, p in model.named_parameters() if 'text_encoder' not in n and p.requires_grad]

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

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = GroundingLoss(
        lambda_giou=2.0,
        lambda_l1=5.0,
        lambda_spa=opt.lambda_spa,
        lambda_spa_quad=opt.lambda_spa_quad,
        lambda_sem_cls=opt.lambda_sem_cls,
        lambda_reconstruct=opt.lambda_reconstruct,
        lambda_orth=opt.lambda_orth,
        lambda_div=opt.lambda_div,      # NEW: branch diversity
        spatial_grid=3,
    )

    ema    = ModelEMA(model)
    scaler = amp.GradScaler(enabled=True)

    nb         = len(train_loader)
    nw_warmup  = max(round(opt.warmup_epochs * nb), 100)
    accumulate = max(1, round(64 / opt.batch_size))

    # 8 tracked loss components (7 named + total)
    # FIX 6: added 'div'
    loss_keys = ['giou', 'l1', 'spa', 'spa_quad', 'sem_cls', 'reconstruct', 'orth', 'div']

    stage2_start = opt.stage1_epochs
    stage3_start = opt.stage1_epochs + opt.stage2_offset
    print(f'  Steps/epoch: {nb} | Warmup: {nw_warmup} | Accum: {accumulate}')
    print(f'  Stage schedule: S1=0→{stage2_start} | S2={stage2_start}→{stage3_start} | S3={stage3_start}→{opt.epochs}')
    print(f'  Stage 1 branch seeding: sem_cls={opt.lambda_sem_cls_s1}, spa_quad={opt.lambda_spa_quad_s1}, div={opt.lambda_div}')

    best_acc = 0.0

    for epoch in range(opt.epochs):
        model.train()
        lambdas = get_lambdas(opt, epoch)

        if epoch == 0:
            tqdm.write(f'\n📍 Epoch 1: Stage 1 — localization + branch seeding '
                       f'(sem_cls={lambdas["sem_cls"]:.2f} spa_quad={lambdas["spa_quad"]:.2f} div={lambdas["div"]:.2f})')
        elif epoch == stage2_start:
            tqdm.write(f'\n📍 Epoch {epoch+1}: Stage 2 — full branch supervision +spa +spa_quad +sem_cls')
        elif epoch == stage3_start:
            tqdm.write(f'\n📍 Epoch {epoch+1}: Stage 3 — +reconstruct (ramping) +orth')

        mloss = torch.zeros(len(loss_keys) + 1, device=device)

        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{opt.epochs}',
            bar_format='{l_bar}{bar:10}{r_bar}',
        )

        for batch_i, batch in enumerate(pbar):
            ni = batch_i + nb * epoch

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
                vals  = [loss_dict.get(k, 0.0) for k in loss_keys] + [loss_dict.get('total', 0.0)]
                items = torch.tensor(vals, device=device)
                mloss = (mloss * batch_i + items) / (batch_i + 1)

            # FIX 4: now shows Orth + Div so branch losses are visible during training
            pbar.set_postfix(
                GIoU=f'{mloss[0]:.3f}',
                L1  =f'{mloss[1]:.3f}',
                Spa =f'{mloss[2]:.3f}',
                Quad=f'{mloss[3]:.3f}',
                SCls=f'{mloss[4]:.3f}',
                Rec =f'{mloss[5]:.3f}',
                Orth=f'{mloss[6]:.3f}',
                Div =f'{mloss[7]:.3f}',
                Tot =f'{mloss[8]:.3f}',
                lr  =f'{optimizer.param_groups[0]["lr"]:.5f}',
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

    p.add_argument('--device',  default='2')
    p.add_argument('--workers', type=int, default=4)

    p.add_argument('--epochs',        type=int,   default=150)
    p.add_argument('--batch-size',    type=int,   default=48)
    p.add_argument('--imgsz',         type=int,   default=640)
    p.add_argument('--warmup-epochs', type=int,   default=5)
    p.add_argument('--save-period',   type=int,   default=10)
    p.add_argument('--val-interval',  type=int,   default=5)

    p.add_argument('--project',    default=str(ROOT / 'runs/train'))
    p.add_argument('--name',       default='grounding_v5')
    p.add_argument('--data-root',  default='/data/bxc/DIOR-RSVG/JPEGImages')
    p.add_argument('--xml-root',   default='/data/bxc/DIOR-RSVG/Annotations')
    p.add_argument('--train-txt',  default='/data/bxc/DIOR-RSVG/train.txt')
    p.add_argument('--val-txt',    default='/data/bxc/DIOR-RSVG/val.txt')
    p.add_argument('--text-model', default='./pretrain_ckp/deberta-v3-small')
    p.add_argument('--yolo-weight',default='yolov8s.pt')

    p.add_argument('--max-len',    type=int, default=77)
    p.add_argument('--hidden-dim', type=int, default=256)

    p.add_argument('--min-objects-per-image',   type=int,   default=1)
    p.add_argument('--use-all-images',          action='store_true')
    p.add_argument('--train-spatial-hint-prob', type=float, default=0.0)

    p.add_argument('--lr0',          type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.05)

    p.add_argument('--stage1-epochs',           type=int, default=10)
    p.add_argument('--stage2-offset',           type=int, default=10)
    p.add_argument('--reconstruct-ramp-epochs', type=int, default=10)

    # Loss weights
    p.add_argument('--lambda-spa',         type=float, default=1.0)
    p.add_argument('--lambda-spa-quad',    type=float, default=0.5)
    p.add_argument('--lambda-sem-cls',     type=float, default=1.0)
    p.add_argument('--lambda-reconstruct', type=float, default=0.5)
    p.add_argument('--lambda-orth',        type=float, default=0.3)

    # FIX 5: Stage-1 branch seeding weights (small but non-zero)
    p.add_argument('--lambda-sem-cls-s1',  type=float, default=0.2,
                   help='sem_cls weight in Stage 1 to break branch symmetry from epoch 0')
    p.add_argument('--lambda-spa-quad-s1', type=float, default=0.1,
                   help='spa_quad weight in Stage 1 to break branch symmetry from epoch 0')

    # FIX 6: Branch diversity loss
    p.add_argument('--lambda-div',         type=float, default=0.5,
                   help='Branch diversity loss (pairwise cosine penalty). Active all stages.')

    p.add_argument('--probe-interval', type=int, default=10)
    p.add_argument('--compile',               action='store_true')
    p.add_argument('--freeze-shared-encoder', action='store_true')

    return p.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)