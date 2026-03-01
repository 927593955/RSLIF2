"""
train_grounding.py

Training script for the Visual Grounding model.

Key changes vs old train.py:
1. Uses RemoteSensingVLM from vlm_grounding.py (grounding head)
2. Uses GroundingLoss instead of RSVLMLoss (YOLO detection loss)
3. Simpler loss items display (no DFL/cls/box confusion)
4. Validation calls updated val.validate()
5. Staged lambda scheduling still available but simplified
"""

import argparse
import math
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
from val import validate


# ──────────────────────────────────────────────────────────────────────────────
# Lambda schedule
# ──────────────────────────────────────────────────────────────────────────────

def get_lambdas(opt, epoch):
    """
    Stage 1 (< stage1_epochs): localization only (giou + l1)
    Stage 2 (>= stage1_epochs): add spa + orth constraints
    """
    if epoch < opt.stage1_epochs:
        return {'giou': 2.0, 'l1': 5.0, 'spa': 0.0, 'orth': 0.0}
    return {
        'giou': 2.0,
        'l1':   5.0,
        'spa':  opt.lambda_spa,
        'orth': opt.lambda_orth,
    }


# ──────────────────────────────────────────────────────────────────────────────
# FP32 cast utility (for AMP mixed-precision)
# ──────────────────────────────────────────────────────────────────────────────

def to_fp32(x):
    if torch.is_tensor(x):
        return x.float()
    if isinstance(x, dict):
        return {k: to_fp32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_fp32(v) for v in x)
    return x


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(opt):
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        f'cuda:{opt.device}' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu'
    )
    print(f'Training on {device} | Saving to {save_dir}')

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
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers, collate_fn=rsvlm_collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=max(1, opt.workers // 2), collate_fn=rsvlm_collate_fn, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = {
        'deberta_path':  opt.text_model,
        'yolo_weight':   opt.yolo_weight,
        'hidden_dim':    opt.hidden_dim,
        'backbone_width': 0.5,
        'backbone_depth': 0.33,
    }
    model = RemoteSensingVLM(model_cfg).to(device)

    # ── Optimizer (split LR: lower for pretrained text encoder) ───────────────
    text_params   = [p for n, p in model.named_parameters() if 'text_encoder' in n and p.requires_grad]
    vision_params = [p for n, p in model.named_parameters() if 'text_encoder' not in n and p.requires_grad]

    optimizer = optim.AdamW([
        {'params': vision_params, 'lr': opt.lr0},
        {'params': text_params,   'lr': opt.lr0 * 0.1},
    ], weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.epochs, eta_min=1e-6
    )

    # ── Loss & EMA & Scaler ───────────────────────────────────────────────────
    loss_fn = GroundingLoss(
        lambda_giou=2.0,
        lambda_l1=5.0,
        lambda_spa=opt.lambda_spa,
        lambda_orth=opt.lambda_orth,
    )
    ema    = ModelEMA(model)
    scaler = amp.GradScaler(enabled=True)

    # ── Warmup config ─────────────────────────────────────────────────────────
    nb = len(train_loader)
    nw = max(round(opt.warmup_epochs * nb), 100)
    accumulate = max(1, round(64 / opt.batch_size))

    best_acc = 0.0

    for epoch in range(opt.epochs):
        model.train()
        lambdas = get_lambdas(opt, epoch)
        mloss = torch.zeros(5, device=device)  # giou, l1, spa, orth, total

        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{opt.epochs}',
            bar_format='{l_bar}{bar:10}{r_bar}'
        )

        for batch_i, batch in enumerate(pbar):
            ni = batch_i + nb * epoch

            # Warmup LR
            if ni <= nw:
                for j, x in enumerate(optimizer.param_groups):
                    base = opt.lr0 if j == 0 else opt.lr0 * 0.1
                    x['lr'] = 0.1 * base + (base - 0.1 * base) * (ni / nw)

            imgs, input_ids, masks, yolo_targets, _ = [
                x.to(device, non_blocking=True) for x in batch
            ]

            # Forward
            with amp.autocast(enabled=True):
                preds = model(imgs, [input_ids, masks])

            # Loss (in fp32 for numerical stability)
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

            # Logging
            with torch.no_grad():
                items = torch.tensor(
                    [loss_dict.get(k, 0.0) for k in ('giou', 'l1', 'spa', 'orth', 'total')],
                    device=device
                )
                mloss = (mloss * batch_i + items) / (batch_i + 1)

            pbar.set_postfix(
                GIoU=f'{mloss[0]:.3f}',
                L1=f'{mloss[1]:.3f}',
                Spa=f'{mloss[2]:.3f}',
                Ort=f'{mloss[3]:.3f}',
                Tot=f'{mloss[4]:.3f}',
                lr=f'{optimizer.param_groups[0]["lr"]:.5f}'
            )

            del imgs, input_ids, masks, loss

        # ── Validation ────────────────────────────────────────────────────────
        do_val = ((epoch + 1) % opt.val_interval == 0
                  or (epoch + 1) == opt.epochs
                  or epoch < 5)
        if do_val:
            torch.cuda.empty_cache()
            val_model = ema.ema if ema else model
            metrics = validate(val_model, val_loader, device, verbose=True)

            acc5 = metrics['acc@0.5']
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
                tqdm.write(f'✅ Best saved: Acc@0.5={best_acc:.4f}')

        # Regular checkpoint
        if opt.save_period > 0 and (epoch + 1) % opt.save_period == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict()},
                       save_dir / 'last.pt')

        scheduler.step()

    print('✅ Training completed.')


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_opt():
    p = argparse.ArgumentParser()

    # Hardware
    p.add_argument('--device',  default='0')
    p.add_argument('--workers', type=int, default=8)

    # Training schedule
    p.add_argument('--epochs',       type=int,   default=150)
    p.add_argument('--batch-size',   type=int,   default=2)
    p.add_argument('--imgsz',        type=int,   default=640)
    p.add_argument('--warmup-epochs',type=int,   default=5)
    p.add_argument('--save-period',  type=int,   default=10)
    p.add_argument('--val-interval', type=int,   default=5)

    # Paths
    p.add_argument('--project',    default=str(ROOT / 'runs/train'))
    p.add_argument('--name',       default='grounding')
    p.add_argument('--data-root',  default='/data/bxc/DIOR-RSVG/JPEGImages')
    p.add_argument('--xml-root',   default='/data/bxc/DIOR-RSVG/Annotations')
    p.add_argument('--train-txt',  default='/data/bxc/DIOR-RSVG/train.txt')
    p.add_argument('--val-txt',    default='/data/bxc/DIOR-RSVG/val.txt')
    p.add_argument('--text-model', default='./pretrain_ckp/deberta-v3-small')
    p.add_argument('--yolo-weight',default='yolov8s.pt')

    # Model
    p.add_argument('--max-len',   type=int,   default=77)
    p.add_argument('--hidden-dim',type=int,   default=256,
                   help='Grounding head hidden dimension')

    # Dataset
    p.add_argument('--min-objects-per-image', type=int,   default=1)
    p.add_argument('--use-all-images',        action='store_true')
    p.add_argument('--train-spatial-hint-prob',type=float, default=0.0)

    # Optimizer
    p.add_argument('--lr0',          type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.05)

    # Loss weights
    p.add_argument('--lambda-spa',  type=float, default=1.0)
    p.add_argument('--lambda-orth', type=float, default=0.3)
    p.add_argument('--stage1-epochs', type=int, default=0,
                   help='Epochs with spa/orth losses disabled (localization warmup)')

    return p.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)