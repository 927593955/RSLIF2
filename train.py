import argparse
import os
import sys
import yaml
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda import amp
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# --- 1. 路径设置 (Path Setup) ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append("/home/bxc/RSLIF")

# --- 2. 模块导入 ---
from data.dataset import DIORRSVGDataset, rsvlm_collate_fn
from models.fusion_module_GroundingQueryInjector import RemoteSensingVLM
#from models.fusion_model_SpatialRelationReasoner import RemoteSensingVLM
from utils.loss import RSVLMLoss 
from utils.ema import ModelEMA
from val import validate

def get_stage_lambdas(opt, epoch_idx):
        # Stage 1: detection only, stabilize box/class learning
        if epoch_idx < opt.stage1_epochs:
            return {'spa': 0.0, 'sem': 0.0, 'attr': 0.0, 'orth': 0.0, 'rel': 0.0}

        # Stage 2: enable spatial supervision first
        if epoch_idx < opt.stage2_epochs:
            return {'spa': opt.lambda_spa, 'sem': 0.0, 'attr': 0.0, 'orth': 0.0, 'rel': opt.lambda_rel}

        # Stage 3: gradually add semantic/attribute constraints
        #remain = max(opt.epochs - (opt.stage1_epochs + opt.stage2_epochs), 1)
        #prog = min((epoch_idx - (opt.stage1_epochs + opt.stage2_epochs) + 1) / remain, 1.0)
        #ramp = 0.2 + 0.8 * prog
        return {
            'spa': opt.lambda_spa,
            'sem': opt.lambda_sem,# * ramp,
            'attr': opt.lambda_attr,# * ramp,
            'orth': opt.lambda_orth,# * ramp,
            'orth': opt.lambda_orth,# * ramp,
            'rel': opt.lambda_rel# * ramp
        }

def to_fp32(value):
                if torch.is_tensor(value):
                    return value.float()
                if isinstance(value, dict):
                    return {k: to_fp32(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return type(value)(to_fp32(v) for v in value)
                return value


def cosine_lr_factor(progress, lrf=0.01):
    """YOLO-style cosine schedule from 1.0 -> lrf as training progresses."""
    progress = min(max(progress, 0.0), 1.0)
    return lrf + 0.5 * (1.0 - lrf) * (1.0 + math.cos(math.pi * progress))

def train(opt):
    # --- A. 初始化环境 ---
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() and opt.device != 'cpu' else "cpu")
    print(f"Training on {device} | Saving to {save_dir}")
    print(f"Scratch Training Mode | Batch={opt.batch_size}, Epochs={opt.epochs}")

    # --- B. 加载数据 ---
    train_dataset = DIORRSVGDataset(
        data_root=opt.data_root,
        xml_root=opt.xml_root,
        split_txt_path=opt.train_txt,
        tokenizer_path=opt.text_model,
        img_size=opt.imgsz,
        max_len=opt.max_len,
        min_objects_per_image=opt.min_objects_per_image,
        use_all_images=opt.use_all_images,
        spatial_hint_prob=opt.train_spatial_hint_prob
    )
    # 注意：验证集建议稍微少一点 worker，防止内存抢占
    val_dataset = DIORRSVGDataset(
        data_root=opt.data_root,
        xml_root=opt.xml_root,
        split_txt_path=opt.val_txt,
        tokenizer_path=opt.text_model,
        img_size=opt.imgsz, max_len=opt.max_len,
        min_objects_per_image=1,
        use_all_images=False,
        spatial_hint_prob=0.0
    )
    
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, 
                              num_workers=opt.workers, collate_fn=rsvlm_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, 
                            num_workers=max(1, opt.workers//2), collate_fn=rsvlm_collate_fn, pin_memory=True)

    # --- C. 初始化模型 (Train from Scratch) ---
    model_cfg = {
        'deberta_path': opt.text_model,
        'yolo_weight': 'yolov8s.pt',
        'nc': 20,
        'use_relation_reasoner': opt.use_relation_reasoner,
        'use_query_injector': opt.use_query_injector,
        'use_cls_text_bias': opt.use_cls_text_bias
    }
    print("Building model for Scratch Training...")
    model = RemoteSensingVLM(model_cfg).to(device)

    # --- D. 优化器配置 (分层学习率) ---
    text_params = []
    vision_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "text_encoder" in name:
            text_params.append(param)
        else:
            # 包括 vision_encoder, fusion_blocks, head
            vision_params.append(param)

    # AdamW 通常配 1e-3, SGD 配 1e-2
    vision_lr = opt.lr0 
    text_lr = opt.lr0 * 0.1 # 比如 1e-3 * 0.01 = 1e-5
    
    optimizer = optim.AdamW([
        {'params': vision_params, 'lr': vision_lr},
        {'params': text_params,   'lr': text_lr}
    ], weight_decay=opt.weight_decay)   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=opt.epochs, eta_min=1e-6
                )

    print(f"Optimizer Config: Vision LR={vision_lr}, Text LR={text_lr}")

    # --- E. 初始化 Loss ---
    hyp = {
        'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'fl_gamma': 1.0
    }
    loss_calculator = RSVLMLoss(model, device, hyp).to(device)

    # --- F. 初始化 EMA & Scaler ---
    ema = ModelEMA(model)
    scaler = amp.GradScaler(enabled=True)

    # --- G. 训练循环 ---
    # 恢复“检测优先”的权重配置
    #current_lambdas = {
    #    'det': opt.lambda_det,  # 建议 1.0
    #    'spa': opt.lambda_spa,  # 建议 1.0 (不要太高)
    #    'orth': opt.lambda_orth # 建议 0.1
    #}
    
    best_acc = 0.0
    nb = len(train_loader) # number of batches
    
    # Warmup 配置
    nw = max(round(opt.warmup_epochs * nb), 100)  # number of warmup iterations
    accumulate = max(1, round(64 / opt.batch_size))
    for epoch in range(opt.epochs):
        model.train()
        mloss = torch.zeros(9, device=device)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs}", bar_format='{l_bar}{bar:10}{r_bar}')

        current_lambdas = get_stage_lambdas(opt, epoch)

        for batch_i, batch in enumerate(pbar):
            # 1. Warmup & Accumulate 计算
            ni = batch_i + nb * epoch
            
            if ni <= nw:
                # Warmup 学习率调整 (保持之前的线性逻辑)
                for j, x in enumerate(optimizer.param_groups):
                    target_lr = (vision_lr if j == 0 else text_lr)
                    start_lr = 0.1 * target_lr
                    current_lr = start_lr + (target_lr - start_lr) * (ni / nw)
                    x['lr'] = current_lr

            # 2. Forward
            imgs, input_ids, masks, yolo_targets, spa_targets = [item.to(device, non_blocking=True) for item in batch]
            
            with amp.autocast(enabled=True):
                model_output = model(imgs, [input_ids, masks])

            with amp.autocast(enabled=False):
                model_output_fp32 = to_fp32(model_output)
                # ✅ 修改点 1: 简化调用，让 Loss 内部去解包字典
                loss, loss_dict = loss_calculator(
                    preds=model_output_fp32,          # 传入包含所有分支输出的字典
                    batch_targets=yolo_targets,  # GT Bbox
                    lambdas=current_lambdas,     # 权重 (虽然目前Loss里硬编码了，但保留接口)
                    imgsz=opt.imgsz 
                )

            scaler.scale(loss).backward()

            # 4. Optimizer Step (梯度累积核心逻辑)
            if (batch_i + 1) % accumulate == 0 or (batch_i + 1) == len(train_loader):
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 建议设为 1.0
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # 💡 移动到这里，累积周期结束后再清空
                if ema: 
                    ema.update(model)

            # 3. Log
            with torch.no_grad():
                # ✅ 修改点 2: 更新日志 Keys 以匹配新的 Loss 组件
                # 对应 loss.py 返回的: 'box', 'cls', 'dfl', 't_spa', 'align', 'orth', 'total'
                loss_keys = ['box', 'cls', 'dfl', 't_spa', 'sem', 'attr', 'orth', 'rel', 'total']
                
                # 确保 loss_dict 里有这些 key (防止第一步里没算某些 loss 报错)
                current_loss_items = [loss_dict.get(k, 0.0) for k in loss_keys]
                loss_items = torch.tensor(current_loss_items, device=device)
                
                # 平滑更新
                mloss = (mloss * batch_i + loss_items) / (batch_i + 1)
                
                # ✅ 修改点 3: 更新进度条显示，简写一下名字以防太长
                pbar.set_postfix(
                    Box=f"{mloss[0]:.2f}", 
                    Cls=f"{mloss[1]:.2f}", 
                    Dfl=f"{mloss[2]:.2f}",
                    TSpa=f"{mloss[3]:.2f}",    # Text Spatial Loss
                    Sem=f"{mloss[4]:.2f}",     # Alignment Loss
                    Att=f"{mloss[5]:.2f}",     # Attention Loss
                    Ort=f"{mloss[6]:.2f}",     # Orthogonal Loss
                    T=f"{mloss[8]:.2f}",       # Total
                    lr=f"{optimizer.param_groups[0]['lr']:.5f}"
                )

            del imgs, input_ids, masks, loss

        # --- Validation ---
        if (epoch+1) % 5 == 0 or (epoch+1) == opt.epochs or epoch<5:
            if val_loader:
                torch.cuda.empty_cache()
                tqdm.write(f"Validating epoch {epoch+1}...")
                val_model = ema.ema if ema else model
                
                # Scratch 训练初期 Acc 可能是 0，不要灰心，看 Box Loss
                p, r, f1, acc = validate(
                    val_model,
                    val_loader,
                    device,
                    conf_thres=opt.val_conf_thres,
                    iou_thres=opt.val_iou_thres,
                    class_agnostic=opt.val_class_agnostic
                )
                
                print(f"Epoch {epoch+1}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, Acc={acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    ckpt = {
                        'epoch': epoch,
                        'model': deepcopy(val_model).half().state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': best_acc,
                    }
                    torch.save(ckpt, save_dir / "best.pt")
                    tqdm.write(f"Best saved: {best_acc:.4f}")

        # Regular Save
        if opt.save_period > 0 and (epoch + 1) % opt.save_period == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict()}, save_dir / f"last.pt")
        
        scheduler.step()
            
    print("✅ Training Completed.")

# ===============================================================
# 4. 参数解析 (Updated Defaults)
# ===============================================================
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='2')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--save-period', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--imgsz', type=int, default=640)

    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='GroundingQueryInjector-pretrained', help='save name')
    #parser.add_argument('--name', default='SpatialRelationReasoner-vert', help='save name')
    parser.add_argument('--data-root', type=str, default='/data/bxc/DIOR-RSVG/JPEGImages')
    parser.add_argument('--xml-root', type=str, default='/data/bxc/DIOR-RSVG/Annotations')
    parser.add_argument('--train_txt', type=str, default='/data/bxc/DIOR-RSVG/train.txt')
    parser.add_argument('--val_txt', type=str, default='/data/bxc/DIOR-RSVG/val.txt')
    parser.add_argument('--text-model', type=str, default='./pretrain_ckp/deberta-v3-small')

    parser.add_argument('--max-len', type=int, default=77, help='max token length')
    parser.add_argument('--min_objects_per-image', type=int, default=1,
                        help='minimum valid objects required per image when building dataset')
    parser.add_argument('--use_all_images', action='store_true',
                        help='ignore split file ids and use all XML files under --xml-root')
    parser.add_argument('--train_spatial_hint_prob', type=float, default=0.0,
                        help='probability to append continuous spatial metadata prompt during training (default off)')
    parser.add_argument('--lr0', type=float, default=1e-3, help='Vision LR (Text will be 0.01*LR)')
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')

    parser.add_argument('--lambda-det', type=float, default=1.0, help='Detection Loss Weight')
    parser.add_argument('--lambda-spa', type=float, default=2.0, help='Spatial Loss Weight')
    parser.add_argument('--lambda-sem', type=float, default=0.5, help='Semantic Alignment Loss Weight')
    parser.add_argument('--lambda-attr', type=float, default=0.5, help='Attribute Alignment Loss Weight')
    parser.add_argument('--lambda-orth', type=float, default=0.5, help='Orthogonal Loss Weight')
    parser.add_argument('--lambda-rel', type=float, default=0.5, help='Relative Position Loss Weight')
    parser.add_argument('--stage1-epochs', type=int, default=0, help='Epochs for detection-only training')
    parser.add_argument('--stage2-epochs', type=int, default=0, help='Epochs for detection + spatial training')


    parser.add_argument('--val-conf-thres', type=float, default=0.25, help='validation confidence threshold')
    parser.add_argument('--val-iou-thres', type=float, default=0.6, help='validation NMS IoU threshold')
    parser.add_argument('--val-class-agnostic', action='store_true', help='use class-agnostic NMS in validation')
    parser.add_argument('--use-relation-reasoner', action='store_true', help='enable explicit spatial relation reasoner')
    parser.add_argument('--use-query-injector', action='store_true', help='enable grounding query injector')
    parser.add_argument('--use-cls-text-bias', action='store_true', help='enable semantic class bias injection')

    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    train(opt)