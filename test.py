"""
test.py  (rewritten for grounding model)

Uses RemoteSensingVLM from vlm_grounding.py which directly outputs pred_box [B,4].
No NMS, no decode_outputs — the grounding head regresses one box per text query.
"""

import argparse
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from data.dataset import DIORRSVGDataset, rsvlm_collate_fn
from models.vlm_grounding import RemoteSensingVLM


# ──────────────────────────────────────────────────────────────────────────────
# Box utilities
# ──────────────────────────────────────────────────────────────────────────────

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[..., (cx,cy,w,h)] → [..., (x1,y1,x2,y2)]"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def box_iou_single(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """IoU between two [4] xyxy boxes."""
    x1 = max(pred[0], gt[0]);  y1 = max(pred[1], gt[1])
    x2 = min(pred[2], gt[2]);  y2 = min(pred[3], gt[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (pred[2]-pred[0]) * (pred[3]-pred[1])
    a2 = (gt[2]-gt[0])   * (gt[3]-gt[1])
    union = float(a1 + a2 - inter)
    return float(inter) / (union + 1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_grounding(img_tensor, pred_xyxy_px, gt_xyxy_px, text_prompt, iou, save_path):
    """
    img_tensor  : [3, H, W] float32 in [0,1]
    pred_xyxy_px: [4] pixel coords
    gt_xyxy_px  : [4] pixel coords
    """
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.ascontiguousarray((img * 255).astype(np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # GT — green
    x1, y1, x2, y2 = map(int, gt_xyxy_px.tolist())
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(img, "GT", (x1, max(y1-4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    # Pred — red / orange depending on correctness
    px1, py1, px2, py2 = map(int, pred_xyxy_px.tolist())
    color = (0, 180, 0) if iou >= 0.5 else (0, 0, 220)
    cv2.rectangle(img, (px1, py1), (px2, py2), color, 2)
    label = f"IoU={iou:.2f}"
    cv2.putText(img, label, (px1, max(py1-4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Text prompt banner
    top = 36
    img = cv2.copyMakeBorder(img, top, 0, 0, 0, cv2.BORDER_CONSTANT,
                              value=(255, 255, 255))
    disp = text_prompt[:70] + ('…' if len(text_prompt) > 70 else '')
    cv2.putText(img, disp, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.imwrite(str(save_path), img)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test(opt):
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / 'visualization'
    if opt.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f'cuda:{opt.device}' if torch.cuda.is_available() else 'cpu')
    print(f'🚀  Testing on {device}  |  saving to {save_dir}')

    # ── Load checkpoint ──────────────────────────────────────────────
    print(f'⏳  Loading {opt.weights} …')
    checkpoint = torch.load(opt.weights, map_location=device)

    saved_opt    = checkpoint.get('opt', {}) if isinstance(checkpoint.get('opt'), dict) else {}
    deberta_path = saved_opt.get('text_model', opt.text_model)

    model_cfg = {
        'deberta_path':   deberta_path,
        'yolo_weight':    None,
        'hidden_dim':     saved_opt.get('hidden_dim', opt.hidden_dim),
        'backbone_width': 0.5,
        'backbone_depth': 0.33,
    }

    model = RemoteSensingVLM(model_cfg).to(device)

    state_dict = checkpoint.get('model', checkpoint)
    clean_sd   = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)

    if missing:
        print(f'  ⚠️  Missing keys ({len(missing)}): {missing[:5]} …')
    if unexpected:
        print(f'  ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]} …')

    print('✅  Weights loaded.')

    model.eval()

    # ── Dataset ──────────────────────────────────────────────────────
    test_dataset = DIORRSVGDataset(
        data_root=opt.data_root,
        xml_root=opt.xml_root,
        split_txt_path=opt.test_txt,
        tokenizer_path=deberta_path,
        img_size=opt.imgsz,
        max_len=opt.max_len,
        min_objects_per_image=1,
        use_all_images=False,
        spatial_hint_prob=0.0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        collate_fn=rsvlm_collate_fn,
        pin_memory=True,
        persistent_workers=(opt.workers > 0),
    )

    thresholds = (0.25, 0.5, 0.75)
    correct = {t: 0 for t in thresholds}
    total = 0
    sum_iou = 0.0

    pbar = tqdm(test_loader, desc='Testing')

    for batch_i, batch in enumerate(pbar):
        imgs, input_ids, masks, targets, _ = [x.to(device) for x in batch]

        # 自动混合精度推理
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            output = model(imgs, [input_ids, masks])

        pred_box = output['pred_box'].float().clamp(0.0, 1.0)

        B  = imgs.shape[0]
        H, W = imgs.shape[2], imgs.shape[3]

        for si in range(B):
            gt_rows = targets[targets[:, 0].long() == si]
            if len(gt_rows) == 0:
                continue

            total += 1

            gt_norm  = gt_rows[0, 2:6]
            gt_xyxy  = cxcywh_to_xyxy(gt_norm.unsqueeze(0)).squeeze(0)
            gt_xyxy_px = gt_xyxy * torch.tensor([W, H, W, H],
                                                 device=device, dtype=torch.float32)

            pb = pred_box[si]
            pb_xyxy = cxcywh_to_xyxy(pb.unsqueeze(0)).squeeze(0)
            pb_xyxy_px = pb_xyxy * torch.tensor([W, H, W, H],
                                                 device=device, dtype=torch.float32)

            iou = box_iou_single(pb_xyxy_px.cpu(), gt_xyxy_px.cpu())
            sum_iou += iou

            for t in thresholds:
                if iou >= t:
                    correct[t] += 1

            if opt.save_vis:
                text_prompt = test_dataset.tokenizer.decode(
                    input_ids[si].cpu(),
                    skip_special_tokens=True
                )

                status = 'OK' if iou >= 0.5 else 'FAIL'
                img_name = f'idx{batch_i * opt.batch_size + si:06d}_{status}_iou{iou:.2f}.jpg'

                plot_grounding(
                    img_tensor=imgs[si].float(),
                    pred_xyxy_px=pb_xyxy_px.cpu(),
                    gt_xyxy_px=gt_xyxy_px.cpu(),
                    text_prompt=text_prompt,
                    iou=iou,
                    save_path=vis_dir / img_name,
                )

        pbar.set_postfix(
            Acc50=f"{correct[0.5]/(total+1e-7):.4f}",
            mIoU=f"{sum_iou/(total+1e-7):.4f}"
        )

    metrics = {f'Acc@{t}': correct[t] / max(total, 1) for t in thresholds}
    metrics['mIoU'] = sum_iou / max(total, 1)
    metrics['total'] = total

    print('\n📊  Final Results:')
    print(f"   Total samples : {total}")
    for t in thresholds:
        print(f"   Acc@{t:<4}      : {metrics[f'Acc@{t}']:.4f}")
    print(f"   Mean IoU      : {metrics['mIoU']:.4f}")

    results_path = save_dir / 'results.txt'
    with open(results_path, 'w') as f:
        f.write(f"Total samples: {total}\n")
        for t in thresholds:
            f.write(f"Acc@{t}: {metrics[f'Acc@{t}']:.4f}\n")
        f.write(f"Mean IoU: {metrics['mIoU']:.4f}\n")

    print(f"   Results saved to {results_path}")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    type=str,   default='./runs/train/grounding_stage/best.pt')
    parser.add_argument('--data-root',  type=str,   default='/data/bxc/DIOR-RSVG/JPEGImages')
    parser.add_argument('--xml-root',   type=str,   default='/data/bxc/DIOR-RSVG/Annotations')
    parser.add_argument('--test-txt',   type=str,   default='/data/bxc/DIOR-RSVG/test.txt')
    parser.add_argument('--text-model', type=str,   default='./pretrain_ckp/deberta-v3-small')
    parser.add_argument('--imgsz',      type=int,   default=640)
    parser.add_argument('--max-len',    type=int,   default=77)
    parser.add_argument('--hidden-dim', type=int,   default=256)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--device',     default='0')
    parser.add_argument('--project',    default='runs/test')
    parser.add_argument('--name',       default='grounding_stage')
    parser.add_argument('--workers',    type=int,   default=4)
    parser.add_argument('--save-vis',   type=bool, default=False)

    opt = parser.parse_args()
    test(opt)