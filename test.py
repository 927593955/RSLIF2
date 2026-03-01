import argparse
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

# --- 核心模块导入 ---
from data.dataset import DIORRSVGDataset, rsvlm_collate_fn
from models.fusion_module_GroundingQueryInjector import RemoteSensingVLM
#from models.fusion_model_SpatialRelationReasoner import RemoteSensingVLM
# 使用 ultralytics 官方工具 (如果已安装) 或 本地 utils
from utils.general import non_max_suppression, xywh2xyxy
from utils.metrics import box_iou
from utils.detection import decode_outputs

# 设置 Matplotlib 后端
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# 2. 可视化工具
# ==============================================================================
def plot_visual_grounding(img_tensor, pred_boxes, gt_boxes, text_prompt, save_path):
    # 1. 图像反归一化
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.ascontiguousarray((img * 255).astype(np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 2. 绘制 Ground Truth (绿色)
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "GT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 3. 绘制 Prediction (红色)
    if pred_boxes is not None and len(pred_boxes) > 0:
        box = pred_boxes[0]
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"Pred {conf:.2f}"
        
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), (0, 0, 255), -1)
        cv2.putText(img, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 4. 添加文本
    top_border = 40
    img_padded = cv2.copyMakeBorder(img, top_border, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    display_text = text_prompt[:60] + "..." if len(text_prompt) > 60 else text_prompt
    cv2.putText(img_padded, display_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(save_path), img_padded)

# ==============================================================================
# 3. 测试主程序
# ==============================================================================
@torch.no_grad()
def test(opt):
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / "visualization"
    if opt.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Testing on {device} | 📂 Saving results to {save_dir}")

    # --- 加载模型 ---
    print(f"⏳ Loading checkpoint from {opt.weights}...")
    checkpoint = torch.load(opt.weights, map_location=device)
    
    if 'opt' in checkpoint and isinstance(checkpoint['opt'], dict):
        saved_opt = checkpoint['opt']
        deberta_path = saved_opt.get('text_model', opt.text_model)
    else:
        deberta_path = opt.text_model

    model_cfg = {'deberta_path': deberta_path, 'yolo_weight': None}
    model = RemoteSensingVLM(model_cfg).to(device)
    
    state_dict = checkpoint.get('model', checkpoint)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    try:
        model.load_state_dict(new_state_dict)
        print("✅ Model weights loaded successfully.")
    except Exception as e:
        print(f"⚠️ Strict loading failed, trying strict=False...")
        model.load_state_dict(new_state_dict, strict=False)
        
    model.eval()

    # --- 数据集 ---
    test_dataset = DIORRSVGDataset(
        data_root=opt.data_root, xml_root=opt.xml_root,
        split_txt_path=opt.test_txt, tokenizer_path=deberta_path,
        img_size=opt.imgsz, max_len=opt.max_len,
        min_objects_per_image=1,
        use_all_images=False,
        spatial_hint_prob=0.0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, collate_fn=rsvlm_collate_fn, pin_memory=True
    )

    correct_count = 0
    total_count = 0
    pbar = tqdm(test_loader, desc="Testing")
    
    saved_count = 0
    
    for batch_i, batch in enumerate(pbar):
        imgs, input_ids, masks, targets, spa_gt = [item.to(device) for item in batch]
        
        # 1. Forward
        output = model(imgs, [input_ids, masks])
        if isinstance(output, dict):
            preds_raw = output['feats']
        elif isinstance(output, (list, tuple)):
             preds_raw = output[0] if isinstance(output[0], list) else output
        else:
             preds_raw = output

        # 2. ✅【关键】解码 Raw Output -> Decoded Boxes
        preds_decoded = decode_outputs(preds_raw, model)

        # 3. NMS
        # conf_thres 极低，确保总有框出来，防止漏检
        preds = non_max_suppression(
            preds_decoded,
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres,
            max_det=300
        )
        
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si]
            if len(labels) == 0: continue
                
            total_count += 1
            
            h, w = imgs.shape[2:]
            tbox = labels[:, 2:].clone()
            tbox[:, 0] *= w; tbox[:, 2] *= w
            tbox[:, 1] *= h; tbox[:, 3] *= h
            gt_xyxy = xywh2xyxy(tbox) 
            
            is_correct = False
            
            if len(pred) > 0:
                sorted_pred = pred[pred[:, 4].argsort(descending=True)]
                top1_box = sorted_pred[0, :4].unsqueeze(0)
                
                # 计算 IoU
                ious = box_iou(top1_box, gt_xyxy)
                max_iou = ious.max().item()
                
                if max_iou >= 0.5:
                    correct_count += 1
                    is_correct = True
            
            # --- 可视化策略 ---
            # 优先保存正确的，如果没有正确的，保存少量错误的用于 debug
            if opt.save_vis:
                # 1. 强制设为 True，保存所有结果
                save_this = True 
                
                if save_this:
                    saved_count += 1
                    curr_input_ids = input_ids[si]
                    text_prompt = test_dataset.tokenizer.decode(curr_input_ids, skip_special_tokens=True)
                    
                    # 2. 修改文件名命名规则
                    # 因为要存所有图片，建议在文件名加入更详细的信息，防止重名
                    # 同时保留 OK/FAIL 标签方便你后续筛选查看
                    status = 'OK' if is_correct else 'FAIL'
                    img_name = f"idx{batch_i * opt.batch_size + si}_{status}.jpg"
                    
                    # 3. 绘图调用保持不变
                    plot_visual_grounding(
                        img_tensor=imgs[si],
                        # 即使 pred 为空，plot 函数也能处理 None
                        pred_boxes=pred if len(pred) > 0 else None, 
                        gt_boxes=gt_xyxy,
                        text_prompt=text_prompt,
                        save_path=vis_dir / img_name
                    )
        
        current_acc = correct_count / (total_count + 1e-7)
        pbar.set_postfix(Acc=f"{current_acc:.4f}")

    final_acc = correct_count / (total_count + 1e-7)
    print(f"\nFinal Results:")
    print(f"Total Samples: {total_count}")
    print(f"Acc@0.5: {final_acc:.4f}")
    
    with open(save_dir / 'results.txt', 'w') as f:
        f.write(f"Total Samples: {total_count}\n")
        f.write(f"Acc@0.5: {final_acc:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./runs/train/GroundingQueryInjector/best.pt', help='weights path')
    parser.add_argument('--data-root', type=str, default='/data/bxc/DIOR-RSVG/JPEGImages')
    parser.add_argument('--xml-root', type=str, default='/data/bxc/DIOR-RSVG/Annotations')
    parser.add_argument('--test-txt', type=str, default='/data/bxc/DIOR-RSVG/test.txt')
    parser.add_argument('--text-model', type=str, default='./pretrain_ckp/deberta-v3-small')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--max-len', type=int, default=77)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold for NMS')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', default='runs/test')
    parser.add_argument('--name', default='GroundingQueryInjector')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save-vis', action='store_true', help='save visualization')
    
    opt = parser.parse_args()
    opt.save_vis = True 
    test(opt)