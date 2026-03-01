import torch
import numpy as np
from tqdm import tqdm

def box_iou(box1, box2):
    """
    计算两组框的 IoU (用于验证)
    box1: [N, 4] (x1, y1, x2, y2)
    box2: [M, 4] (x1, y1, x2, y2)
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = box_area(box1)
    area2 = box_area(box2)

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union

def process_batch(detections, labels, iou_thres):
    """
    计算 TP 矩阵
    """
    correct = torch.zeros(detections.shape[0], dtype=torch.bool, device=detections.device)
    if len(labels) == 0:
        return correct
    
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where(iou > iou_thres)
    
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).detach().cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(detections.device)
        
        # 这里的 [:, 5] 是类别索引。因为我们在 validate 里强制设为 0 了，所以这里相当于只比较 IoU
        correct[matches[:, 1].long()] = (detections[matches[:, 1].long(), 5] == labels[matches[:, 0].long(), 0])
        
    return correct