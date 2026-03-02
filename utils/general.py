import time
import numpy as np
import torch
import torchvision
import logging
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

LOGGING_NAME = "RSLIF"
LOGGER = logging.getLogger(LOGGING_NAME)

from utils.metrics import box_iou

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  
):
    """
    通用 NMS 函数
    Input: [Batch, Anchors, 4+Classes] OR [Batch, 4+Classes, Anchors]
    """
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"
    
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0] 

    device = prediction.device
    
    # ✅ 智能转置：如果通道数在第1维 (例如 [B, 24, 8400])，则转置为 [B, 8400, 24]
    # 判断逻辑：通常 Anchors (8400) 远大于 Channels (24)
    if prediction.shape[1] < prediction.shape[2]:
        prediction = prediction.transpose(1, 2)
    
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 4
    mi = 4 + nc
    
    xc = prediction[..., 4:].amax(dim=-1) > conf_thres

    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    multi_label &= nc > 1

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device) for _ in range(bs)]
    
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), 1)
        box = xywh2xyxy(box) # center_x, center_y, w, h -> x1, y1, x2, y2
        mask = x[:, mi:]

        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = 0 if agnostic else x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]
        
        if (time.time() - t) > time_limit:
            break

    return output