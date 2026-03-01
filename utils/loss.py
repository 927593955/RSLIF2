import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. 基础工具函数
# =============================================================================
def bbox_iou(box1, box2, xywh=True, CIoU=False):
    if xywh:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:  # x1y1x2y2
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-16
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-16
    union = w1 * h1 + w2 * h2 - inter + 1e-16
    iou = inter / union

    if CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + 1e-16
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        v = (4 / (3.1415926535 ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + 1e-16))
        return iou - (rho2 / c2 + v * alpha)
    return iou

def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

# =============================================================================
# 2. 对齐损失组件 (修复数值稳定性)
# =============================================================================
class AlignmentLoss(nn.Module):
    """
    视觉-文本对比损失 (Contrastive Loss)
    增加 AMP 稳定性保护
    """
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp
        self.ce = nn.CrossEntropyLoss()

    def forward(self, visual_feats, text_feats, gate=None):
        """
        visual_feats: [N, D]
        text_feats:   [N, D]
        """
        # 1. 空数据保护
        if visual_feats.shape[0] == 0:
            return torch.tensor(0.0, device=visual_feats.device)

        # 2. 输入数据清洗 (防止脏数据进入导致 NaN)
        if torch.isnan(visual_feats).any() or torch.isnan(text_feats).any():
            # 这种情况下返回 0，避免破坏整个训练
            return torch.tensor(0.0, device=visual_feats.device)

        # 3. 归一化 (增加 eps)
        v_norm = F.normalize(visual_feats, dim=-1, eps=1e-6)
        t_norm = F.normalize(text_feats, dim=-1, eps=1e-6)
        
        # 4. 计算相似度 logits
        #logits = torch.matmul(v_norm, t_norm.T) / self.temp
        logits = torch.matmul(v_norm, t_norm.T) / max(self.temp, 0.07)
        logits = torch.clamp(logits, max=80.0)  # 80 是比较极限的安全值 (FP32), FP16 建议更低
        
        labels = torch.arange(logits.size(0), device=logits.device)
        
        if gate is not None:
            gate = gate.squeeze()
            if gate.ndim == 0: gate = gate.unsqueeze(0)
            
            # 使用 reduction='none' 获得每个样本的 loss
            loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
            
            # 确保 gate 数值正常
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0)
            
            loss = (loss_per_sample * gate).mean()
        else:
            loss = self.ce(logits, labels)

        # 5. 最终输出检查
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=visual_feats.device)
            
        return loss

# =============================================================================
# 3. 核心模块: BboxLoss & Assigner
# =============================================================================
class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        if anchor_points.ndim == 2:
            batch_size = pred_dist.shape[0]
            anchor_points = anchor_points.unsqueeze(0).repeat(batch_size, 1, 1)

        weight = target_scores.sum(-1)[fg_mask]
        if weight.numel() == 0 or target_scores_sum <= 0:
            zero = torch.tensor(0.0, device=pred_dist.device)
            return zero, zero
        
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        if iou.ndim > weight.ndim:
            iou = iou.squeeze(-1)
        
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.use_dfl:
            target_ltrb = torch.cat((anchor_points[fg_mask] - target_bboxes[fg_mask][:, :2], 
                                     target_bboxes[fg_mask][:, 2:] - anchor_points[fg_mask]), 1)
            target_ltrb = target_ltrb.clamp(min=0, max=self.reg_max - 0.01)
            
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb) 
            loss_dfl = (loss_dfl * weight.unsqueeze(-1)).sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl = tr.float() - target
        wr = 1 - wl
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)

class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        batch_size = pd_scores.size(0)
        num_anchors = pd_scores.size(1)
        
        target_labels = torch.full((batch_size, num_anchors), self.bg_idx, dtype=torch.long, device=pd_scores.device)
        target_bboxes = torch.zeros((batch_size, num_anchors, 4), device=pd_scores.device)
        target_scores = torch.zeros((batch_size, num_anchors, self.num_classes), device=pd_scores.device)
        fg_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=pd_scores.device)

        for i in range(batch_size):
            if mask_gt[i].sum() == 0: continue
            
            cur_gt_labels = gt_labels[i][mask_gt[i].squeeze(-1)].long().squeeze(-1)
            cur_gt_bboxes = gt_bboxes[i][mask_gt[i].squeeze(-1)]
            
            bbox_scores = pd_scores[i]
            scores = bbox_scores[:, cur_gt_labels]
            
            ious = bbox_iou(cur_gt_bboxes.unsqueeze(0), pd_bboxes[i].unsqueeze(1), xywh=False, CIoU=False).squeeze(0)
            align_metric = scores.pow(self.alpha) * ious.pow(self.beta)
            
            lt = anc_points.unsqueeze(1) - cur_gt_bboxes[:, :2].unsqueeze(0)
            rb = cur_gt_bboxes[:, 2:].unsqueeze(0) - anc_points.unsqueeze(1)
            is_in_gts = torch.cat([lt, rb], dim=-1).min(dim=-1)[0] > 0
            
            align_metric_filter = align_metric * is_in_gts
            topk_val, topk_ind = torch.topk(align_metric_filter, self.topk, dim=0)
            mask_topk = torch.zeros_like(align_metric_filter, dtype=torch.bool).scatter_(0, topk_ind, True)
            mask_pos = mask_topk * is_in_gts
            
            if mask_pos.sum() > 0:
                fg_mask_i = mask_pos.sum(1) > 0
                if fg_mask_i.sum() > 0:
                    target_labels[i][fg_mask_i] = cur_gt_labels[align_metric_filter[fg_mask_i].argmax(1)]
                    target_bboxes[i][fg_mask_i] = cur_gt_bboxes[align_metric_filter[fg_mask_i].argmax(1)]
                    
                    align_metric_max = align_metric_filter[fg_mask_i].max(1)[0].clamp(min=0)
                    norm_align_metric = (align_metric_max - align_metric_max.min()) / (align_metric_max.max() - align_metric_max.min() + 1e-9)
                    
                    target_scores[i, fg_mask_i, target_labels[i][fg_mask_i]] = norm_align_metric
                    fg_mask[i] = fg_mask_i

        return target_labels, target_bboxes, target_scores, fg_mask

# =============================================================================
# 4. Relative position GT builder
# =============================================================================
def build_gt_relation_vectors(batch_targets, batch_size, device):
    """Build GT relative_position=[dx, dy, orientation] from precise GT boxes.
    Uses first GT as reference and the farthest GT as context when available.
    Fallback: relation to image center.
    """
    gt_rel = torch.zeros((batch_size, 3), device=device)
    valid = torch.zeros((batch_size,), dtype=torch.bool, device=device)

    for i in range(batch_size):
        t = batch_targets[batch_targets[:, 0] == i]
        if t.shape[0] == 0:
            continue

        boxes = t[:, 2:6]  # normalized xywh
        cx = boxes[:, 0]
        cy = boxes[:, 1]

        ref_cx, ref_cy = cx[0], cy[0]
        if boxes.shape[0] >= 2:
            d2 = (cx - ref_cx) ** 2 + (cy - ref_cy) ** 2
            d2[0] = -1.0
            j = int(torch.argmax(d2).item())
            ctx_cx, ctx_cy = cx[j], cy[j]
        else:
            ctx_cx = torch.tensor(0.5, device=device)
            ctx_cy = torch.tensor(0.5, device=device)

        dx = (ctx_cx - ref_cx).clamp(-1.0, 1.0)
        dy = (ctx_cy - ref_cy).clamp(-1.0, 1.0)
        orientation = torch.atan2(dy, dx) / torch.pi  # [-1, 1]

        gt_rel[i] = torch.stack([dx, dy, orientation])
        valid[i] = True

    return gt_rel, valid

# =============================================================================
# 4. 主 Loss 类: RSVLMLoss
# =============================================================================
class RSVLMLoss(nn.Module):
    def __init__(self, model, device, hyp=None):
        super().__init__()
        self.device = device
        self.hyp = hyp or {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}
        
        head = model.head
        self.nc = head.nc
        self.no = head.no
        self.reg_max = head.reg_max
        self.stride = head.stride.to(device)

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max, use_dfl=self.reg_max > 1).to(device)
        self.proj = torch.arange(self.reg_max + 1, dtype=torch.float, device=device)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        self.align_loss = AlignmentLoss(temp=0.1) 
        self.mse_loss = nn.MSELoss() 

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)
        
        i = targets[:, 0].int()
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
        
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                box_norm = targets[matches, 2:]
                box_scaled = box_norm * scale_tensor[[0, 1, 0, 1]]
                
                xyxy = torch.zeros_like(box_scaled)
                xyxy[:, 0] = box_scaled[:, 0] - box_scaled[:, 2] / 2
                xyxy[:, 1] = box_scaled[:, 1] - box_scaled[:, 3] / 2
                xyxy[:, 2] = box_scaled[:, 0] + box_scaled[:, 2] / 2
                xyxy[:, 3] = box_scaled[:, 1] + box_scaled[:, 3] / 2
                
                out[j, :n, 0] = targets[matches, 1]
                out[j, :n, 1:] = xyxy
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.reg_max > 1:
            b, a, c = pred_dist.shape
            pred_dist = torch.nan_to_num(pred_dist, nan=0.0, posinf=50.0, neginf=-50.0)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def extract_visual_embeds(self, feats, gt_bboxes, stride_tensor):
        """Return ROI-mean embeddings and a valid-image mask."""
        last_feat = feats[-1]  # [B, C, H, W]
        B, C, H, W = last_feat.shape
        stride = float(self.stride[-1].item())

        visual_embeds = []
        valid_mask = []

        for i in range(B):
            if gt_bboxes[i].sum() == 0:
                visual_embeds.append(torch.zeros(C, device=self.device))
                valid_mask.append(False)
                continue

            box = gt_bboxes[i][0]  # xyxy on image pixels
            x1 = int(torch.floor(box[0] / stride).clamp(0, W - 1).item())
            y1 = int(torch.floor(box[1] / stride).clamp(0, H - 1).item())
            x2 = int(torch.ceil(box[2] / stride).clamp(0, W - 1).item())
            y2 = int(torch.ceil(box[3] / stride).clamp(0, H - 1).item())

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            roi = last_feat[i, :, y1:y2 + 1, x1:x2 + 1]
            if roi.numel() == 0:
                cx = int((((box[0] + box[2]) / 2) / stride).long().clamp(0, W - 1).item())
                cy = int((((box[1] + box[3]) / 2) / stride).long().clamp(0, H - 1).item())
                emb = last_feat[i, :, cy, cx]
            else:
                emb = roi.mean(dim=(1, 2))

            visual_embeds.append(emb)
            valid_mask.append(True)

        return torch.stack(visual_embeds), torch.tensor(valid_mask, device=self.device, dtype=torch.bool)

    def forward(self, preds, batch_targets, lambdas=None, imgsz=640, **kwargs):
        head_output = preds['feats'] if isinstance(preds, dict) else preds
        visual_feats_list = preds.get('visual_feats', None) 
        
        # 1. 基础准备
        shape = head_output[0].shape
        cat_x = [xi.view(shape[0], self.no, -1) for xi in head_output]
        pred_distri, pred_scores = torch.cat(cat_x, 2).split(((self.reg_max + 1) * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous().float()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous().float()
        pred_scores = pred_scores.clamp(min=-50.0, max=50.0)
        pred_distri = pred_distri.clamp(min=-50.0, max=50.0)
        pred_scores = torch.nan_to_num(pred_scores, nan=0.0, posinf=50.0, neginf=-50.0)
        pred_distri = torch.nan_to_num(pred_distri, nan=0.0, posinf=50.0, neginf=-50.0)
        
        batch_size = pred_scores.shape[0]
        img_h, img_w = shape[2] * self.stride[0], shape[3] * self.stride[0]
        anchor_points, stride_tensor = make_anchors(head_output, self.stride, 0.5)
        anchor_points = anchor_points.float()
        stride_tensor = stride_tensor.float()
        
        # 2. GT 处理
        targets = self.preprocess(batch_targets, batch_size, torch.tensor([img_w, img_h, img_w, img_h], device=self.device))
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt(0)
        
        # 3. Decode & Assign
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            (anchor_points * stride_tensor).type(gt_bboxes.dtype),
            gt_labels, gt_bboxes, mask_gt
        )
        target_scores_sum = target_scores.sum().clamp(min=1.0)
        
        target_bboxes /= stride_tensor.unsqueeze(0)
        
        # 4. 计算检测 Loss
        loss_cls_raw = self.bce(pred_scores, target_scores)
        pred_prob = pred_scores.sigmoid()
        alpha = 0.25 
        gamma = 2.0

        # Stabilized focal weighting: keep positives normalized by positive quality,
        # and negatives normalized by anchor count to avoid cls explosion.
        focal_weight = torch.where(
            target_scores > 0,
            (target_scores - pred_prob).abs().pow(gamma),
            pred_prob.pow(gamma)
        )

        pos_mask = (target_scores > 0).float()
        neg_mask = 1.0 - pos_mask

        pos_norm = target_scores_sum.clamp(min=1.0)
        neg_norm = torch.tensor(float(pred_scores.shape[0] * pred_scores.shape[1]), device=self.device).clamp(min=1.0)

        loss_cls_pos = (loss_cls_raw * focal_weight * pos_mask).sum() / pos_norm
        loss_cls_neg = (loss_cls_raw * focal_weight * neg_mask).sum() / neg_norm
        loss_cls = loss_cls_pos + loss_cls_neg

        if torch.isnan(loss_cls) or torch.isinf(loss_cls):
            loss_cls = torch.tensor(0.0, device=self.device)


        loss_box, loss_dfl = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)
        if fg_mask.sum():
            loss_box, loss_dfl = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, 
                target_scores, target_scores_sum, fg_mask
            )
            if torch.isnan(loss_box) or torch.isinf(loss_box):
                loss_box = torch.tensor(0.0, device=self.device)
            if torch.isnan(loss_dfl) or torch.isinf(loss_dfl):
                loss_dfl = torch.tensor(0.0, device=self.device)

        # =====================================================================
        # 5. 辅助分支约束 Loss (修复 NaN 问题)
        # =====================================================================
        loss_txt_spa = torch.tensor(0., device=self.device)
        loss_sem_align = torch.tensor(0., device=self.device)
        loss_attr_align = torch.tensor(0., device=self.device)
        loss_orth = torch.tensor(0., device=self.device)
        loss_rel_pos = torch.tensor(0., device=self.device)

        if isinstance(preds, dict) and 'sem_vec' in preds:
            # --- A. 空间分支约束 (Text -> Box) 修复版 ---
            valid_indices = []
            gt_boxes_list = []
            
            for i in range(batch_size):
                # 检查这张图有没有真正的物体
                t = batch_targets[batch_targets[:, 0] == i]
                if len(t) > 0:
                    # 只收集有物体的 GT
                    gt_boxes_list.append(t[0, 2:]) # [x, y, w, h] 归一化坐标
                    valid_indices.append(i)
            
            if len(valid_indices) > 0:
                # 只对有目标的样本计算 MSE
                valid_pred_boxes = preds['pred_box'][valid_indices]
                valid_gt_boxes = torch.stack(gt_boxes_list)
                
                # 增加 SmoothL1Loss 代替 MSE，它对异常值更鲁棒，不容易 NaN
                loss_txt_spa = F.smooth_l1_loss(valid_pred_boxes, valid_gt_boxes, beta=0.1)
            else:
                loss_txt_spa = torch.tensor(0., device=self.device)



            # --- A2. 结构化相对位置监督 (relative_position=[dx,dy,orientation]) ---
            if 'relative_position' in preds:
                gt_rel, rel_valid = build_gt_relation_vectors(batch_targets, batch_size, self.device)
                if rel_valid.any():
                    pred_rel = preds['relative_position']
                    if pred_rel.dim() == 3:
                        pred_rel = pred_rel.mean(dim=1)  # [B, 3]
                    pred_rel = pred_rel[rel_valid]
                    gt_rel = gt_rel[rel_valid]

                    loss_rel_xy = F.smooth_l1_loss(pred_rel[:, :2], gt_rel[:, :2], beta=0.1)
                    pred_ang = pred_rel[:, 2] * torch.pi
                    gt_ang = gt_rel[:, 2] * torch.pi
                    loss_rel_ori = (1.0 - torch.cos(pred_ang - gt_ang)).mean()
                    loss_rel_pos = loss_rel_xy + 0.5 * loss_rel_ori

            # --- B. 正交约束 ---
            # 增加稳定性处理
            sem_v = preds['sem_vec']
            attr_v = preds['attr_vec']

            # 1. 显式进行带 eps 的归一化，防止除以 0
            sem_v_norm = F.normalize(sem_v, p=2, dim=1, eps=1e-8)
            attr_v_norm = F.normalize(attr_v, p=2, dim=1, eps=1e-8)

            # 2. 计算余弦相似度并平方（平方在 0 点梯度更平滑，且能达到 abs 一样的优化目的）
            loss_orth = (torch.sum(sem_v_norm * attr_v_norm, dim=1) ** 2).mean()

            # 3. 增加万一 NaN 的兜底
            if torch.isnan(loss_orth):
                loss_orth = torch.tensor(0.0, device=self.device)

            # C. 视觉对齐 (修复空目标 NaN)
            if visual_feats_list is not None:
                # 获取特征和有效掩码
                gt_vis_embeds, valid_mask = self.extract_visual_embeds(visual_feats_list, gt_bboxes, stride_tensor)
                
                # ✅ 关键修复: 只对 valid_mask 为 True 的样本计算对齐 Loss
                if valid_mask.any():
                    valid_vis = gt_vis_embeds[valid_mask]
                    valid_sem = sem_v_norm[valid_mask]
                    valid_attr = attr_v_norm[valid_mask]
                    
                    # C1. 语义-视觉对齐
                    loss_sem_align = self.align_loss(valid_vis, valid_sem)
                    
                    # C2. 属性-视觉对齐
                    gate = preds.get('gate', None)
                    if gate is not None:
                        gate = gate[valid_mask]
                    loss_attr_align = self.align_loss(valid_vis, valid_attr, gate=gate)
                else:
                    # 如果整个 Batch 都没有目标，Loss 保持为 0
                    pass

        # 聚合 Loss
        lambdas = lambdas or {}
        spa_w = float(lambdas.get('spa', 0.0))
        sem_w = float(lambdas.get('sem', 0.0))
        attr_w = float(lambdas.get('attr', 0.0))
        orth_w = float(lambdas.get('orth', 0.0))
        rel_w = float(lambdas.get('rel', 0.0))

        total_loss = (self.hyp['box'] * loss_box + 
                      self.hyp['cls'] * loss_cls + 
                      self.hyp['dfl'] * loss_dfl + 
                      spa_w * loss_txt_spa +     
                      sem_w * loss_sem_align +    
                      attr_w * loss_attr_align +   
                      orth_w * loss_orth +
                      rel_w * loss_rel_pos)          
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=self.device)

        loss_items = {
            'box': self.hyp['box'] * loss_box.item(), 
            'cls': self.hyp['cls'] * loss_cls.item(), 
            'dfl': self.hyp['dfl'] * loss_dfl.item(),
            't_spa': spa_w * loss_txt_spa.item(), 
            'sem': sem_w * loss_sem_align.item(), 
            'attr': attr_w * loss_attr_align.item(), 
            'orth': orth_w * loss_orth.item(),
            'rel': rel_w * loss_rel_pos.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_items