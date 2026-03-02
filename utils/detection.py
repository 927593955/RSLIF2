import torch


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


def decode_outputs(preds, model):
    """Decode raw head outputs (DFL distributions) into xywh + cls logits."""
    head = model.head
    stride = head.stride
    reg_max = head.reg_max
    nc = head.nc
    no = head.no
    device = preds[0].device

    anchor_points, stride_tensor = make_anchors(preds, stride, 0.5)
    x_cat = torch.cat([p.view(p.shape[0], no, -1) for p in preds], 2).permute(0, 2, 1)
    box_dist, cls_score = x_cat.split(((reg_max + 1) * 4, nc), 2)

    b, a, c = box_dist.shape
    proj = torch.arange(reg_max + 1, dtype=torch.float, device=device)
    dist_val = box_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(box_dist.dtype))

    decoded_boxes = dist2bbox(dist_val, anchor_points, xywh=True)
    decoded_boxes *= stride_tensor
    return torch.cat((decoded_boxes, cls_score.sigmoid()), 2)