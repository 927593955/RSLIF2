import torch
import numpy as np
from tqdm import tqdm

from utils.general import non_max_suppression
from utils.metrics import process_batch, box_iou
from utils.detection import decode_outputs


def validate(model, val_loader, device, conf_thres=0.25, iou_thres=0.6, class_agnostic=False):
    """Class-aware validation for grounding/detection quality."""
    model.eval()

    stats = []
    top1_correct = 0
    total_images_with_gt = 0
    total_gt_instances = 0

    pbar = tqdm(val_loader, desc="🔍 Validating", leave=False)

    for batch in pbar:
        imgs, input_ids, masks, targets, _ = [item.to(device) for item in batch]

        with torch.no_grad():
            output = model(imgs, [input_ids, masks])
            preds_raw = output['feats'] if isinstance(output, dict) else output
            if isinstance(preds_raw, tuple):
                preds_raw = preds_raw[0]

        preds_decoded = decode_outputs(preds_raw, model)
        preds = non_max_suppression(
            preds_decoded,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            agnostic=class_agnostic,
            multi_label=False,
        )

        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si]
            nl = len(labels)
            if nl:
                total_images_with_gt += 1
                total_gt_instances += nl

                h, w = imgs.shape[2:]
                tbox = labels[:, 2:].clone()
                tbox[:, 0] *= w
                tbox[:, 2] *= w
                tbox[:, 1] *= h
                tbox[:, 3] *= h

                box = torch.zeros_like(tbox)
                box[:, 0] = tbox[:, 0] - tbox[:, 2] / 2
                box[:, 1] = tbox[:, 1] - tbox[:, 3] / 2
                box[:, 2] = tbox[:, 0] + tbox[:, 2] / 2
                box[:, 3] = tbox[:, 1] + tbox[:, 3] / 2
                labels_pixel = torch.cat((labels[:, 1:2], box), 1)
            else:
                labels_pixel = torch.zeros((0, 5), device=device)

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, dtype=torch.bool), torch.tensor([]), torch.tensor([]), labels[:, 1].tolist()))
                continue

            predn = pred.clone()
            correct = process_batch(predn, labels_pixel, iou_thres=0.5)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels[:, 1].tolist()))

            if nl:
                top1_idx = torch.argsort(predn[:, 4], descending=True)[0]
                top1_box = predn[top1_idx, :4].unsqueeze(0)
                top1_cls = int(predn[top1_idx, 5].item())
                ious = box_iou(labels_pixel[:, 1:], top1_box).squeeze(1)
                best_gt_idx = int(torch.argmax(ious).item())
                if ious[best_gt_idx] > 0.5 and top1_cls == int(labels_pixel[best_gt_idx, 0].item()):
                    top1_correct += 1

    acc_top1 = top1_correct / (total_images_with_gt + 1e-7)

    if len(stats):
        stats = [np.concatenate(x, 0) if len(x) else np.array([]) for x in zip(*stats)]
    else:
        stats = []

    if len(stats) and stats[0].size:
        tp = stats[0].sum()
        fp = stats[0].shape[0] - tp
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (total_gt_instances + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0

    print(
        f"📊 Validation Results: Acc@0.5={acc_top1:.4f} | "
        f"P={precision:.4f} | R={recall:.4f} | F1={f1:.4f}"
    )
    return precision, recall, f1, acc_top1