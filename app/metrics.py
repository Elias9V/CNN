import torch
import torch.nn.functional as F

def _is_multiclass(logits):
    return logits.shape[1] > 1  # C > 1

def _get_predictions(logits):
    if _is_multiclass(logits):
        return torch.argmax(logits, dim=1)  # (B, H, W)
    else:
        return (torch.sigmoid(logits) > 0.5).int().squeeze(1)  # (B, H, W)

def compute_accuracy(logits, targets):
    preds = _get_predictions(logits)
    correct = (preds == targets).float()
    acc = correct.sum() / correct.numel()
    return acc.item()

def compute_iou(logits, targets, num_classes=2):
    preds = _get_predictions(logits)
    iou = 0.0
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou += 1.0  # sin presencia en ambos = perfecto
        else:
            iou += intersection / union
    return (iou / num_classes).item()

def compute_f1_score(logits, targets):
    preds = _get_predictions(logits)

    if _is_multiclass(logits):
        f1_total = 0.0
        num_classes = logits.shape[1]
        for cls in range(num_classes):
            TP = ((preds == cls) & (targets == cls)).sum().float()
            FP = ((preds == cls) & (targets != cls)).sum().float()
            FN = ((preds != cls) & (targets == cls)).sum().float()

            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_total += f1
        return (f1_total / num_classes).item()

    else:  # binario
        TP = ((preds == 1) & (targets == 1)).sum().float()
        FP = ((preds == 1) & (targets == 0)).sum().float()
        FN = ((preds == 0) & (targets == 1)).sum().float()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.item()
