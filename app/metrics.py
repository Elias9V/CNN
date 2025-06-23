import torch
import torch.nn.functional as F

def compute_accuracy(logits, targets):
    """
    Calcula el accuracy: porcentaje de píxeles correctamente clasificados.
    """
    preds = torch.argmax(logits, dim=1)  # (B, H, W)
    correct = (preds == targets).float()
    acc = correct.sum() / correct.numel()
    return acc.item()

def compute_iou(logits, targets, num_classes=2):
    """
    Calcula el IoU (Intersection over Union) promedio para todas las clases.
    """
    preds = torch.argmax(logits, dim=1)  # (B, H, W)
    iou = 0.0
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou += 1.0  # si no hay presencia en ambos, considerar perfecto
        else:
            iou += intersection / union
    return (iou / num_classes).item()

def compute_f1(logits, targets):
    """
    Cálculo simple de F1 score (binario).
    """
    preds = torch.argmax(logits, dim=1)  # (B, H, W)
    TP = ((preds == 1) & (targets == 1)).sum().float()
    FP = ((preds == 1) & (targets == 0)).sum().float()
    FN = ((preds == 0) & (targets == 1)).sum().float()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.item()
