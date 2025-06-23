#!/usr/bin/env python
"""
Evaluación global:
  ▸ Curva ROC  + AUC
  ▸ Matriz de confusión (umbral óptimo)
  ▸ Métricas: IoU, F1, Accuracy
Guarda:
    data/outputs/roc_curve.png
    data/outputs/confusion.png
"""

import argparse, os, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from app.metrics import compute_iou, compute_f1_score, compute_accuracy

# ─── Rutas fijas ─────────────────────────────────────────
MASKS_PT    = "data/tensors/masks.pt"
PROB_DIR    = "data/outputs/prob_patches"
MOS_NPY     = "data/outputs/mosaico_prob.npy"
OUT_DIR     = "data/outputs"

# ════════════════════════════════════════════════════════
def load_probs(src: str) -> np.ndarray:
    """ Devuelve un array (Npixels,) de probabilidades [0-1] """
    if src == "mosaic":
        prob = np.load(MOS_NPY)
        return prob.flatten()
    elif src == "patches":
        files = sorted(f for f in os.listdir(PROB_DIR) if f.endswith(".pt"))
        probs = [torch.load(os.path.join(PROB_DIR, f)).flatten().numpy() for f in files]
        return np.concatenate(probs)
    else:
        raise ValueError("src debe ser 'mosaic' o 'patches'")

def load_masks() -> np.ndarray:
    """ Devuelve un array (Npixels,) 0/1 """
    masks = torch.load(MASKS_PT).numpy()
    if masks.ndim == 4:       # (B,1,H,W)
        masks = masks.squeeze(1)
    return masks.flatten().astype(int)

# ════════════════════════════════════════════════════════
def main(src: str):
    os.makedirs(OUT_DIR, exist_ok=True)

    y_true = load_masks()
    y_prob = load_probs(src)

    # ── ROC & AUC ────────────────────────────────────────
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    roc_auc       = auc(fpr, tpr)

    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--',c='gray')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/roc_curve.png")
    print("✅ Curva ROC guardada.")

    # ── Umbral óptimo (Youden J) ─────────────────────────
    best_thr = thr[(tpr - fpr).argmax()]
    y_pred   = (y_prob >= best_thr).astype(int)

    # ── Matriz de confusión ─────────────────────────────
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Seguro","Riesgo"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusión (thr={best_thr:.2f})")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion.png")
    print("✅ Matriz de confusión guardada.")

    # ── Métricas IoU / F1 / Acc ─────────────────────────
    # convert back to 2-D tensors for helper funcs
    y_pred_t = torch.tensor(y_pred.reshape(-1,1,1))
    y_true_t = torch.tensor(y_true.reshape(-1,1,1))

    iou = compute_iou(y_pred_t, y_true_t)
    f1  = compute_f1_score(y_pred_t, y_true_t)
    acc = compute_accuracy(y_pred_t, y_true_t)

    print(f"\n📊  Métricas globales")
    print(f"    AUC      : {roc_auc:.4f}")
    print(f"    IoU      : {iou:.4f}")
    print(f"    F1-score : {f1:.4f}")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Umbral óptimo : {best_thr:.3f}")

# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar modelo con ROC/AUC.")
    parser.add_argument("--src", choices=["mosaic", "patches"], default="mosaic",
                        help="Fuente de probabilidades: 'mosaic' (mosaico_prob.npy) "
                             "o 'patches' (prob_patch_###.pt)")
    args = parser.parse_args()
    main(args.src)
