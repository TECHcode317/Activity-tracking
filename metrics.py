import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score)


def compute_metrics(yt, P):
    P = np.asarray(P, dtype=np.float64)
    P = P / P.sum(1, keepdims=True)
    pred = P.argmax(1)
    pm = precision_recall_fscore_support(yt, pred, average="macro", zero_division=0)
    pw = precision_recall_fscore_support(yt, pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(yt, P, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    out = dict(accuracy=accuracy_score(yt, pred), balanced_acc=balanced_accuracy_score(yt, pred),
               macro_precision=pm[0], macro_recall=pm[1], macro_f1=pm[2],
               weighted_precision=pw[0], weighted_recall=pw[1], weighted_f1=pw[2], macro_auc_ovr=auc)
    return out, pred
