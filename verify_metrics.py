import argparse
import math

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from metrics import compute_metrics


def load(path):
    d = pd.read_csv(path)
    pc = [c for c in d.columns if c.startswith("p_")]
    classes = [c[2:] for c in pc]
    idx = {c: i for i, c in enumerate(classes)}
    return d, pc, classes, d.label_true.map(idx).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/vit/predictions_main.csv")
    ap.add_argument("--cv", default=None)
    ap.add_argument("--expect-accuracy", type=float, default=None)
    a = ap.parse_args()

    d, pc, classes, yt = load(a.main)
    P = d[pc].values
    M, pred = compute_metrics(yt, P)
    assert (pred == d.label_pred.map({c: i for i, c in enumerate(classes)}).values).all(), \
        "label_pred column disagrees with arg max of the probabilities"
    n, k = len(d), int((pred == yt).sum())
    se = math.sqrt(M["accuracy"] * (1 - M["accuracy"]) / n)
    print("Images: %d | correct: %d" % (n, k))
    for key, v in M.items():
        print("%-20s: %.4f" % (key, v))
    print("95%% normal-approximation CI for accuracy: %.2f - %.2f %%" % (
        (M["accuracy"] - 1.96 * se) * 100, (M["accuracy"] + 1.96 * se) * 100))
    print(classification_report(yt, pred, target_names=classes, digits=4, zero_division=0))
    cm = confusion_matrix(yt, pred, labels=range(len(classes)))
    assert cm.sum() == n
    if a.expect_accuracy is not None:
        ok = abs(M["accuracy"] * 100 - a.expect_accuracy) < 0.005
        print("Expected accuracy %.2f%% -> %s" % (a.expect_accuracy, "MATCH" if ok else "MISMATCH"))
        if not ok:
            raise SystemExit(1)
    if a.cv:
        c = pd.read_csv(a.cv)
        pcs = [x for x in c.columns if x.startswith("p_")]
        cls = [x[2:] for x in pcs]
        idx = {x: i for i, x in enumerate(cls)}
        rows = []
        for f, g in c.groupby("fold"):
            m, _ = compute_metrics(g.label_true.map(idx).values, g[pcs].values)
            rows.append((f, len(g), m["accuracy"], m["macro_f1"]))
        r = pd.DataFrame(rows, columns=["fold", "n_test", "accuracy", "macro_f1"])
        print(r.to_string(index=False))
        print("CV accuracy %.2f +- %.2f %% | macro-F1 %.2f +- %.2f %% (sample SD)" % (
            r.accuracy.mean() * 100, r.accuracy.std() * 100, r.macro_f1.mean() * 100, r.macro_f1.std() * 100))


if __name__ == "__main__":
    main()
