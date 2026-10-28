import gc
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

IMG, SEED, HAM_T = 224, 42, 4
MAIN_FOLDS, VAL_FOLDS, CV_FOLDS = 7, 6, 5
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def find_root(root):
    root = Path(root)
    if not (root / "Training_set.csv").exists():
        cand = list(root.rglob("Training_set.csv"))
        assert cand, "Training_set.csv not found under %s" % root
        root = cand[0].parent
    return root


def load_table(root):
    df = pd.read_csv(root / "Training_set.csv")
    classes = sorted(df.label.unique())
    c2i = {c: i for i, c in enumerate(classes)}
    return df, classes, df.label.map(c2i).values


def load_images(root, df, cache):
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)

    def load(fn):
        im = Image.open(root / "train" / fn)
        im.draft("RGB", (IMG * 2, IMG * 2))
        return np.asarray(im.convert("RGB").resize((IMG, IMG), Image.BILINEAR))

    t0 = time.time()
    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        X = np.stack(list(ex.map(load, df.filename.tolist())))
    np.save(cache, X)
    print("Cached %s in %.0fs" % (X.shape, time.time() - t0))
    return X


@torch.no_grad()
def dhash_bits(Xu8):
    out = []
    for i in range(0, len(Xu8), 512):
        g = Xu8[i:i + 512].float().mean(-1, keepdim=True).permute(0, 3, 1, 2)
        g = F.adaptive_avg_pool2d(g, (8, 9))[:, 0]
        out.append((g[:, :, 1:] > g[:, :, :-1]).flatten(1).float())
    return torch.cat(out)


def near_duplicate_groups(Xg, ham_t=HAM_T):
    N = len(Xg)
    Hb = dhash_bits(Xg)
    rows, cols = [], []
    with torch.no_grad():
        for i in range(0, N, 1024):
            a = Hb[i:i + 1024]
            ham = 64 - (a @ Hb.T + (1 - a) @ (1 - Hb).T)
            r, c = torch.nonzero(ham <= ham_t, as_tuple=True)
            r = r + i
            m = c > r
            rows.append(r[m].cpu())
            cols.append(c[m].cpu())
    rows = torch.cat(rows).numpy()
    cols = torch.cat(cols).numpy()
    _, groups = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)), directed=False)
    return groups, len(rows)


def main_split(y, groups, seed=SEED):
    idx = np.arange(len(y))
    trv, te = next(StratifiedGroupKFold(MAIN_FOLDS, shuffle=True, random_state=seed).split(idx, y, groups))
    a, b = next(StratifiedGroupKFold(VAL_FOLDS, shuffle=True, random_state=seed).split(trv, y[trv], groups[trv]))
    return trv[a], trv[b], te


def cv_splits(y, groups, seed=SEED, folds=CV_FOLDS):
    for f, (trv, te) in enumerate(
            StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(np.arange(len(y)), y, groups)):
        a, b = next(StratifiedGroupKFold(VAL_FOLDS, shuffle=True, random_state=seed).split(trv, y[trv], groups[trv]))
        yield f + 1, trv[a], trv[b], te


def write_manifests(df, y, groups, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame(dict(filename=df.filename.values, label=df.label.values, group=groups))
    base.to_csv(outdir / "groups.csv", index=False)
    tr, va, te = main_split(y, groups)
    main = base.copy()
    main["split"] = ""
    main.loc[tr, "split"] = "train"
    main.loc[va, "split"] = "val"
    main.loc[te, "split"] = "test"
    main.to_csv(outdir / "split_main.csv", index=False)
    cv = base.copy()
    for f, a, b, c in cv_splits(y, groups):
        col = np.full(len(y), "", dtype=object)
        col[a], col[b], col[c] = "train", "val", "test"
        cv["fold%d" % f] = col
    cv.to_csv(outdir / "split_cv.csv", index=False)
    return tr, va, te


def read_main_manifest(path):
    m = pd.read_csv(path)
    return tuple(np.where(m.split.values == s)[0] for s in ("train", "val", "test"))


def read_cv_manifest(path):
    m = pd.read_csv(path)
    for f in range(1, CV_FOLDS + 1):
        col = m["fold%d" % f].values
        yield f, np.where(col == "train")[0], np.where(col == "val")[0], np.where(col == "test")[0]


def prep(xb):
    return xb.permute(0, 3, 1, 2).float().div(255)


def augment(x, zoom_lo):
    B = x.size(0)

    def u(lo, hi):
        return torch.empty(B, device=x.device).uniform_(lo, hi)

    s = u(zoom_lo, 1.0)
    flip = (torch.rand(B, device=x.device) < 0.5).float() * 2 - 1
    r = u(-0.2, 0.2)
    tx, ty = (1 - s) * u(-1, 1), (1 - s) * u(-1, 1)
    th = torch.stack([torch.stack([s * torch.cos(r) * flip, -s * torch.sin(r), tx], 1),
                      torch.stack([s * torch.sin(r) * flip, s * torch.cos(r), ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False),
                      padding_mode="reflection", align_corners=False)
    b = u(0.8, 1.2).view(B, 1, 1, 1)
    c = u(0.8, 1.2).view(B, 1, 1, 1)
    m = x.mean((1, 2, 3), keepdim=True)
    return ((x - m) * c + m * b).clamp(0, 1)


@torch.no_grad()
def predict(model, Xg, idx, mean, std, tta=True, bs=128):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        x = (prep(Xg[torch.as_tensor(idx[i:i + bs], device=Xg.device)]) - mean) / std
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(x.flip(3)).float().softmax(1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


def free():
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
