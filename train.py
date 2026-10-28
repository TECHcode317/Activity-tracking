import argparse
import copy
import json
import math
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.flop_counter import FlopCounterMode

import har_common as hc
from har_common import dev, IMG, SEED, augment, free, predict, prep, seed_all
from metrics import compute_metrics

CFG = {
    "vit": dict(name="vit_base_patch16_clip_224.openai", epochs=8, bs=32, lr=1e-4, head_mult=5.0, llrd=0.75,
                wd=0.05, drop_path=0.1, dropout=0.1, ema=True, schedule="warmup_cosine", zoom_lo=0.65,
                bench_bs=32, bench_iters=80, cv_epochs=4),
    "effnet": dict(name="efficientnet_b0", epochs=15, bs=64, lr=6e-4, head_mult=1.0, llrd=1.0,
                   wd=0.01, drop_path=0.0, dropout=0.3, ema=False, schedule="one_cycle", zoom_lo=0.7,
                   bench_bs=64, bench_iters=150, cv_epochs=None),
}
LABEL_SMOOTH, WARM_EP, EMA_DECAY, ONECYCLE_PCT = 0.1, 1, 0.998, 0.15

Xg = y = ytr_all = classes = filenames = MEAN = STD = None
NC = 0


class EMA:
    def __init__(self, model):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(EMA_DECAY, (1 + step) / (10 + step))
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)


def build(cfg):
    global MEAN, STD
    m = timm.create_model(cfg["name"], pretrained=True, num_classes=NC,
                          drop_rate=cfg["dropout"], drop_path_rate=cfg["drop_path"])
    pc = m.pretrained_cfg
    MEAN = torch.tensor(pc.get("mean", (0.485, 0.456, 0.406)), device=dev).view(1, 3, 1, 1)
    STD = torch.tensor(pc.get("std", (0.229, 0.224, 0.225)), device=dev).view(1, 3, 1, 1)
    return m.to(dev)


def param_groups(model, cfg):
    has_blocks = hasattr(model, "blocks") and len(model.blocks) > 0
    n = len(model.blocks) if has_blocks else 0
    emb = ("cls_token", "pos_embed", "patch_embed", "norm_pre", "reg_token")
    groups_d = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if has_blocks:
            lid = 0 if name.startswith(emb) else (int(name.split(".")[1]) + 1 if name.startswith("blocks.") else n + 1)
        else:
            lid = 1
        is_head = name.startswith(("head", "fc", "classifier"))
        scale = (cfg["llrd"] ** (n + 1 - lid)) if has_blocks else 1.0
        if is_head:
            scale = cfg["head_mult"]
        nd = (p.ndim <= 1) or name.endswith(".bias") or name in ("cls_token", "pos_embed")
        key = (lid, nd, is_head)
        g = groups_d.setdefault(key, dict(params=[], base=cfg["lr"] * scale,
                                          weight_decay=0.0 if nd else cfg["wd"]))
        g["params"].append(p)
    return list(groups_d.values())


def train_eval(cfg, tr, va, te, tag, epochs):
    seed_all(SEED)
    model = build(cfg)
    ema = EMA(model) if cfg["ema"] else None
    opt = torch.optim.AdamW(param_groups(model, cfg), lr=cfg["lr"], betas=(0.9, 0.999))
    for g in opt.param_groups:
        g["lr"] = g["base"]
    BS = cfg["bs"]
    spe = int(np.ceil(len(tr) / BS))
    total, warm = epochs * spe, WARM_EP * spe
    sched = None
    if cfg["schedule"] == "one_cycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["base"] for g in opt.param_groups],
                                                    total_steps=total, pct_start=ONECYCLE_PCT)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    trg = torch.as_tensor(tr, device=dev)
    hist, best, best_state, step = [], -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = trg[torch.randperm(len(trg), device=dev)]
        t0 = time.time()
        tl = torch.zeros((), device=dev)
        tc = torch.zeros((), device=dev)
        for i in range(0, len(perm), BS):
            ib = perm[i:i + BS]
            if sched is None:
                f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
                for g in opt.param_groups:
                    g["lr"] = g["base"] * max(f, 0.01)
            x = (augment(prep(Xg[ib]), cfg["zoom_lo"]) - MEAN) / STD
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), ytr_all[ib], label_smoothing=LABEL_SMOOTH)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if cfg["ema"]:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            if ema is not None:
                ema.update(model, step)
            step += 1
            tl += loss.detach() * len(ib)
            tc += (logits.argmax(1) == ytr_all[ib]).sum()
        ev = ema.m if ema is not None else model
        Pv = predict(ev, Xg, va, MEAN, STD, tta=False)
        vl = float(F.nll_loss(torch.log(torch.as_tensor(Pv) + 1e-9), torch.as_tensor(y[va])))
        va_acc = float((Pv.argmax(1) == y[va]).mean())
        hist.append(dict(epoch=ep + 1, train_loss=float(tl) / len(tr), train_acc=float(tc) / len(tr),
                         val_loss=vl, val_acc=va_acc))
        if va_acc > best:
            best = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in ev.state_dict().items()}
        print("[%s] ep %02d/%d loss %.4f tr_acc %.4f val_loss %.4f val_acc %.4f (%.0fs)" % (
            tag, ep + 1, epochs, hist[-1]["train_loss"], hist[-1]["train_acc"], vl, va_acc, time.time() - t0))
    ev = ema.m if ema is not None else model
    ev.load_state_dict(best_state)
    del opt
    P = predict(ev, Xg, te, MEAN, STD, tta=True)
    P = P / P.sum(1, keepdims=True)
    M, pred = compute_metrics(y[te], P)
    M.update(n_train=len(tr), n_val=len(va), n_test=len(te), best_val_acc=best)
    return ev, M, hist, y[te], pred, P


def prediction_frame(te, P, fold=None):
    d = pd.DataFrame(P, columns=["p_" + c for c in classes])
    d.insert(0, "label_pred", [classes[i] for i in P.argmax(1)])
    d.insert(0, "label_true", [classes[i] for i in y[te]])
    d.insert(0, "filename", filenames[te])
    if fold is not None:
        d.insert(0, "fold", fold)
    return d


def plot_cm(mat, title, path, fmt):
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(NC))
    ax.set_yticks(range(NC))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    thr = mat.max() / 2
    for i in range(NC):
        for j in range(NC):
            ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=6.5,
                    color="w" if mat[i, j] > thr else "k")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, fraction=.046)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def save_main_outputs(out, M, hist, yt, pred, P, te):
    rep = classification_report(yt, pred, target_names=classes, digits=4, zero_division=0)
    (out / "classification_report.txt").write_text(rep)
    cm = confusion_matrix(yt, pred, labels=range(NC))
    assert cm.sum() == len(te)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_matrix.csv")
    pd.DataFrame(hist).to_csv(out / "training_history.csv", index=False)
    prediction_frame(te, P).to_csv(out / "predictions_main.csv", index=False)
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="Train loss (label-smoothed)")
    ax[0].plot(h.epoch, h.val_loss, "s-", label="Val loss (unsmoothed CE)")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training / validation loss")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(h.epoch, h.train_acc * 100, "o-", label="Train acc (augmented)")
    ax[1].plot(h.epoch, h.val_acc * 100, "s-", label="Val acc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title("Training / validation accuracy")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=170)
    plt.close()
    plot_cm(cm, "Multiclass confusion matrix (n=%d, acc=%.2f%%)" % (cm.sum(), M["accuracy"] * 100),
            out / "confusion_matrix.png", "d")
    plot_cm(cm / cm.sum(1, keepdims=True), "Row-normalised confusion matrix",
            out / "confusion_matrix_normalized.png", ".2f")
    f1 = precision_recall_fscore_support(yt, pred, zero_division=0)[2]
    plt.figure(figsize=(9, 4))
    plt.bar(classes, f1 * 100)
    plt.xticks(rotation=90)
    plt.ylabel("F1 (%)")
    plt.title("Per-class F1")
    plt.tight_layout()
    plt.savefig(out / "per_class_f1.png", dpi=170)
    plt.close()
    print(rep)


def run_cv(cfg, out, manifest):
    rows, preds = [], []
    for f, tr, va, te in hc.read_cv_manifest(manifest):
        mdl, mf, _, _, _, P = train_eval(cfg, tr, va, te, "cv%d" % f, cfg["cv_epochs"])
        mf["fold"] = f
        rows.append(mf)
        preds.append(prediction_frame(te, P, fold=f))
        del mdl
        free()
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv_folds.csv", index=False)
    pd.concat(preds).to_csv(out / "predictions_cv.csv", index=False)
    ma, sa, mf1, sf1 = cv.accuracy.mean(), cv.accuracy.std(), cv.macro_f1.mean(), cv.macro_f1.std()
    print(cv[["fold", "n_train", "n_val", "n_test", "accuracy", "macro_f1"]].to_string(index=False))
    print("%d-fold CV accuracy: %.2f +- %.2f %% | macro-F1: %.2f +- %.2f %%" % (
        len(cv), ma * 100, sa * 100, mf1 * 100, sf1 * 100))
    fig, ax = plt.subplots(figsize=(8, 4))
    w, xs = 0.38, np.arange(len(cv))
    ax.bar(xs - w / 2, cv.accuracy * 100, w, label="Accuracy")
    ax.bar(xs + w / 2, cv.macro_f1 * 100, w, label="Macro-F1")
    ax.axhline(ma * 100, color="r", ls="--", label="Mean acc %.2f+-%.2f%%" % (ma * 100, sa * 100))
    ax.set_xticks(xs)
    ax.set_xticklabels(["Fold %d" % (i + 1) for i in xs])
    ax.set_ylim(max(0, cv.accuracy.min() * 100 - 8), 100)
    ax.set_ylabel("%")
    ax.set_title("%d-fold group-stratified cross-validation" % len(cv))
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "kfold_validation.png", dpi=170)
    plt.close()


def flops_params(name):
    m = timm.create_model(name, pretrained=False, num_classes=NC).eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(torch.randn(1, 3, IMG, IMG))
    return fc.get_total_flops() / 1e9, sum(p.numel() for p in m.parameters()) / 1e6


def bench(model, bs, iters):
    x = torch.randn(bs, 3, IMG, IMG, device=dev)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
    return (time.time() - t) / iters


class PowerMeter:
    def __enter__(self):
        self.s, self.run = [], True

        def poll():
            while self.run:
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"])
                    self.s.append(float(out.decode().split()[0]))
                except Exception:
                    break
                time.sleep(0.1)

        self.th = threading.Thread(target=poll)
        self.th.start()
        return self

    def __exit__(self, *a):
        self.run = False
        self.th.join()


def efficiency(model, cfg, key, gpu, out):
    try:
        timm.layers.set_fused_attn(False)
    except Exception:
        pass
    mg = model.eval()
    eff = {"backbone": cfg["name"], "params_total": sum(p.numel() for p in mg.parameters())}
    eff["GFLOPs_(2xMAC)"], _ = flops_params(cfg["name"])
    eff["GMACs"] = eff["GFLOPs_(2xMAC)"] / 2
    eff["latency_ms_bs1_fp16"] = bench(mg, 1, 100) * 1e3
    bs, n_it = cfg["bench_bs"], cfg["bench_iters"]
    eff["throughput_batch"] = bs
    eff["throughput_img_per_s"] = bs / bench(mg, bs, 30)
    if dev.type == "cuda":
        with PowerMeter() as pmr:
            t0 = time.time()
            bench(mg, bs, n_it)
            dt = time.time() - t0
        if pmr.s:
            P = float(np.mean(pmr.s))
            eff.update(gpu_avg_power_W=P, power_samples=len(pmr.s), energy_elapsed_s=dt,
                       energy_images=n_it * bs, energy_per_image_J=P * dt / (n_it * bs),
                       energy_per_image_steady_state_J=P / eff["throughput_img_per_s"],
                       power_note="Whole-board power on %s via nvidia-smi every 100 ms; elapsed time includes "
                                  "10 warm-up passes; not edge-device power." % gpu)
    if key == "vit":
        names = ["mobilenetv3_large_100", "efficientnet_b0", "resnet50", "convnext_tiny", cfg["name"]]
        cmp = []
        for nme in dict.fromkeys(names):
            g, p = flops_params(nme)
            cmp.append((nme + (" (proposed)" if nme == cfg["name"] else ""), g, p))
        cdf = pd.DataFrame(cmp, columns=["model", "GFLOPs", "Params_M"])
        cdf.to_csv(out / "complexity_comparison.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        cols = ["tab:red" if "proposed" in m_ else "tab:blue" for m_ in cdf.model]
        ax[0].barh(cdf.model, cdf.GFLOPs, color=cols)
        ax[0].set_xlabel("GFLOPs / image (2xMAC, 224x224)")
        ax[0].set_title("Computational complexity")
        for i, v in enumerate(cdf.GFLOPs):
            ax[0].text(v, i, " %.2f" % v, va="center", fontsize=8)
        ax[1].barh(cdf.model, cdf.Params_M, color=cols)
        ax[1].set_xlabel("Parameters (M)")
        ax[1].set_title("Model size")
        for i, v in enumerate(cdf.Params_M):
            ax[1].text(v, i, " %.1f" % v, va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "gflops_comparison.png", dpi=170)
        plt.close()
    print("=== EFFICIENCY ===")
    for k, v in eff.items():
        print("%-34s: %s" % (k, v))
    return eff


def main():
    global Xg, y, ytr_all, classes, filenames, NC
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CFG), required=True)
    ap.add_argument("--root", default=os.environ.get("HAR_ROOT", "data/Human Action Recognition"))
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="cache_224.npy")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--no-benchmark", action="store_true")
    a = ap.parse_args()
    cfg = CFG[a.model]
    if a.cv and cfg["cv_epochs"] is None:
        raise SystemExit("Cross-validation is defined for the ViT only (--model vit).")

    torch.backends.cudnn.benchmark = True
    seed_all(SEED)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    print("Device:", dev, gpu)
    root = hc.find_root(a.root)
    df, classes, y = hc.load_table(root)
    NC, filenames = len(classes), df.filename.values
    Xg = torch.from_numpy(hc.load_images(root, df, a.cache)).to(dev)
    ytr_all = torch.as_tensor(y, device=dev)

    man = Path(a.manifests)
    tr, va, te = hc.read_main_manifest(man / "split_main.csv")
    print("Split train/val/test = %d/%d/%d" % (len(tr), len(va), len(te)))
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)

    model, M, hist, yt, pred, P = train_eval(cfg, tr, va, te, "main", cfg["epochs"])
    print("Backbone: %s\n=== TEST RESULTS (held-out) ===" % cfg["name"])
    for k, v in M.items():
        print("%-20s: %s" % (k, "%.4f" % v if isinstance(v, float) else v))
    save_main_outputs(out, M, hist, yt, pred, P, te)

    weights = Path(a.out) / ("%s_best_weights.pt" % a.model)
    torch.save(model.state_dict(), weights)

    if a.cv:
        del model
        free()
        run_cv(cfg, out, man / "split_cv.csv")
        model = build(cfg)
        model.load_state_dict(torch.load(weights, map_location=dev))

    eff = {}
    if not a.no_benchmark:
        eff = efficiency(model, cfg, a.model, gpu, out)

    hp = dict(model=a.model, **cfg, label_smoothing=LABEL_SMOOTH, warmup_epochs=WARM_EP, ema_decay=EMA_DECAY,
              onecycle_pct_start=ONECYCLE_PCT, grad_clip=1.0 if cfg["ema"] else None, betas=(0.9, 0.999),
              precision="AMP fp16", tta="hflip", seed=SEED, ham_threshold=hc.HAM_T, gpu=gpu,
              torch=torch.__version__, timm=timm.__version__, python=platform.python_version())
    json.dump(dict(hyperparameters=hp, test_metrics=M, efficiency=eff),
              open(out / "all_results.json", "w"), indent=2, default=str)
    pd.Series({**M, **eff}).to_csv(out / "results_summary.csv", header=["value"])
    print("Saved:", sorted(p.name for p in out.iterdir()))
    print("FINAL held-out accuracy: %.2f%%" % (M["accuracy"] * 100))


if __name__ == "__main__":
    main()
