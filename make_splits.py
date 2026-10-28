import argparse
import os

import numpy as np
import torch

from har_common import (dev, find_root, load_images, load_table, near_duplicate_groups, seed_all,
                        SEED, write_manifests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("HAR_ROOT", "data/Human Action Recognition"))
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--cache", default="cache_224.npy")
    a = ap.parse_args()

    seed_all(SEED)
    root = find_root(a.root)
    df, classes, y = load_table(root)
    print("Labelled images: %d | classes: %d | device: %s" % (len(df), len(classes), dev))
    Xg = torch.from_numpy(load_images(root, df, a.cache)).to(dev)
    groups, n_pairs = near_duplicate_groups(Xg)
    print("Near-duplicate pairs: %d | groups: %d / %d" % (n_pairs, groups.max() + 1, len(df)))
    tr, va, te = write_manifests(df, y, groups, a.out)
    shared = sum(len(set(groups[p]) & set(groups[q])) for p, q in [(tr, te), (tr, va), (va, te)])
    print("Split train/val/test = %d/%d/%d | groups shared across splits: %d" % (len(tr), len(va), len(te), shared))
    assert shared == 0
    print("Class counts in test split:", dict(zip(*np.unique(y[te], return_counts=True))))
    print("Manifests written to", a.out)


if __name__ == "__main__":
    main()
