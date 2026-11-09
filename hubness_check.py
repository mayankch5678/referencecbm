"""Do a few book images dominate the rankings? (uses cached embeddings only, no model)

For every sampled tile, take the top-K most similar book images and count how often
each book image appears. If a handful of images show up for most tiles, the ranking is
driven by properties of those book images ("hubs"), not by the tissue in the tile.
"""
import argparse
from collections import Counter

import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--tile_h5", default="outputs/sample_tiles_uni_v1.h5")
parser.add_argument("--index_csv", default="outputs/book_index.csv")
parser.add_argument("--cache", default="outputs/book_embeddings_uni_v1.npz")
parser.add_argument("--min_size", type=int, default=100)
parser.add_argument("--top_k", type=int, default=5)
args = parser.parse_args()


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


idx = pd.read_csv(args.index_csv)
for c in ["figure", "title", "description"]:
    idx[c] = idx[c].fillna("")
idx = idx[idx.has_caption & (idx.width >= args.min_size) & (idx.height >= args.min_size)].reset_index(drop=True)
c = np.load(args.cache, allow_pickle=True)
cache = dict(zip(c["files"], c["features"]))
idx = idx[idx.file.isin(cache)].reset_index(drop=True)
book = l2(np.stack([cache[f] for f in idx.file]))

with h5py.File(args.tile_h5, "r") as f:
    tiles = l2(f["features"][:].astype(np.float32))

sims = tiles @ book.T                                   # (n_tiles, n_book)
top = np.argsort(-sims, axis=1)[:, :args.top_k]
counts = Counter(top.ravel().tolist())
n_tiles = len(tiles)

print(f"{n_tiles} tiles x {len(idx)} book images; top-{args.top_k} per tile")
print(f"distinct book images appearing in any top-{args.top_k}: {len(counts)} of {len(idx)}")
print(f"mean top-1 similarity: {sims.max(1).mean():.3f}  (median over all pairs: {np.median(sims):.3f})")
print(f"\nMost frequent book images (appearances out of {n_tiles} tiles):")
for i, n in counts.most_common(10):
    r = idx.iloc[i]
    print(f"  {n:3d}/{n_tiles}  {r.file:18s} {r.figure:12s} {r.title[:55]}")

# which chapters / figures do the top-1 matches come from?
top1 = idx.iloc[top[:, 0]]
print("\nChapter of the top-1 match, over all tiles:")
print(top1.chapter.value_counts().sort_index().to_string())
