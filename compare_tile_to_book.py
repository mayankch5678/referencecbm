"""Christoph's toy comparison: one WSI tile vs. textbook images.

1. take tile <tile_index> from the sampled tiles (feature vector already computed by UNI),
2. encode the textbook images with the SAME encoder (cached in outputs/),
3. cosine similarity: first with ONE random book image (the toy example),
   then with all captioned book images,
4. assign the caption of the best-matching book image to the tile.
"""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--tile_h5", default="outputs/sample_tiles_uni_v1.h5")
parser.add_argument("--tile_dir", default="outputs/sample_tiles_uni_v1")
parser.add_argument("--tile_index", type=int, default=26)
parser.add_argument("--index_csv", default="outputs/book_index.csv")
parser.add_argument("--img_dir", default="data/book_images")
parser.add_argument("--encoder", default="uni_v1")
parser.add_argument("--min_size", type=int, default=100, help="skip book images smaller than this (icons)")
parser.add_argument("--limit", type=int, default=0, help="encode only the first N book images (quick test)")
parser.add_argument("--cache", default="outputs/book_embeddings_uni_v1.npz")
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--out_png", default="outputs/tile_vs_book.png")
args = parser.parse_args()


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ---- tile -------------------------------------------------------------------------------
with h5py.File(args.tile_h5, "r") as f:
    tile_feat = f["features"][args.tile_index].astype(np.float32)
    x, y = f["coords"][args.tile_index]
tile_path = Path(args.tile_dir) / f"tile_{x}_{y}.png"
print(f"Tile {args.tile_index}: {tile_path}")

# ---- candidate book images --------------------------------------------------------------
idx = pd.read_csv(args.index_csv)
for c in ["figure", "title", "description"]:
    idx[c] = idx[c].fillna("")
idx = idx[idx.has_caption & (idx.width >= args.min_size) & (idx.height >= args.min_size)].reset_index(drop=True)
if args.limit:
    idx = idx.head(args.limit)
print(f"{len(idx)} captioned book images (>= {args.min_size}px)")

# ---- encode book images (cached) --------------------------------------------------------
cache = {}
if Path(args.cache).exists():
    c = np.load(args.cache, allow_pickle=True)
    cache = dict(zip(c["files"], c["features"]))
todo = [f for f in idx.file if f not in cache]
if todo:
    import torch
    from trident.patch_encoder_models import encoder_factory
    encoder = encoder_factory(args.encoder).eval().to("cpu")
    transform = encoder.eval_transforms
    print(f"Encoding {len(todo)} book images with {args.encoder} on CPU ...")
    for i in range(0, len(todo), args.batch):
        names = todo[i:i + args.batch]
        batch = torch.stack([transform(Image.open(Path(args.img_dir) / n).convert("RGB")) for n in names])
        with torch.inference_mode():
            feats = encoder(batch).float().cpu().numpy()
        cache.update(zip(names, feats))
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)}", flush=True)
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.cache, files=np.array(list(cache.keys())), features=np.stack(list(cache.values())))
else:
    print("All book image embeddings found in cache.")

feats = np.stack([cache[f] for f in idx.file])
sims = l2(feats) @ l2(tile_feat)          # cosine similarity tile <-> every book image
idx["similarity"] = sims

# ---- 1) toy example: ONE random book image ----------------------------------------------
r = idx.sample(1, random_state=args.seed).iloc[0]
print("\n--- Toy example: tile vs. ONE random book image ---")
print(f"{r.file}  ({r.figure}, ch{r.chapter} p{r.page}): {r.title}")
print(f"cosine similarity = {r.similarity:.3f}")

# ---- 2) all images: best match and its caption ------------------------------------------
top = idx.sort_values("similarity", ascending=False).head(args.top_k)
print(f"\n--- Top {args.top_k} of {len(idx)} book images ---")
for rank, (_, t) in enumerate(top.iterrows(), 1):
    print(f"{rank}. sim={t.similarity:.3f}  {t.file}  {t.figure} (ch{t.chapter} p{t.page}, panel {t.panel}/{t.n_panels})")
    print(f"   {t.title}")
best = top.iloc[0]
print("\n>>> Caption assigned to the tile:")
print(f"{best.figure}. {best.title}")
print(f"{best.description}")
print(f"\nSimilarity spread: min={sims.min():.3f} median={np.median(sims):.3f} max={sims.max():.3f}; "
      f"top1-top2 gap={top.similarity.iloc[0] - top.similarity.iloc[1]:.3f}")

# ---- figure -----------------------------------------------------------------------------
k = min(3, len(top))
fig, axes = plt.subplots(1, k + 1, figsize=(3.2 * (k + 1), 4))
axes[0].imshow(Image.open(tile_path)); axes[0].set_title(f"WSI tile {args.tile_index}", fontsize=9)
for ax, (_, t) in zip(axes[1:], top.head(k).iterrows()):
    ax.imshow(Image.open(Path(args.img_dir) / t.file))
    ax.set_title(f"{t.figure}\nsim={t.similarity:.3f}\n{t.title[:34]}", fontsize=8)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out_png, dpi=150)
print(f"\nSaved figure to {args.out_png}")
