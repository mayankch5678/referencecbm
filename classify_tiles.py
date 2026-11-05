"""Steps 4-5 of Christoph's task: compare each WSI tile with every prototype
by cosine similarity and assign class = argmax_i sim(patch, prototype_i).

Reads the TRIDENT feature file (h5 with 'features' and 'coords').
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--features_h5", required=True,
                    help="e.g. trident_out/20x_256px_0px_overlap/features_uni_v1/<slide>.h5")
parser.add_argument("--prototypes", default="outputs/prototypes_uni_v1.npz")
parser.add_argument("--out", default="outputs/tile_predictions.csv")
args = parser.parse_args()

with h5py.File(args.features_h5, "r") as f:
    tiles = f["features"][:]          # (n_tiles, dim)
    coords = f["coords"][:]           # (n_tiles, 2), level-0 x/y

proto = np.load(args.prototypes)
protos, labels, files = proto["features"], proto["labels"], proto["files"]

def l2norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

sim = l2norm(tiles) @ l2norm(protos).T   # cosine similarity, (n_tiles, n_prototypes)
best = sim.argmax(axis=1)                # class = max_i sim(patch, prototype_i)

df = pd.DataFrame({
    "x": coords[:, 0], "y": coords[:, 1],
    "pred_class": labels[best],
    "best_prototype": [Path(files[i]).name for i in best],
    "max_sim": sim.max(axis=1),
})
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, index=False)
print(df["pred_class"].value_counts())
print(f"Saved predictions for {len(df)} tiles to {args.out}")
