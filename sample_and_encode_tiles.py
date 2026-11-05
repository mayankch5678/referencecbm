"""Step 2, lightweight version: encode only a small SAMPLE of WSI tiles.

Uses TRIDENT's patch coordinates (from --task seg + --task coords) and
TRIDENT's UNI encoder, but encodes N random tiles instead of the whole slide.
Writes an h5 with 'features' and 'coords', so classify_tiles.py works unchanged.
Tile images are also saved as PNG so you can look at them.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from trident.IO import read_coords
from trident.patch_encoder_models import encoder_factory

parser = argparse.ArgumentParser()
parser.add_argument("--slide", default="data/wsis/patient_073_node_1.tif")
parser.add_argument("--coords_h5", default="trident_out/20x_256px_0px_overlap/patches/patient_073_node_1_patches.h5")
parser.add_argument("--n", type=int, default=50, help="number of random tiles")
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--encoder", default="uni_v1")
parser.add_argument("--out", default="outputs/sample_tiles_uni_v1.h5")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

attrs, coords = read_coords(args.coords_h5)
slide = openslide.OpenSlide(args.slide)
ps0 = attrs.get("patch_size_level0")
if ps0 is None:  # fallback: 20x = 0.5 um/px
    ps0 = round(args.patch_size * 0.5 / float(slide.properties["openslide.mpp-x"]))
ps0 = int(ps0)
print(f"{len(coords)} tiles in slide, tile size at level 0 = {ps0}px")

rng = np.random.default_rng(args.seed)
sample = coords[rng.choice(len(coords), size=min(args.n, len(coords)), replace=False)]

encoder = encoder_factory(args.encoder).eval().to("cpu")
png_dir = Path(args.out).with_suffix("")
png_dir.mkdir(parents=True, exist_ok=True)

feats = []
for x, y in sample:
    tile = slide.read_region((int(x), int(y)), 0, (ps0, ps0)).convert("RGB")
    tile = tile.resize((args.patch_size, args.patch_size))
    tile.save(png_dir / f"tile_{x}_{y}.png")
    with torch.inference_mode():
        feats.append(encoder(encoder.eval_transforms(tile).unsqueeze(0)).squeeze(0).float().numpy())

with h5py.File(args.out, "w") as f:
    f.create_dataset("features", data=np.stack(feats))
    f.create_dataset("coords", data=sample)
print(f"Saved {len(feats)} tile embeddings to {args.out} and PNGs to {png_dir}/")
