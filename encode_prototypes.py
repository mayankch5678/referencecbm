"""Step 3 of Christoph's task: embed the chosen book reference images as prototypes.

Expects one folder per class:  data/prototypes/<class_name>/*.jpg
Uses the SAME TRIDENT encoder as the WSI tiles, so both live in one feature space.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from trident.patch_encoder_models import encoder_factory

parser = argparse.ArgumentParser()
parser.add_argument("--proto_dir", default="data/prototypes")
parser.add_argument("--encoder", default="uni_v1")
parser.add_argument("--out", default="outputs/prototypes_uni_v1.npz")
args = parser.parse_args()

encoder = encoder_factory(args.encoder).eval().to("cpu")
transform = encoder.eval_transforms  # same preprocessing TRIDENT uses for tiles

features, labels, files = [], [], []
for class_dir in sorted(p for p in Path(args.proto_dir).iterdir() if p.is_dir()):
    for img_path in sorted(class_dir.glob("*.jp*g")):
        img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            feat = encoder(img).squeeze(0).float().numpy()
        features.append(feat)
        labels.append(class_dir.name)
        files.append(str(img_path))
        print(f"{class_dir.name:25s} {img_path.name}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, features=np.stack(features), labels=np.array(labels), files=np.array(files))
print(f"Saved {len(features)} prototypes to {args.out}")
