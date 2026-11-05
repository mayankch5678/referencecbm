# referencecbm

## Goal

Classify WSI tiles using prototypes. TRIDENT's **UNI** patch encoder (`uni_v1`) embeds two things in the same feature space:

- tiles from a CAMELYON17 whole-slide image
- reference images from a pathology textbook, which act as class prototypes

Each tile gets the class of the prototype it is closest to:

```
class(tile) = argmax_i  cos_sim(tile, prototype_i)
```

## Scripts

| Script | What it does |
|---|---|
| `sample_and_encode_tiles.py` | Picks N random tiles from TRIDENT's patch coords, encodes them with UNI, and writes `features` and `coords` to an h5 file (plus the tile PNGs) |
| `encode_prototypes.py` | Encodes `data/prototypes/<class_name>/*.jpg` with the same encoder and writes them to `outputs/prototypes_uni_v1.npz` |
| `classify_tiles.py` | Computes cosine similarity between tiles and prototypes, assigns argmax classes, and writes `outputs/tile_predictions.csv` |

`data/`, `outputs/` and `trident_out/` are not in the repo. `data/` holds the copyrighted textbook images and the 4.3 GB CAMELYON17 slide.

## Setup

- Conda env `trident`, Python 3.10
- [TRIDENT](https://github.com/mahmoodlab/trident) cloned next to this repo at `../trident` and installed in editable mode:

  ```bash
  conda create -n trident python=3.10 -y
  conda activate trident
  git clone https://github.com/mahmoodlab/trident.git ../trident
  git -C ../trident checkout bdeeb74b7eaad8d48bda232bde7de66cae980918
  pip install -e ../trident
  ```

  TRIDENT commit used: `bdeeb74b7eaad8d48bda232bde7de66cae980918` (version 0.3.2)
- **UNI is a gated HuggingFace model.** Request access at https://huggingface.co/MahmoodLab/UNI, then run `huggingface-cli login` before encoding.

## Commands run so far

Run from this directory. The slide is `data/wsis/patient_073_node_1.tif`.

```bash
# 1. Tissue segmentation (Otsu, CPU, no DataLoader workers)
python ../trident/run_batch_of_slides.py --task seg \
    --wsi_dir data/wsis --job_dir trident_out \
    --segmenter otsu --max_workers 0 --gpus -1

# 2. Patch coordinates at 20x, 256 px
python ../trident/run_batch_of_slides.py --task coords \
    --wsi_dir data/wsis --job_dir trident_out \
    --mag 20 --patch_size 256 --max_workers 0 --gpus -1

# 3. Encode 50 random tiles with UNI
python sample_and_encode_tiles.py --n 50
```

Next steps:

```bash
# 4. Encode the textbook prototypes (data/prototypes/<class>/*.jpg)
python encode_prototypes.py

# 5. Classify the tiles
python classify_tiles.py --features_h5 outputs/sample_tiles_uni_v1.h5
```

## macOS notes

- `--max_workers 0` avoids a DataLoader segfault on macOS.
- The `otsu` segmenter replaces the default `hest` segmenter because HEST is too slow on CPU.
- `run_single_slide.py` needs CUDA, so these runs use `run_batch_of_slides.py` on CPU (`--gpus -1`).

## Status

- Slide `patient_073_node_1` has **12,476 tiles** at 20x/256 px, which is 512 px at level 0.
- 50 random tiles are encoded (`outputs/sample_tiles_uni_v1.h5`).
- **Open question:** which textbook images to use as the prototypes for each class.
