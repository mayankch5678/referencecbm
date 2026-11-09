# ReferenceCBM

Prototype-based, explainable comparison of whole-slide-image (WSI) tiles with reference images
from a lymphoma pathology textbook. Master's practical, Computational Pathology group,
Heidelberg University Hospital. Work in progress.

## Pipeline

1. TRIDENT segments the slide and cuts it into tiles (coordinates only).
2. `sample_and_encode_tiles.py` encodes a random sample of tiles with the UNI foundation model.
3. `build_book_index.py` reads the textbook OCR JSON files, extracts the images per chapter and
   pairs each image with its figure caption (`outputs/book_index.csv`).
4. `compare_tile_to_book.py` encodes the book images with the same encoder, computes the cosine
   similarity between one tile and every book image, and assigns the caption of the best match.
5. `hubness_check.py` counts how often each book image reaches a tile's top 5 across all sampled tiles.

`encode_prototypes.py` and `classify_tiles.py` are an earlier variant: one folder of reference
images per class, class = argmax cosine similarity.

## Setup

- Python 3.10, TRIDENT cloned next to this repository and installed with `pip install -e ../trident`.
- UNI is a gated HuggingFace model; request access and log in with `hf auth login`.

## Commands (as run on an Apple Silicon laptop, CPU)

    python ../trident/run_batch_of_slides.py --task seg --wsi_dir data/wsis --job_dir trident_out --gpus -1 --max_workers 0 --segmenter otsu
    python ../trident/run_batch_of_slides.py --task coords --wsi_dir data/wsis --job_dir trident_out --mag 20 --patch_size 256 --overlap 0 --max_workers 0
    python sample_and_encode_tiles.py --n 50
    python build_book_index.py
    python compare_tile_to_book.py --tile_index 26
    python hubness_check.py

`--max_workers 0` avoids a DataLoader segmentation fault on macOS. Otsu replaces the HEST
segmenter, which is too slow on CPU.

## Data (not included)

- A CAMELYON17 slide (test data, not lymphoma) in `data/wsis/`.
- The textbook OCR output (JSON files and images) in `data/book/`. The book is copyrighted and
  is not redistributed here.

## Status

- Test slide patient_073_node_1: 12,476 tiles at 20x and 256 px, 50 random tiles encoded.
- Textbook index: 665 images, 596 with a caption.
- On this non-lymphoma slide the similarity scores are low and the best matches are not
  meaningful. Lymphoma slides are needed for a real test.

TRIDENT commit used: bdeeb74b7eaad8d48bda232bde7de66cae980918
