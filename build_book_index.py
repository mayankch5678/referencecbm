"""Build an image -> caption index from the textbook's OCR JSON files.

Per chapter JSON:
  1. extract the embedded images (image_base64) to data/book_images/ch<N>_<img-id>
     (ids restart at img-0 in every chapter, so the chapter number makes them unique);
  2. walk through each page's markdown in reading order and assign captions to images:
       - a caption claims ALL still-unclaimed images before it (multi-panel figures:
         panels A-F first, one "FIGURE 3-1 ..." caption after them)    -> match = single / grouped
       - a caption that finds no image before it takes the next image(s) on the page
         (OCR sometimes puts the image after its caption)               -> match = trailing
       - images left at the end of a page are offered to a caption that
         opens the next page (figure continued across pages)            -> match = carried
       - everything else stays without caption                          -> match = none
  3. write outputs/book_index.csv (one row per image).
"""
import argparse
import base64
import csv
import glob
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--book_dir", default="data/book")
parser.add_argument("--img_out", default="data/book_images")
parser.add_argument("--csv_out", default="outputs/book_index.csv")
args = parser.parse_args()

IMG_REF = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
FIG_LINE = re.compile(r"^\s*(FIGURE|FIG\.?)\s*(\d+\s*[-–.]\s*\d+[A-Za-z]?)\.?\s*(.*)$", re.IGNORECASE)

Path(args.img_out).mkdir(parents=True, exist_ok=True)
Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)


def parse_page(md):
    """Ordered events: ('img', id) or ('cap', {figure, title, desc})."""
    events, cur = [], None
    for raw in md.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        ref = IMG_REF.search(ln)
        if ref:
            events.append(("img", ref.group(1)))
            cur = None
            continue
        fm = FIG_LINE.match(ln)
        if fm:
            cur = {"figure": f"FIGURE {fm.group(2).replace(' ', '')}",
                   "title": fm.group(3).strip(), "desc": []}
            events.append(("cap", cur))
        elif cur is not None and not re.fullmatch(r"\d{1,3}", ln):   # skip bare page numbers
            cur["desc"].append(ln)
    return events


def assign(rows_group, cap, how):
    n = len(rows_group)
    for k, r in enumerate(rows_group, start=1):
        r.update(figure=cap["figure"], title=cap["title"], description=" ".join(cap["desc"]),
                 has_caption=True, match=how if n == 1 or how != "single" else "single",
                 panel=k, n_panels=n)
        if how == "single" and n > 1:
            r["match"] = "grouped"


all_rows, n_orphan_caps = [], 0
for jf in sorted(glob.glob(f"{args.book_dir}/*_ocr_raw.json")):
    m = re.search(r"Chapter (\d+)", jf)
    chapter = m.group(1) if m else Path(jf).stem.split(" - ")[-1].replace("_ocr_raw", "")
    pending = []                                   # unclaimed image rows from the previous page
    for page in json.load(open(jf))["pages"]:
        page_no = page["index"] + 1
        info = {}
        for im in page.get("images", []):
            info[im["id"]] = (im["bottom_right_x"] - im["top_left_x"], im["bottom_right_y"] - im["top_left_y"])
            b64 = im.get("image_base64")
            if b64:
                if "," in b64[:80]:
                    b64 = b64.split(",", 1)[1]
                (Path(args.img_out) / f"ch{chapter}_{im['id']}").write_bytes(base64.b64decode(b64))

        events = parse_page(page["markdown"])
        unclaimed = pending if (events and events[0][0] == "cap") else []
        carried_ids = {id(r) for r in unclaimed}
        orphan_caps = []
        for kind, val in events:
            if kind == "img":
                w, h = info.get(val, (0, 0))
                row = {"chapter": chapter, "page": page_no, "image_id": val, "file": f"ch{chapter}_{val}",
                       "width": w, "height": h, "figure": "", "title": "", "description": "",
                       "has_caption": False, "match": "none", "panel": 0, "n_panels": 0}
                all_rows.append(row)
                unclaimed.append(row)
            else:
                if unclaimed:
                    how = "carried" if any(id(r) in carried_ids for r in unclaimed) else "single"
                    assign(unclaimed, val, how)
                    unclaimed = []
                else:
                    orphan_caps.append(val)
        # captions that had no image before them take the images that follow on the page
        for cap in list(orphan_caps):
            if unclaimed:
                assign([unclaimed.pop(0)], cap, "trailing")
                orphan_caps.remove(cap)
        n_orphan_caps += len(orphan_caps)
        pending = unclaimed                          # offered to the next page's first caption

fields = ["chapter", "page", "image_id", "file", "width", "height", "figure", "title",
          "description", "has_caption", "match", "panel", "n_panels"]
with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(all_rows)

from collections import Counter
cnt = Counter(r["match"] for r in all_rows)
print(f"{len(all_rows)} image references in {len({r['chapter'] for r in all_rows})} chapters")
print("match types:", dict(cnt))
print(f"captions without any image: {n_orphan_caps}")
print(f"Images extracted to {args.img_out}/, index written to {args.csv_out}")