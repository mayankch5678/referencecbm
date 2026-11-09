import h5py
from PIL import Image, ImageDraw

f = h5py.File("outputs/sample_tiles_uni_v1.h5", "r")
coords = f["coords"][:]
cols, size = 10, 128
rows = (len(coords) + cols - 1) // cols
sheet = Image.new("RGB", (cols * size, rows * size), "white")

for i, (x, y) in enumerate(coords):
    im = Image.open(f"outputs/sample_tiles_uni_v1/tile_{x}_{y}.png").convert("RGB").resize((size, size))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 28, 14], fill="black")
    d.text((2, 1), str(i), fill="white")
    sheet.paste(im, ((i % cols) * size, (i // cols) * size))

sheet.save("outputs/contact_sheet.png")
print("Saved outputs/contact_sheet.png")
