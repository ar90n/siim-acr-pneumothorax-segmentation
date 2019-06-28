import sys
import json
import csv
from pathlib import Path

import piexif


try:
    csv_path = sys.argv[-2]
    root = sys.argv[-1]
except Exception:
    print('Usage: python addmask.py <csv> <root>')
    exit()

labels = {}
with open(csv_path) as fp:
    for row in csv.DictReader(fp):
        label = [int(v) for v in row[' EncodedPixels'].split(' ') if v != '']
        labels.setdefault(row['ImageId'], []).append(label)

for p in Path(root).glob('*.jpg'):
    if p.stem in labels:
        exif_dict = piexif.load(str(p))
        exif = json.loads(exif_dict["Exif"][piexif.ExifIFD.MakerNote].decode("ascii"))
        exif["Masks"] = labels[p.stem]
        exif_dict["Exif"][piexif.ExifIFD.MakerNote] = json.dumps(exif).encode("ascii")
        piexif.insert(piexif.dump(exif_dict), str(p))
