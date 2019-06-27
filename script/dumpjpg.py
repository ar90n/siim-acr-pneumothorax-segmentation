import json
import sys

import piexif
from PIL import Image

try:
    src_path = sys.argv[-1]
except Exception:
    print("Usage: python dumpjpg.py <src>")
    exit()

img = Image.open(src_path)
makernote_bytes = piexif.load(img.info["exif"])["Exif"][piexif.ExifIFD.MakerNote]
attr = json.loads(makernote_bytes.decode("ascii"))

print("Image Size: {} x {}".format(*img.size))
print("StudyInstanceUID: {}".format(attr["StudyInstanceUID"]))
print("SeriesInstanceUID: {}".format(attr["SeriesInstanceUID"]))
print("SOPInstanceUID: {}".format(attr["SOPInstanceUID"]))
print("PatientSex: {}".format(attr["PatientSex"]))
print("PatientAge: {}".format(attr["PatientAge"]))
print("ViewPosition: {}".format(attr["ViewPosition"]))
print("PixelSpacing: {} x {}".format(*attr["PixelSpacing"]))
