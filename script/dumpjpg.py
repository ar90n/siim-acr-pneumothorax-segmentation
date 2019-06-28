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
if "StudyInstanceUID" in attr:
    print("StudyInstanceUID: {}".format(attr["StudyInstanceUID"]))
if "SeriesInstanceUID" in attr:
    print("SeriesInstanceUID: {}".format(attr["SeriesInstanceUID"]))
if "SOPInstanceUID" in attr:
    print("SOPInstanceUID: {}".format(attr["SOPInstanceUID"]))
if "PatientSex" in attr:
    print("PatientSex: {}".format(attr["PatientSex"]))
if "PatientAge" in attr:
    print("PatientAge: {}".format(attr["PatientAge"]))
if "ViewPosition" in attr:
    print("ViewPosition: {}".format(attr["ViewPosition"]))
if "PixelSpacing" in attr:
    print("PixelSpacing: {} x {}".format(*attr["PixelSpacing"]))
if "Masks" in attr:
    print("Masks:")
    for i, mask in enumerate(attr["Masks"]):
        print("    {}: {}".format(i, ' '.join([str(v) for v in mask])))
