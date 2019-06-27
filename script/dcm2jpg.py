import json
import sys

import piexif
import pydicom
from PIL import Image

try:
    src_path = sys.argv[-2]
    dst_path = sys.argv[-1]
except Exception:
    print('Usage: python dcm2jpg.py <src> <dst>')
    exit()


dcm = pydicom.dcmread(src_path)
attr = {
    "StudyInstanceUID": dcm.StudyInstanceUID,
    "SeriesInstanceUID": dcm.SeriesInstanceUID,
    "SOPInstanceUID": dcm.SOPInstanceUID,
    "PatientSex": dcm.PatientSex,
    "PatientAge": int(dcm.PatientAge),
    "ViewPosition": dcm.ViewPosition,
    "PixelSpacing": [float(s) for s in dcm.PixelSpacing],
}

img = Image.fromarray(dcm.pixel_array)
exif_ifd = {piexif.ExifIFD.MakerNote: json.dumps(attr).encode("ascii")}
exif = {"Exif": exif_ifd}
img.save(dst_path, format='jpeg', exif=piexif.dump(exif))
