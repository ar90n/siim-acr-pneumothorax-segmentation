import json

import numpy as np
from PIL import Image

import piexif


def rle2mask(array, width, height):
    mask = np.zeros(width * height)
    if len(array) != 1:
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position : current_position + lengths[index]] = 255
            current_position += lengths[index]

    return mask.reshape(height, width).T


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


def read_jpg(path, empty_mask_is_negative=False):
    img = Image.open(path)
    makernote_bytes = piexif.load(img.info["exif"])["Exif"][piexif.ExifIFD.MakerNote]
    attr = json.loads(makernote_bytes.decode("ascii"))

    if empty_mask_is_negative:
        attr["Masks"] = attr.get("Masks", [[-1]])

    masks = None
    if "Masks" in attr:
        masks = [
            rle2mask(encoded_pixels, img.width, img.height)
            for encoded_pixels in attr["Masks"]
        ]
        del attr["Masks"]

    return np.asarray(img), attr, masks
