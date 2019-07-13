# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import json
import pydicom
import piexif
import csv
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

# %%
data_root = Path(os.environ.get('INPUT_PATH', '.')) / "siim-acr-pneumothorax-segmentation" / "jpeg-images-train"


# %%
def rle2mask(array, width, height):
    mask= np.zeros(width* height)
    if len(array) != 1:
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position+lengths[index]] = 255
            current_position += lengths[index]

    return mask.reshape(height, width).T


# %%
def read_jpg(path, empty_mask_is_negative=False):
    img = Image.open(path)
    makernote_bytes = piexif.load(img.info["exif"])["Exif"][piexif.ExifIFD.MakerNote]
    attr = json.loads(makernote_bytes.decode("ascii"))

    if empty_mask_is_negative:
        attr['Masks'] = attr.get('Masks', [[-1]])
        
    masks = None
    if 'Masks' in attr:
        masks = [rle2mask(encoded_pixels, img.width, img.height) for encoded_pixels in attr['Masks']]
        del attr['Masks']

    return np.asarray(img), attr, masks


# %%
for i, p in enumerate(data_root.glob('*.jpg')):
    pixel_array, attr, masks = read_jpg(p, True)
    plt.figure(i)
    fig, axs = plt.subplots(1, 1 + len(masks))
    axs[0].imshow(pixel_array)
    for j, m in enumerate(masks):
        axs[j+1].imshow(masks[j])
    if 8 < i:
        break

