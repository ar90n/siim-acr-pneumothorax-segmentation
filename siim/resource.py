import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import io

ROOT_DIR_PATH = Path(__file__).parent.parent
INPUT_DIR_PATH = os.environ.get("KAGGLE_INPUT_DIR", ROOT_DIR_PATH / ".input")
WEIGHTS_DIR_PATH = INPUT_DIR_PATH / "yolact"
TRAIN_DATA_DIR_PATH = (
    INPUT_DIR_PATH / "siim-acr-pneumothorax-segmentation" / "jpeg-images-train"
)
TEST_DATA_DIR_PATH = (
    INPUT_DIR_PATH / "siim-acr-pneumothorax-segmentation" / "jpeg-images-test"
)


def bbox(mask):
    loc = np.argwhere(mask)
    (ystart, xstart), (ystop, xstop) = loc.min(0), loc.max(0) + 1
    return [xstart, ystart, xstop, ystop]


class TrainDataset(Dataset):
    def __init__(self, transform=None, empty_mask_is_negative=False):
        self.transform = transform
        self.empty_mask_is_negative = empty_mask_is_negative
        self._paths = list(Path(TRAIN_DATA_DIR_PATH).glob("*.jpg"))

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        img, attr, pneumothorax_masks = io.read_jpg(
            self._paths[idx], self.empty_mask_is_negative)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=2)

        mask = [np.ones(img.shape[:2], dtype=np.float32)]
        boxes = [[0, 0, img.shape[1], img.shape[0]]]
        if pneumothorax_masks is not None:
            for pm in pneumothorax_masks:
                if np.count_nonzero(pm) == 0:
                    continue
                boxes.append(bbox(pm))
                mask.append(np.clip(pm, 0, 1.0).astype(np.float32))
        mask = np.stack(mask, axis=0)
        boxes = np.stack(boxes, axis=0).astype(np.float32)
        classes = np.clip(np.arange(boxes.shape[0], dtype=np.float32), 0, 1.0)

        if self.transform is not None:
            labels = {'num_crowds': 0}
            img, masks, boxes, _ = self.transform(img, mask, boxes, labels)

        img = torch.from_numpy(img).permute(2, 0, 1)
        target = np.hstack([boxes, np.expand_dims(classes, 0).T])
        mask = torch.from_numpy(mask)
        return img, (target, mask, 0)


class ScatterWrapper:
    """ Input is any number of lists. This will preserve them through a dataparallel scatter. """

    def __init__(self, *args, device=torch.device('cpu')):
        for arg in args:
            if not isinstance(arg, list):
                print("Warning: ScatterWrapper got input of non-list type.")
        self.args = args
        self.batch_size = len(args[0])
        self._device = device

    def make_mask(self):
        out = torch.Tensor(list(range(self.batch_size))).long()
        return out.to(device=self._device)

    def get_args(self, mask):
        device = mask.device
        mask = [int(x) for x in mask]
        out_args = [[] for _ in self.args]

        for out, arg in zip(out_args, self.args):
            for idx in mask:
                x = arg[idx]
                if isinstance(x, torch.Tensor):
                    x = x.to(device)
                out.append(x)

        return out_args
