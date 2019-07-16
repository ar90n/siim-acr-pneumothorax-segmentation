import torch

from . import config, io, resource  # noqa


def bootstrap():
    torch.set_default_tensor_type("torch.FloatTensor")
