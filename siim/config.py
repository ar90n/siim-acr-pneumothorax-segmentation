from yolact.data.config import cfg, dataset_base, yolact_base_config

from .resource import TRAIN_DATA_DIR_PATH

siim_dataset = dataset_base.copy(
    {
        "name": "siim-acr-pneumothorax-segmentation",
        # Training images and annotations
        "train_images": TRAIN_DATA_DIR_PATH,
        "train_info": None,
        # Validation images and annotations.
        "valid_images": TRAIN_DATA_DIR_PATH,
        "valid_info": None,
        # A list of names for each of you classes.
        "class_names": ("normam", "pneumothorax"),
    }
)

yolact_siim_base_config = yolact_base_config.copy(
    {
        "name": "yolact_siim_base_config",
        "max_iter": 16,

        # Dataset stuff
        "dataset": siim_dataset,
        "num_classes": 2,

        # Custom parameters
        "batch_size": 1,
        "num_workers": 1,
        "save_interval": 10000,
        "keep_latest": False,
        "save_folder": "weights",
        "validation_epoch": 1
    }
)

cfg.replace(yolact_siim_base_config)
