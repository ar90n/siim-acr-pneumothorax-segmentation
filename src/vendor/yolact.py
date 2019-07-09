from mock import Mock, patch

from .utils import SUBMODULES_DIR_PATH, inject_path

# import yolact
YOLACT_DIR_PATH = SUBMODULES_DIR_PATH / 'yolact'
with patch('torch.cuda.current_device', Mock()), \
        patch('torch.cuda.device_count', Mock(return_value=0)), \
        inject_path(str(YOLACT_DIR_PATH)):
    from vendor.submodules.yolact import (
        backbone,
        eval,
        layers,
        utils,
        web,
        yolact
    )
