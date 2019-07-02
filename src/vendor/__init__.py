import os
import sys
from contextlib import contextmanager
from pathlib import Path

from mock import Mock, patch


@contextmanager
def inject_path(path):
    sys.path.append(path)
    try:
        yield
    finally:
        sys.path.remove(path)


vendor_dir_path = str(Path(__file__).parent)

# import yolact
yolact_dir_path = str(Path(__file__).parent / 'yolact')
with patch('torch.cuda.current_device', Mock()), \
        patch('torch.cuda.device_count', Mock(return_value=0)), \
        inject_path(vendor_dir_path), \
        inject_path(yolact_dir_path):
    import yolact
