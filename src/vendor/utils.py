import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def inject_path(path):
    sys.path.append(path)
    try:
        yield
    finally:
        sys.path.remove(path)


VENDOR_DIR_PATH = Path(__file__).parent
SUBMODULES_DIR_PATH = VENDOR_DIR_PATH / 'submodules'
