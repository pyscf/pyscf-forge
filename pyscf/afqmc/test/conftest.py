import importlib.util
import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
HAS_JAX = (
    importlib.util.find_spec("jax") is not None
    and importlib.util.find_spec("jaxlib") is not None
)


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))
    if path.parent != TEST_DIR:
        return False
    if path.name == "conftest.py":
        return False
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return False
    if sys.version_info[:2] < (3, 10):
        return True
    if path.name == "test_import.py":
        return False
    if not HAS_JAX:
        return True
    return False
