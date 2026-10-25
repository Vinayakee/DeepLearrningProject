"""Acquire a staged directory, never fabricate observations or edit the archive."""
import importlib
import os
import time
from urllib.error import URLError
from pathlib import Path


class AcquisitionUnavailable(ValueError):
    pass


def acquire_with_retry(directory=None, attempts=3):
    """Retry transport failures only; invalid schemas/configuration fail immediately."""
    if not 1 <= attempts <= 5:
        raise ValueError('Acquisition attempts must be between one and five.')
    for attempt in range(attempts):
        try:
            return acquire(directory)
        except (TimeoutError, ConnectionError, URLError):
            if attempt == attempts-1:
                raise
            time.sleep(min(2**attempt, 5))


def acquire(directory=None):
    """Source plugins return a real QC bundle directory; credentials stay in env."""
    if directory is None:
        directory = os.getenv('HAB_LIVE_INPUT_DIR')
    if not directory:
        adapter = os.getenv('HAB_LIVE_ADAPTER', '')
        if not adapter:
            raise AcquisitionUnavailable('Set HAB_LIVE_INPUT_DIR or HAB_LIVE_ADAPTER=module:function.')
        module, separator, function = adapter.partition(':')
        if not separator or not module or not function:
            raise ValueError('Acquisition adapter must be module:function.')
        directory = getattr(importlib.import_module(module), function)()
    if not directory:
        raise AcquisitionUnavailable('Acquisition adapter returned no observations.')
    path = Path(directory).expanduser().resolve()
    if not all((path / name).is_file() for name in ('inputs.npz', 'metadata.json')):
        raise AcquisitionUnavailable('Real staged inputs.npz and metadata.json are required.')
    return path
