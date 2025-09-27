import contextlib
import numpy as np
import pandas as pd
import typing as tp
from _typeshed import Incomplete
from pathlib import Path

_NAMED_URLS: Incomplete

def get_cache_folder() -> Path:
    """Removes all cached datasets.
    This can be helpful in case of download issue
    """
def get_dataset_filepath(name: str) -> Path: ...
def get_data(name: str) -> np.ndarray | pd.DataFrame: ...
def _make_fake_get_data(name: str) -> np.ndarray | pd.DataFrame: ...
@contextlib.contextmanager
def mocked_data() -> tp.Iterator[tp.Any]:
    """Mocks all data that should be downloaded, in order to simplify testing"""
def make_perceptron_data(name: str) -> np.ndarray:
    """Creates the data (see https://drive.google.com/file/d/1fc1sVwoLJ0LsQ5fzi4jo3rDJHQ6VGQ1h/view)"""
