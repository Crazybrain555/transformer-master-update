from torch.utils.data import IterableDataset
import os
from typing import Tuple, Union
from functools import partial
from torchdata.datapipes.iter import FileOpener, IterableWrapper
import os
from functools import partial
from typing import Union, Tuple

from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext._download_hooks import GDriveReader  # noqa
from torchtext._download_hooks import HttpReader
from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)

DATASET_NAME = "new_Multi30k"
def _filepath_fn(root, split, language_pair,_=None):
    return os.path.join(root, f"{split}.{language_pair[0]}"), os.path.join(root,f"{split}.{language_pair[1]}")

@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "val", "test"))
def newMulti30k(root: str, split: Union[Tuple[str], str], language_pair: Tuple[str] = ("de", "en")):
    """Multi30k dataset

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: ('train', 'valid', 'test')
        language_pair: tuple or list containing src and tgt language. Available options are ('de','en') and ('en', 'de')

    :return: DataPipe that yields tuple of source and target sentences
    :rtype: (str, str)
    """
    assert len(language_pair) == 2, "language_pair must contain only 2 elements: src and tgt language respectively"
    assert tuple(sorted(language_pair)) == (
        "de",
        "en",
    ), "language_pair must be either ('de','en') or ('en', 'de')"

    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    src_file, tgt_file = _filepath_fn(root, split,language_pair)

    src_data_dp = FileOpener([src_file], encoding="utf-8").readlines(
        return_path=False, strip_newline=True
    )
    tgt_data_dp = FileOpener([tgt_file], encoding="utf-8").readlines(
        return_path=False, strip_newline=True
    )

    return src_data_dp.zip(tgt_data_dp).shuffle().set_shuffle(False).sharding_filter()
