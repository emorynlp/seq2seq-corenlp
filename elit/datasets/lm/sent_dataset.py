# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-21 22:22
from typing import Iterator, Any, Dict, Union, Callable, List

from glob import glob

from elit.common.dataset import TransformSequentialDataset


class SentenceDataset(TransformSequentialDataset):
    def __init__(self, data, transform: Union[Callable, List] = None) -> None:
        """
        Datasets where documents are segmented by two newlines and sentences are separated by one newline.

        Args:
            transform:
        """
        super().__init__(transform)
        if isinstance(data, str):
            self.files = glob(data, recursive=True)
            assert self.files, f'No such file(s): {data}'
        else:
            self.files = None
            self.data = data

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for f in self.files:
            with open(f) as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    yield self.transform_sample({'sent': line}, inplace=True)
