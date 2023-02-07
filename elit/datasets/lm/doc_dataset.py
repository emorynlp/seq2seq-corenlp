# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-21 22:16
from typing import Iterator, Any, Dict, Union, Callable, List

from elit.common.dataset import TransformSequentialDataset


class DocumentDataset(TransformSequentialDataset):

    def __init__(self, transform: Union[Callable, List] = None) -> None:
        """
        Datasets where documents are segmented by two newlines and sentences are separated by one newline.

        Args:
            transform:
        """
        super().__init__(transform)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        pass
