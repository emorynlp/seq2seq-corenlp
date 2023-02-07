# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-11-24 20:31
from typing import Union, List, Callable

from elit.common.dataset import TransformableDataset
from elit.utils.io_util import load_jsonl


class DetokenizationDataset(TransformableDataset):

    def __init__(self, data: Union[str, List], transform: Union[Callable, List] = None, cache=None,
                 generate_idx=None, **kwargs) -> None:
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath: str):
        for sample in load_jsonl(filepath):
            text = sample['text']
            offsets = sample['offsets']
            tokens = [text[x[0]:x[1]] for x in offsets]
            spaces = [' ' if x.isspace() else '' for x in text] + ['']
            tags = [spaces[x[1]] for x in offsets]
            yield {'token': tokens, 'tag': tags}
