# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-03 00:16
from collections.abc import MutableMapping

from elit.metrics.metric import Metric


class MetricDict(Metric, MutableMapping):
    _COLORS = ["magenta", "cyan", "green", "yellow"]

    def __init__(self, *args, primary_key=None, **kwargs) -> None:
        self.store = dict(*args, **kwargs)
        self.primary_key = primary_key

    @property
    def score(self):
        return float(self[self.primary_key]) if self.primary_key else sum(float(x) for x in self.values()) / len(self)

    def __call__(self, pred, gold):
        for metric in self.values():
            metric(pred, gold)

    def reset(self):
        for metric in self.values():
            metric.reset()

    def __repr__(self) -> str:
        return ' '.join(f'({k} {v})' for k, v in self.items())

    def cstr(self, idx=None, level=0) -> str:
        if idx is None:
            idx = [0]
        prefix = ''
        for _, (k, v) in enumerate(self.items()):
            color = self._COLORS[idx[0] % len(self._COLORS)]
            idx[0] += 1
            child_is_dict = isinstance(v, MetricDict)
            _level = min(level, 2)
            # if level != 0 and not child_is_dict:
            #     _level = 2
            lb = '{[('
            rb = '}])'
            k = f'[bold][underline]{k}[/underline][/bold]'
            prefix += f'[{color}]{lb[_level]}{k} [/{color}]'
            if child_is_dict:
                prefix += v.cstr(idx, level + 1)
            else:
                prefix += f'[{color}]{v}[/{color}]'
            prefix += f'[{color}]{rb[_level]}[/{color}]'
        return prefix

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)
