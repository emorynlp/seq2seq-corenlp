# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-14 12:46
from collections import defaultdict, deque


class MovingAverage(object):
    def __init__(self, maxlen=5) -> None:
        self._queue = defaultdict(lambda: deque(maxlen=maxlen))

    def append(self, key, value: float):
        self._queue[key].append(value)

    def average(self, key) -> float:
        queue = self._queue[key]
        return sum(queue) / len(queue)
