# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-17 19:53
import os
import tempfile
from typing import Iterable

import torch

from elit.utils.io_util import merge_files
from elit.utils.time_util import CountdownTimer


class FileCache(object):
    def __init__(self, filename=None, delete=True) -> None:
        self.delete = delete
        if not filename:
            filename = tempfile.NamedTemporaryFile(prefix='elit-cache-', suffix='.pkl', delete=delete).name
        self._filename = filename

    def close(self):
        if self.delete:
            if os.path.isfile(self._filename):
                os.remove(self._filename)

    def __del__(self):
        self.close()


class RandomAccessFileCache(FileCache):
    def __init__(self, filename=None) -> None:
        super().__init__(filename)
        self.fp = open(filename, 'wb+')
        self.offsets = dict()

    def __setitem__(self, key, value):
        # Always write to the end of file
        self.fp.seek(0, 2)
        start = self.fp.tell()
        torch.save(value, self.fp, _use_new_zipfile_serialization=False)
        self.offsets[key] = start

    def __getitem__(self, key):
        offset = self.offsets.get(key, None)
        assert offset is not None, f'{key} does not exist in the cache'
        self.fp.seek(offset)
        return torch.load(self.fp)

    def __contains__(self, key):
        return key in self.offsets

    def close(self):
        if self.fp:
            self.fp.close()
            self.fp = None
        super().close()

    def __len__(self):
        return len(self.offsets)


class SequentialFileCache(FileCache):
    def __init__(self, iterator: Iterable = None, size=None, filename=None, delete=True, device=None) -> None:
        super().__init__(filename, delete)
        if isinstance(device, int):
            device = torch.device(f'cuda:{device}' if device >= 0 else torch.device('cpu'))
        self.device = device
        self.size = size
        # If the cache is already there then load the size and return
        if not delete and filename and os.path.isfile(filename):
            if not size:
                with open(self._filename, "rb") as f:
                    self.size = torch.load(f)
            return
        os.makedirs(os.path.dirname(self._filename), exist_ok=True)
        # Otherwise generate the cache
        timer = CountdownTimer(size) if size else None
        with open(self._filename, "wb") as f:
            if size:
                torch.save(size, f, _use_new_zipfile_serialization=False)
            for i, batch in enumerate(iterator):
                torch.save(batch, f, _use_new_zipfile_serialization=False)
                if timer:
                    timer.log(f'Caching {self._filename} [blink][yellow]...[/yellow][/blink]', erase=True)
                self.size = i + 1
        if not size:
            _content = self._filename + '.content'
            os.rename(self._filename, _content)
            _index = self._filename + '.index'
            with open(_index, "wb") as f:
                torch.save(self.size, f, _use_new_zipfile_serialization=False)
            merge_files([_index, _content], self._filename)
            os.remove(_content)
            os.remove(_index)

    def __iter__(self):
        with open(self._filename, "rb") as f:
            self.size = torch.load(f)
            for i in range(self.size):
                batch = torch.load(f, map_location=self.device)
                yield batch

    def __len__(self):
        return self.size
