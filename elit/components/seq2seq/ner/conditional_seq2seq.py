# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-01-18 14:49
import logging
from typing import Callable, List, Tuple

import torch
from hanlp_common.util import reorder

from hanlp_common.constant import IDX
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration
from transformers.generation_utils import GenerationMixin
from transformers.tokenization_utils import PreTrainedTokenizer

from elit.common.dataset import TransformableDataset, PadSequenceDataLoader, SortingSampler
from elit.common.torch_component import TorchComponent
from elit.layers.transformers.pt_imports import AutoTokenizer_
from elit.utils.time_util import CountdownTimer


class DummyDataset(TransformableDataset):

    def load_file(self, filepath: str):
        raise NotImplemented()


class ConditionalSeq2seq(TorchComponent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: GenerationMixin = None
        self.tokenizer: PreTrainedTokenizer = None

    def build_dataloader(self, data: List[Tuple[str, str]], batch_size=32, batch_max_tokens=None, verbose=False,
                         device=None,
                         logger: logging.Logger = None, **kwargs) -> DataLoader:
        dataset = DummyDataset([{'text': x, 'condition': c} for x, c in data], generate_idx=True, cache=True,
                               transform=self._transform)
        if verbose:
            verbose = CountdownTimer(len(dataset))
        lens = []
        for each in dataset:
            lens.append(len(each['input_ids']))
            if verbose:
                verbose.log('Preprocessing and caching samples [blink][yellow]...[/yellow][/blink]')
        dataloader = PadSequenceDataLoader(dataset,
                                           batch_sampler=SortingSampler(lens, batch_size=batch_size,
                                                                        batch_max_tokens=batch_max_tokens),
                                           device=device)
        return dataloader

    def _transform(self, sample: dict):
        sample['input_ids'] = self.tokenizer(sample['text'])['input_ids']
        sample['decoder_input_ids'] = self.tokenizer(sample['condition']).input_ids[:-1]
        return sample

    def build_optimizer(self, **kwargs):
        pass

    def build_criterion(self, **kwargs):
        pass

    def build_metric(self, **kwargs):
        pass

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, **kwargs):
        pass

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, **kwargs):
        pass

    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, **kwargs):
        pass

    def model_cls(self, **kwargs):
        return BartForConditionalGeneration

    def build_model(self, training=True, transformer=None, **kwargs) -> torch.nn.Module:
        return self.model_cls(**self.config).from_pretrained(transformer)

    def predict(self, src_cond_pairs: List[Tuple[str, str]], batch_size=32, batch_max_tokens=None, verbose=False,
                **kwargs):
        model = self.model
        tokenizer = self.tokenizer
        results = []
        orders = []
        dataloader = self.build_dataloader(src_cond_pairs, batch_size, batch_max_tokens, verbose, device=self.device)
        if verbose:
            verbose = CountdownTimer(len(dataloader))
        for batch in dataloader:
            orders.extend(batch[IDX])
            input_ids = batch['input_ids']
            decoder_input_ids = batch['decoder_input_ids']
            generated_tokens = model.generate(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                max_length=int(input_ids.size(1) * 2),
            )
            outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            results.extend(outputs)
            if verbose:
                verbose.log(f"[yellow]{batch['text'][0]}[/yellow] [magenta]{outputs[0]}[/magenta]")
        results = reorder(results, orders)
        return results

    def load_config(self, save_dir, filename='config.json', **kwargs):
        self.config.transformer = save_dir

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        pass

    def load_weights(self, save_dir, filename='model.pt', **kwargs):
        self.tokenizer = AutoTokenizer_.from_pretrained(save_dir, use_fast=True)
        self.model.config.tokenizer = self.tokenizer
