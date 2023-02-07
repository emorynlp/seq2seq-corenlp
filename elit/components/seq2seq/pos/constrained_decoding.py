# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-11-10 16:05
import math
from enum import Enum

import torch
from hanlp_trie import Trie
from transformers import BartTokenizer
from transformers.generation_logits_process import LogitsProcessor

from elit.utils.log_util import cprint


class TagProcessor(LogitsProcessor):
    def __init__(self, batch, tags, tokenizer: BartTokenizer):
        self.tokenizer = tokenizer
        self.tags = tags
        self.batch = batch
        self.eos = tokenizer.eos_token_id
        self.offsets = [0] * len(batch['token'])
        self.batch['_predictions'] = [[] for _ in batch['token']]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        batch = self.batch
        for batch_id, beam_sent in enumerate(input_ids.view(-1, 1, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * 1 + beam_id
                # prefix_str = self.tokenizer.convert_ids_to_tokens(input_ids[index].tolist())
                tokens = batch['token'][index]
                if self.offsets[index] < len(tokens):
                    self.offsets[index] += 1
                    allowed_tokens.update(self.tags)
                else:
                    allowed_tokens.add(self.eos)
                    tag_ids = input_ids[index].tolist()
                    tags = [self.tags[x] for x in tag_ids if x in self.tags]
                    assert len(tags) == len(tokens)
                    batch['_predictions'][index] = tags
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask


class TokenTagProcessor(LogitsProcessor):
    def __init__(self, batch, tags, tokenizer: BartTokenizer):
        self.tokenizer = tokenizer
        self.tags = tags
        self.batch = batch
        self.eos = tokenizer.eos_token_id
        self.token_offsets = [0] * len(batch['token'])
        self.subtoken_offsets = [1] * len(batch['token'])
        self.prev_is_tail = [False] * len(batch['token'])
        self.tag_offsets = [[] for _ in batch['token']]
        self.batch['_predictions'] = [[] for _ in batch['token']]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        batch = self.batch
        for batch_id, beam_sent in enumerate(input_ids.view(-1, 1, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * 1 + beam_id
                prefix_ids = input_ids[index].tolist()
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)
                subtoken_to_token = self.batch['subtoken_to_token'][index]
                tokens = self.batch['token'][index]
                text_token_ids = self.batch['text_token_ids']
                encoder_input_ids = text_token_ids[index].tolist()
                tails = [i + 1 > len(subtoken_to_token) - 1 or x != subtoken_to_token[i + 1] for i, x in
                         enumerate(subtoken_to_token)]
                if self.token_offsets[index] <= len(tokens):
                    if self.prev_is_tail[index]:
                        allowed_tokens.update(self.tags)
                        self.tag_offsets[index].append(len(prefix_ids))
                        self.prev_is_tail[index] = False
                    else:
                        if tails[self.subtoken_offsets[index]]:
                            self.prev_is_tail[index] = True
                            self.token_offsets[index] += 1
                            if self.token_offsets[index] > len(tokens):
                                tags = [self.tags[prefix_ids[x]] for x in self.tag_offsets[index]]
                                assert len(tags) == len(tokens)
                                batch['_predictions'][index] = tags
                        allowed_tokens.add(encoder_input_ids[self.subtoken_offsets[index]])
                        self.subtoken_offsets[index] += 1
                else:
                    allowed_tokens.add(self.eos)
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask


class IsAProcessor(LogitsProcessor):
    def __init__(self, batch, trie: Trie, tokenizer: BartTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.trie = trie
        self.batch = batch
        self.eos = tokenizer.eos_token_id
        self.token_offsets = [0] * len(batch['token'])
        self.subtoken_offsets = [1] * len(batch['token'])
        self.prev_is_tail = [False] * len(batch['token'])
        self.tag_offsets = [[] for _ in batch['token']]
        self.batch['_predictions'] = [[] for _ in batch['token']]
        self.token_generated = [False] * len(batch['token'])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        batch = self.batch
        for batch_id, beam_sent in enumerate(input_ids.view(-1, 1, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * 1 + beam_id
                tokens = self.batch['token'][index]
                if len(batch['_predictions'][index]) == len(tokens):
                    continue
                prefix_ids = input_ids[index].tolist()
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)
                subtoken_to_token = self.batch['subtoken_to_token'][index]
                text_token_ids = self.batch['text_token_ids']
                encoder_input_ids = text_token_ids[index].tolist()
                tails = [i + 1 > len(subtoken_to_token) - 1 or x != subtoken_to_token[i + 1] for i, x in
                         enumerate(subtoken_to_token)]
                if self.token_generated[index]:
                    label_part = prefix_ids[self.token_generated[index]:]
                    node = self.trie.transit(label_part)
                    if not node:
                        print(tokens)
                        raise RuntimeError('Something went wrong with tokenization.')
                    children = node._children
                    if children:
                        allowed_tokens.update(children)
                    else:
                        self.token_generated[index] = False
                        batch['_predictions'][index].append(node._value)
                        if len(batch['_predictions'][index]) == len(tokens):
                            allowed_tokens.add(self.eos)
                if not self.token_generated[index]:
                    if tails[self.subtoken_offsets[index]]:
                        self.prev_is_tail[index] = True
                        self.token_offsets[index] += 1
                        self.token_generated[index] = len(prefix_ids) + 1
                    else:
                        self.token_generated[index] = False
                    allowed_tokens.add(encoder_input_ids[self.subtoken_offsets[index]])
                    self.subtoken_offsets[index] += 1
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask


class IsAQuotationStatus(Enum):
    LEFT_QUOTATION = 0
    TOKEN = 1
    RIGHT_QUOTATION = 2
    LABEL = 3


class IsAProcessorQuotation(LogitsProcessor):
    def __init__(self, batch, trie: Trie, tokenizer: BartTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.trie = trie
        self.batch = batch
        self.eos = tokenizer.eos_token_id
        self.token_offsets = [0] * len(batch['token'])
        self.subtoken_offsets = [1] * len(batch['token'])
        self.prev_is_tail = [False] * len(batch['token'])
        self.tag_offsets = [[] for _ in batch['token']]
        self.batch['_predictions'] = [[] for _ in batch['token']]
        self.status = [None] * len(batch['token'])
        ids = self.tokenizer(' " good "', add_special_tokens=False).input_ids
        self.lq = ids[0]
        self.rq = ids[-1]
        self.token_generated = [-1] * len(batch['token'])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        batch = self.batch
        for batch_id, beam_sent in enumerate(input_ids.view(-1, 1, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * 1 + beam_id
                tokens = self.batch['token'][index]
                if len(batch['_predictions'][index]) == len(tokens):
                    continue
                prefix_ids = input_ids[index].tolist()
                debug = False
                if debug:
                    prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)
                subtoken_to_token = self.batch['subtoken_to_token'][index]
                text_token_ids = self.batch['text_token_ids']
                encoder_input_ids = text_token_ids[index].tolist()
                tails = [i + 1 > len(subtoken_to_token) - 1 or x != subtoken_to_token[i + 1] for i, x in
                         enumerate(subtoken_to_token)]
                if len(prefix_ids) == 1:
                    allowed_tokens.add(self.lq)
                    self.status[index] = IsAQuotationStatus.LEFT_QUOTATION
                elif self.status[index] == IsAQuotationStatus.LEFT_QUOTATION:
                    allowed_tokens.add(encoder_input_ids[self.subtoken_offsets[index]])
                    self.subtoken_offsets[index] += 1
                    self.status[index] = IsAQuotationStatus.TOKEN
                elif self.status[index] == IsAQuotationStatus.TOKEN:
                    if tails[self.subtoken_offsets[index] - 1]:
                        allowed_tokens.add(self.rq)
                        self.status[index] = IsAQuotationStatus.RIGHT_QUOTATION
                        self.token_generated[index] = len(prefix_ids) + 1
                    else:
                        allowed_tokens.add(encoder_input_ids[self.subtoken_offsets[index]])
                        self.subtoken_offsets[index] += 1
                elif self.status[index] == IsAQuotationStatus.RIGHT_QUOTATION:
                    label_part = prefix_ids[self.token_generated[index]:]
                    node = self.trie.transit(label_part)
                    if not node:
                        # print(tokens)
                        raise RuntimeError('Something went wrong with tokenization.')
                    children = node._children
                    if children:
                        allowed_tokens.update(children)
                    else:
                        batch['_predictions'][index].append(node._value)
                        if len(batch['_predictions'][index]) == len(tokens):
                            allowed_tokens.add(self.eos)  # don't really bother to generate the last quotation
                        else:
                            allowed_tokens.add(self.lq)
                            self.status[index] = IsAQuotationStatus.LEFT_QUOTATION
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                if debug:
                    cprint(f'[green]{len(prefix_ids)}[/green] {self.tokenizer.decode(prefix_ids)}[yellow] '
                           f'{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask


def main():
    pass


if __name__ == '__main__':
    main()
