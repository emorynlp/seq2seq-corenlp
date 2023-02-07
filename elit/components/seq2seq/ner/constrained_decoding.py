# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-11-10 16:05
import copy
import math
from typing import List, Dict

import torch

from elit.metrics.chunking.sequence_labeling import get_entities
from hanlp_trie import Trie
from transformers import BartTokenizer, BartTokenizerFast
from transformers.generation_logits_process import LogitsProcessor

from elit.utils.log_util import cprint


def first_index_of(haystack: List, needle, offset=0):
    for i in range(offset, len(haystack)):
        if haystack[i:i + len(needle)] == needle:
            return i


def indices_of(haystack: List, needle, offset=0):
    begin = offset
    indices = []
    while begin < len(haystack):
        begin = first_index_of(haystack, needle, begin)
        if begin is None:
            break
        indices.append(begin)
        begin += len(needle)
    return indices


class CopyPrefixConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(self,
                 encoder_input_ids,
                 labels: Trie,
                 separator,
                 eos,
                 pad,
                 sep,
                 delimiter,
                 batch,
                 num_beams: int = 1,
                 tokenizer=None):
        self.sep = sep
        self.delimiter = delimiter
        self.tokenizer: BartTokenizer = tokenizer
        self.pad = pad
        self.eos = eos
        self.separator = separator
        self.labels = labels
        self.encoder_input_ids = encoder_input_ids
        self._num_beams = num_beams
        self.last_src_indices = [1 if isinstance(tokenizer, BartTokenizerFast) else 0] * (
                num_beams * len(encoder_input_ids))
        self.last_sep_indices = [0] * (num_beams * len(encoder_input_ids))
        self.batch = batch

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * self._num_beams + beam_id
                subtoken_to_token = self.batch['subtoken_to_token'][batch_id]
                tokens = self.batch['token'][batch_id]
                text_token_ids_ = self.batch.get('text_token_ids_', None)
                if text_token_ids_:
                    encoder_input_ids = text_token_ids_[batch_id]
                else:
                    encoder_input_ids = self.encoder_input_ids[batch_id]
                # if len(encoder_input_ids) != len(subtoken_to_token):
                #     print()
                heads = [i == 0 or x != subtoken_to_token[i - 1] for i, x in enumerate(subtoken_to_token)]
                tails = [i + 1 > len(subtoken_to_token) - 1 or x != subtoken_to_token[i + 1] for i, x in
                         enumerate(subtoken_to_token)]
                encoder_input_ids = encoder_input_ids[:len(subtoken_to_token)]
                prefix_ids: list = input_ids[index].tolist()
                # Last token is separator, meaning a NE is generated so update states
                if prefix_ids and prefix_ids[-1] == self.separator:
                    last_prompt = prefix_ids[self.last_sep_indices[index] + 1:]
                    src, label, node = self.segment_src_label(last_prompt)
                    if node._value:
                        src_offset = first_index_of(encoder_input_ids, src, self.last_src_indices[index])
                        if src_offset is None:
                            cprint(f'[red]src_offset is None[/red], {tokens}')
                        else:
                            self.last_src_indices[index] = src_offset + len(src)
                        allowed_tokens.add(self.eos)  # We can choose to terminate
                    self.last_sep_indices[index] = len(prefix_ids) - 1

                prefix_ids = prefix_ids[self.last_sep_indices[index] + 1:]
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)

                if not prefix_ids:
                    # allow any src tokens except for bos and eos
                    allowed_tokens.update([i for i, h in zip(encoder_input_ids[self.last_src_indices[index]:-1],
                                                             heads[self.last_src_indices[index]:-1]) if h])
                    allowed_tokens.add(self.eos)
                else:
                    maybe_tail = False
                    sure_label = True
                    for src_offset in indices_of(encoder_input_ids, prefix_ids, self.last_src_indices[index]):
                        sure_label = False
                        # Copy next token from src. len(encoder_input_ids) - 1 to exclude eos
                        if src_offset is not None:
                            if src_offset + len(prefix_ids) < len(encoder_input_ids) - 1:
                                allowed_tokens.add(encoder_input_ids[src_offset + len(prefix_ids)])
                                # if not maybe_tail and src_offset + len(prefix_ids) - 1 >= len(tails):
                                #     print(src_offset)
                                #     print(prefix_ids)
                                #     print(tails)
                                maybe_tail = maybe_tail or tails[src_offset + len(prefix_ids) - 1]
                            else:
                                maybe_tail = True
                    if prefix_ids != [self.delimiter]:
                        # Leading space is not considered as an entity
                        if maybe_tail or sure_label:
                            src, label, node = self.segment_src_label(prefix_ids)
                            allowed_tokens.update(list(node._children))

                allowed_tokens.discard(self.pad)
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask

    def segment_src_label(self, prefix_ids):
        """
            Split prefix into src + label

        Args:
            prefix_ids:

        Returns:
            src, label, node
            where node is the next node of label

        """
        for i in range(0, len(prefix_ids) + 1):
            src, label = prefix_ids[:i], prefix_ids[i:]
            node = self.labels.transit(label)
            if node:
                return src, label, node


class PairOfTagsProcessor(LogitsProcessor):
    def __init__(
            self,
            encoder_input_ids,
            left_labels: List[int],
            right_labels: List[int],
            eos,
            batch,
            num_beams: int = 1,
            tokenizer=None
    ):
        self.right_labels = right_labels
        self.left_labels = left_labels
        self.tokenizer: BartTokenizer = tokenizer
        self.eos = eos
        self.encoder_input_ids = encoder_input_ids
        self._num_beams = num_beams
        self.last_src_indices = [0] * (
                num_beams * len(encoder_input_ids))
        self.expected_right_labels = [None] * len(encoder_input_ids)
        self.batch = batch
        self.results = batch['_predictions'] = [[] for _ in range(0, len(encoder_input_ids))]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * self._num_beams + beam_id
                subtoken_to_token = self.batch['subtoken_to_token'][batch_id]
                tokens = self.batch['token'][batch_id]
                text_token_ids_ = self.batch.get('text_token_ids_', None)
                if text_token_ids_:
                    encoder_input_ids = text_token_ids_[batch_id]
                else:
                    encoder_input_ids: List[int] = self.encoder_input_ids[batch_id]
                # if len(encoder_input_ids) != len(subtoken_to_token):
                #     print()
                heads = [i == 0 or x != subtoken_to_token[i - 1] for i, x in enumerate(subtoken_to_token)]
                tails = [i + 1 > len(subtoken_to_token) - 1 or x != subtoken_to_token[i + 1] for i, x in
                         enumerate(subtoken_to_token)]
                encoder_input_ids = encoder_input_ids[:len(subtoken_to_token)]
                prefix_ids: list = input_ids[index].tolist()
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)
                # Check what was generated in the last step
                if prefix_ids[-1] in self.left_labels:
                    lid = self.left_labels.index(prefix_ids[-1])
                    self.expected_right_labels[index] = self.right_labels[lid]
                    self.results[index].append([subtoken_to_token[self.last_src_indices[index]]])
                elif prefix_ids[-1] == self.expected_right_labels[index]:
                    self.expected_right_labels[index] = None
                    self.results[index][-1].append(subtoken_to_token[self.last_src_indices[index]])
                    self.results[index][-1].append(self.tokenizer.convert_ids_to_tokens(prefix_ids[-1])[2:-1])
                elif prefix_ids[-1] in encoder_input_ids:
                    self.last_src_indices[index] += 1

                if self.last_src_indices[index] - 1 < len(tails) and tails[self.last_src_indices[index] - 1]:
                    if self.expected_right_labels[index] is not None:
                        allowed_tokens.add(self.expected_right_labels[index])
                    else:
                        for l in self.left_labels:
                            allowed_tokens.add(l)

                if self.last_src_indices[index] < len(encoder_input_ids):
                    if not (self.last_src_indices[index] == len(encoder_input_ids) - 1 and self.expected_right_labels[
                        index] is not None):
                        allowed_tokens.add(encoder_input_ids[self.last_src_indices[index]])

                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask


class TagCountProcessor(LogitsProcessor):
    def __init__(
            self,
            encoder_input_ids,
            labels: List[int],
            counts: List[int],
            eos,
            batch,
            num_beams: int = 1,
            tokenizer=None
    ):
        self.counts = counts
        self.labels = labels
        self.tokenizer: BartTokenizer = tokenizer
        self.eos = eos
        self.encoder_input_ids = encoder_input_ids
        self._num_beams = num_beams
        self.last_src_indices = [0] * (num_beams * len(encoder_input_ids))
        self.batch = batch
        self.results = batch['_predictions'] = [[] for _ in range(0, len(encoder_input_ids))]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * self._num_beams + beam_id
                subtoken_to_token = self.batch['subtoken_to_token'][batch_id]
                tokens = self.batch['token'][batch_id]
                text_token_ids_ = self.batch.get('text_token_ids_', None)
                if text_token_ids_:
                    encoder_input_ids = text_token_ids_[batch_id]
                else:
                    encoder_input_ids: List[int] = self.encoder_input_ids[batch_id]
                # if len(encoder_input_ids) != len(subtoken_to_token):
                #     print()
                heads = [i == 0 or x != subtoken_to_token[i - 1] for i, x in enumerate(subtoken_to_token)]
                tails = [i + 1 > len(subtoken_to_token) - 1 or x != subtoken_to_token[i + 1] for i, x in
                         enumerate(subtoken_to_token)]
                encoder_input_ids = encoder_input_ids[:len(subtoken_to_token)]
                prefix_ids: list = input_ids[index].tolist()
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)

                # Check what was generated in the last step
                could_generate_tokens = False
                could_generate_labels = False
                if prefix_ids[-1] in self.labels:
                    self.results[index].append([subtoken_to_token[self.last_src_indices[index]]])
                    for each in self.counts:
                        allowed_tokens.add(each)
                elif prefix_ids[-1] in self.counts:
                    l = self.counts.index(prefix_ids[-1]) + 1
                    self.results[index][-1].append(self.results[index][-1][0] + l)
                    self.results[index][-1].append(self.tokenizer.convert_ids_to_tokens(prefix_ids[-2])[1:-1])
                    while l:
                        if tails[self.last_src_indices[index]]:
                            l -= 1
                        if self.last_src_indices[index] == len(tails) - 1:
                            break
                        self.last_src_indices[index] += 1
                    could_generate_tokens = True
                    could_generate_labels = True
                elif prefix_ids[-1] in encoder_input_ids:
                    self.last_src_indices[index] += 1
                    could_generate_tokens = True
                    could_generate_labels = True

                if tails[self.last_src_indices[index] - 1] and could_generate_labels:
                    for l in self.labels:
                        allowed_tokens.add(l)

                if self.last_src_indices[index] < len(encoder_input_ids):
                    if could_generate_tokens:
                        if encoder_input_ids[self.last_src_indices[index]] == self.eos:
                            allowed_tokens.clear()
                        allowed_tokens.add(encoder_input_ids[self.last_src_indices[index]])

                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask


class CopyPrefixConstrainedOracleLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 encoder_input_ids,
                 labels: Trie,
                 separator,
                 eos,
                 pad,
                 sep,
                 delimiter,
                 batch,
                 label_ids,
                 full_labels: Trie,
                 num_beams: int = 1,
                 tokenizer=None):
        self.full_labels = full_labels
        self.label_ids = label_ids
        self.sep = sep
        self.delimiter = delimiter
        self.tokenizer: BartTokenizer = tokenizer
        self.pad = pad
        self.eos = eos
        self.separator = separator
        self.labels = labels
        self.encoder_input_ids = encoder_input_ids
        self._num_beams = num_beams
        self.last_src_indices = [1 if isinstance(tokenizer, BartTokenizerFast) else 0] * (
                num_beams * len(encoder_input_ids))
        self.last_sep_indices = [0] * (num_beams * len(encoder_input_ids))
        self.batch = batch
        self.next_label: List[int] = None
        self.next_label_ids: Dict[int, List[int]] = dict()

    # noinspection PyMethodOverriding
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * self._num_beams + beam_id
                subtoken_to_token = self.batch['subtoken_to_token'][batch_id]
                tokens = self.batch['token'][batch_id]
                text_token_ids_ = self.batch.get('text_token_ids_', None)
                if text_token_ids_:
                    encoder_input_ids = text_token_ids_[batch_id]
                else:
                    encoder_input_ids = self.encoder_input_ids[batch_id]
                # if len(encoder_input_ids) != len(subtoken_to_token):
                #     print()
                heads = [i == 0 or x != subtoken_to_token[i - 1] for i, x in enumerate(subtoken_to_token)]
                tails = [i + 1 > len(subtoken_to_token) - 1 or x != subtoken_to_token[i + 1] for i, x in
                         enumerate(subtoken_to_token)]
                encoder_input_ids = encoder_input_ids[:len(subtoken_to_token)]
                prefix_ids: list = input_ids[index].tolist()
                # Last token is separator, meaning a NE is generated so update states
                if prefix_ids and prefix_ids[-1] == self.separator:
                    last_prompt = prefix_ids[self.last_sep_indices[index] + 1:]
                    src, label, node = self.segment_src_label(last_prompt)
                    if node._value:
                        src_offset = first_index_of(encoder_input_ids, src, self.last_src_indices[index])
                        if src_offset is None:
                            cprint(f'[red]src_offset is None[/red], {tokens}')
                        else:
                            self.last_src_indices[index] = src_offset + len(src)
                        allowed_tokens.add(self.eos)  # We can choose to terminate
                    self.last_sep_indices[index] = len(prefix_ids) - 1

                prefix_ids = prefix_ids[self.last_sep_indices[index] + 1:]
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)

                if not prefix_ids:
                    # allow any src tokens except for bos and eos
                    allowed_tokens.update([i for i, h in zip(encoder_input_ids[self.last_src_indices[index]:-1],
                                                             heads[self.last_src_indices[index]:-1]) if h])
                    allowed_tokens.add(self.eos)
                elif batch_id in self.next_label_ids:
                    que = self.next_label_ids[batch_id]
                    allowed_tokens.add(que[0])
                    que = que[1:]
                    if not que:
                        del self.next_label_ids[batch_id]
                    else:
                        self.next_label_ids[batch_id] = que
                else:
                    maybe_tail = False
                    sure_label = True
                    for src_offset in indices_of(encoder_input_ids, prefix_ids, self.last_src_indices[index]):
                        sure_label = False
                        # Copy next token from src. len(encoder_input_ids) - 1 to exclude eos
                        if src_offset is not None:
                            if src_offset + len(prefix_ids) < len(encoder_input_ids) - 1:
                                allowed_tokens.add(encoder_input_ids[src_offset + len(prefix_ids)])
                                # if not maybe_tail and src_offset + len(prefix_ids) - 1 >= len(tails):
                                #     print(src_offset)
                                #     print(prefix_ids)
                                #     print(tails)
                                maybe_tail = maybe_tail or tails[src_offset + len(prefix_ids) - 1]
                            else:
                                maybe_tail = True
                    if prefix_ids != [self.delimiter]:
                        # Leading space is not considered as an entity
                        if maybe_tail or sure_label:
                            src, label, node = self.segment_src_label(prefix_ids)
                            allowed_tokens.update(list(node._children))
                            if not node._children:
                                allowed_tokens.clear()
                                que = self.label_ids[self.next_label[batch_id]]
                                allowed_tokens.add(que[0])
                                que = que[1:]
                                if not que:
                                    del self.next_label_ids[batch_id]
                                else:
                                    self.next_label_ids[batch_id] = que

                allowed_tokens.discard(self.pad)
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask

    def segment_src_label(self, prefix_ids):
        """
            Split prefix into src + label

        Args:
            prefix_ids:

        Returns:
            src, label, node
            where node is the next node of label

        """
        for i in range(0, len(prefix_ids) + 1):
            src, label = prefix_ids[:i], prefix_ids[i:]
            node = self.full_labels.transit(label)
            if node:
                return src, label, node


class DynamicCopyPrefixConstrainedOracleLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 encoder_input_ids,
                 labels: Trie,
                 separator,
                 eos,
                 pad,
                 sep,
                 delimiter,
                 batch,
                 label_ids,
                 full_labels: Trie,
                 num_beams: int = 1,
                 tokenizer=None):
        self.full_labels = full_labels
        self.label_ids = label_ids
        self.sep = sep
        self.delimiter = delimiter
        self.tokenizer: BartTokenizer = tokenizer
        self.pad = pad
        self.eos = eos
        self.separator = separator
        self.labels = labels
        self.encoder_input_ids = encoder_input_ids
        self._num_beams = num_beams
        self.last_src_indices = [1 if isinstance(tokenizer, BartTokenizerFast) else 0] * (
                num_beams * len(encoder_input_ids))
        self.last_sep_indices = [0] * (num_beams * len(encoder_input_ids))
        self.batch = batch
        self.next_label: List[int] = None
        self.next_label_ids: Dict[int, List[int]] = dict()
        self.be_word_id = list(self.labels._children)[0]
        self.num_label_tokens = 0  # single value doesn't matter since batch mode is not enabled

    # noinspection PyMethodOverriding
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Not for batching
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * self._num_beams + beam_id
                if self.num_label_tokens > 1:
                    self.num_label_tokens -= 1
                    mask = torch.full_like(scores, 0)
                    mask[index, [self.eos, self.separator]] = float('-inf')
                    continue
                else:
                    mask = torch.full_like(scores, -math.inf)
                subtoken_to_token = self.batch['subtoken_to_token'][batch_id]
                tokens = self.batch['token'][batch_id]
                text_token_ids_ = self.batch.get('text_token_ids_', None)
                if text_token_ids_:
                    encoder_input_ids = text_token_ids_[batch_id]
                else:
                    encoder_input_ids = self.encoder_input_ids[batch_id]
                # if len(encoder_input_ids) != len(subtoken_to_token):
                #     print()
                heads = [i == 0 or x != subtoken_to_token[i - 1] for i, x in enumerate(subtoken_to_token)]
                tails = [i + 1 > len(subtoken_to_token) - 1 or x != subtoken_to_token[i + 1] for i, x in
                         enumerate(subtoken_to_token)]
                encoder_input_ids = encoder_input_ids[:len(subtoken_to_token)]
                prefix_ids: list = input_ids[index].tolist()
                # Last token is separator, meaning a NE is generated so update states
                if prefix_ids and prefix_ids[-1] == self.separator:
                    last_prompt = prefix_ids[self.last_sep_indices[index] + 1:]
                    src, label, node = self.segment_src_label(last_prompt)
                    if node._value:
                        src_offset = first_index_of(encoder_input_ids, src, self.last_src_indices[index])
                        if src_offset is None:
                            cprint(f'[red]src_offset is None[/red], {tokens}')
                        else:
                            self.last_src_indices[index] = src_offset + len(src)
                        allowed_tokens.add(self.eos)  # We can choose to terminate
                    self.last_sep_indices[index] = len(prefix_ids) - 1

                prefix_ids = prefix_ids[self.last_sep_indices[index] + 1:]
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)

                if not prefix_ids:
                    # allow any src tokens except for bos and eos
                    allowed_tokens.update([i for i, h in zip(encoder_input_ids[self.last_src_indices[index]:-1],
                                                             heads[self.last_src_indices[index]:-1]) if h])
                    allowed_tokens.add(self.eos)
                elif self.num_label_tokens == 1:
                    self.num_label_tokens = 0
                    allowed_tokens.clear()
                    allowed_tokens.add(self.separator)
                else:
                    maybe_tail = False
                    sure_label = True
                    for src_offset in indices_of(encoder_input_ids, prefix_ids, self.last_src_indices[index]):
                        sure_label = False
                        # Copy next token from src. len(encoder_input_ids) - 1 to exclude eos
                        if src_offset is not None:
                            if src_offset + len(prefix_ids) < len(encoder_input_ids) - 1:
                                allowed_tokens.add(encoder_input_ids[src_offset + len(prefix_ids)])
                                # if not maybe_tail and src_offset + len(prefix_ids) - 1 >= len(tails):
                                #     print(src_offset)
                                #     print(prefix_ids)
                                #     print(tails)
                                maybe_tail = maybe_tail or tails[src_offset + len(prefix_ids) - 1]
                            else:
                                maybe_tail = True
                    if prefix_ids != [self.delimiter]:
                        # Leading space is not considered as an entity
                        if maybe_tail or sure_label:
                            src, label, node = self.segment_src_label(prefix_ids)
                            allowed_tokens.update(list(node._children))
                            if sure_label and prefix_ids[-1] == self.be_word_id:
                                self.num_label_tokens = 6

                allowed_tokens.discard(self.pad)
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask

    def segment_src_label(self, prefix_ids):
        """
            Split prefix into src + label

        Args:
            prefix_ids:

        Returns:
            src, label, node
            where node is the next node of label

        """
        for i in range(0, len(prefix_ids) + 1):
            src, label = prefix_ids[:i], prefix_ids[i:]
            node = self.full_labels.transit(label)
            if node:
                return src, label, node


class CopyLogitsProcessor(LogitsProcessor):
    def __init__(self, valid_input_ids) -> None:
        super().__init__()
        self.valid_input_ids = valid_input_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        good_mask_list = []
        for idx, batch_valid_tokens in enumerate(self.valid_input_ids):
            for token in batch_valid_tokens:
                # Eliminates invalid bad word IDs that are over the vocabulary size.
                if token <= scores.shape[1]:
                    good_mask_list.append([idx, token])
                else:
                    print(
                        f"An invalid word ID is defined: {token}. This ID is not contained in the"
                        f"vocabulary, and is therefore ignored."
                    )
        if not good_mask_list:
            return scores

        good_mask = torch.LongTensor(good_mask_list)
        indices = torch.ones(len(good_mask))
        # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
        # [ 0  1  1 ]
        # [ 0  0  0 ]
        # [ 1  0  0 ]

        good_mask = (
            torch.sparse.LongTensor(good_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
        )
        scores = scores.masked_fill(~good_mask, -float("inf"))
        return scores


class FirstTokenProcessor(LogitsProcessor):
    ISA = '<ISA>'
    POT = '<POT>'

    def __init__(self, isa_id, pot_id) -> None:
        super().__init__()
        self.pot_id = pot_id
        self.isa_id = isa_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) == 1:
            mask = torch.full_like(scores, -math.inf)
            mask[:, [self.pot_id, self.isa_id]] = 0
            return scores + mask
        return scores


class DynamicSwitchProcessor(LogitsProcessor):
    def __init__(self, first, isa, pot) -> None:
        super().__init__()
        self.pot = pot
        self.isa = isa
        self.first = first
        self.batch = pot.batch

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) == 1:
            return self.first(input_ids, scores)
        else:
            if input_ids.size(1) == 2:
                self.isa.batch = self.pick_batch_tensor(self.isa.batch, 'isa_')
                self.pot.batch = self.pick_batch_tensor(self.pot.batch, 'pot_')
                isa_mask = input_ids[:, 1] == self.first.isa_id
                self.batch['_isa_mask'] = isa_mask
            isa_mask = self.batch['_isa_mask']
            ids_without_first_token = torch.cat([input_ids[:, :1], input_ids[:, 2:]], dim=-1)
            isa_scores = self.isa(ids_without_first_token, scores)
            pot_scores = self.pot(ids_without_first_token, scores)
            pot_scores[isa_mask] = isa_scores[isa_mask]
            return pot_scores

    @staticmethod
    def pick_batch_tensor(batch, prefix, isa_mask=None):
        isa_batch = copy.copy(batch)
        for k, v in batch.items():
            if k.startswith(prefix):
                if isa_mask is not None and isinstance(v, torch.Tensor):
                    v = v[isa_mask]
                isa_batch[k[len(prefix):]] = v
        return isa_batch


class TagProcessor(LogitsProcessor):
    def __init__(self, batch, tags, tokenizer: BartTokenizer):
        self.tokenizer = tokenizer
        self.tags = dict((i, t) for i, t in zip(tokenizer.convert_tokens_to_ids(tags), tags))
        self.batch = batch
        self.eos = tokenizer.eos_token_id
        self.offsets = [-1] * len(batch['token'])
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
                    tags = [self.tags[x] for x in tag_ids if x in self.tags][:self.offsets[index]]
                    if len(tags) != len(tokens):
                        raise AssertionError('tags and tokens does not match')
                    entities = get_entities(tags)
                    batch['_predictions'][index] = [(x[1], x[2], x[0]) for x in entities]
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask


def main():
    haystack = 'the dog is chasing the cat on the beach'
    indices = indices_of(haystack, 'the', 3)
    for each in indices:
        print(haystack[each:each + 5])


if __name__ == '__main__':
    main()
