# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-11-10 16:05
import copy
import math
from enum import Enum

import torch

from elit.components.seq2seq.dep.arc_standard import State, DependencyTree
from elit.components.seq2seq.dep.verbalizer import make_token_on_stack
from hanlp_trie.dictionary import TupleTrieDict
from transformers import BartTokenizer
from transformers.generation_logits_process import LogitsProcessor

from elit.utils.log_util import cprint


class HeadRelationProcessor(LogitsProcessor):
    def __init__(self, batch, heads, relations, tokenizer: BartTokenizer):
        self.relations = relations
        self.rel_id_to_name = dict(zip(relations, [x[5:-1] for x in tokenizer.convert_ids_to_tokens(relations)]))
        self.heads = heads
        self.head_id_to_digit = dict(zip(heads, [int(x[5:-1]) for x in tokenizer.convert_ids_to_tokens(heads)]))
        self.tokenizer = tokenizer
        self.batch = batch
        self.eos = tokenizer.eos_token_id
        self.bos = tokenizer.bos_token_id
        self.offsets = [0] * len(batch['FORM'])
        self.batch['_predictions'] = [[] for _ in batch['FORM']]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        batch = self.batch
        for batch_id, beam_sent in enumerate(input_ids.view(-1, 1, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * 1 + beam_id
                prefix_ids: list = input_ids[index][1:].tolist()
                if self.eos in prefix_ids:
                    prefix_ids = prefix_ids[:prefix_ids.index(self.eos)]
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)
                tokens = self.batch['FORM'][index]
                if self.offsets[index] < len(tokens):
                    if len(prefix_ids) % 2:
                        allowed_tokens.update(self.relations)
                        self.offsets[index] += 1
                    else:
                        allowed_tokens.update(self.heads[:len(tokens) + 1])
                else:
                    allowed_tokens.add(self.eos)
                    prediction = self.batch['_predictions'][index]
                    if not prediction:
                        for i in range(0, len(prefix_ids), 2):
                            if prefix_ids[i] not in self.head_id_to_digit or prefix_ids[
                                i + 1] not in self.rel_id_to_name:
                                break
                            prediction.append(
                                (self.head_id_to_digit[prefix_ids[i]], self.rel_id_to_name[prefix_ids[i + 1]]))
                        assert len(prediction) == len(tokens)
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask


class TransitionProcessor(LogitsProcessor):
    def __init__(self, batch, actions, tokenizer: BartTokenizer):
        self.actions = actions
        self.id_to_action = dict(zip(actions, tokenizer.convert_ids_to_tokens(actions)))
        self.tokenizer = tokenizer
        self.batch = batch
        self.eos = tokenizer.eos_token_id
        self.bos = tokenizer.bos_token_id
        # self.offsets = [0] * len(batch['FORM'])
        # self.batch['_predictions'] = [[] for _ in batch['FORM']]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        # batch = self.batch
        for batch_id, beam_sent in enumerate(input_ids.view(-1, 1, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * 1 + beam_id
                prefix_ids: list = input_ids[index][1:].tolist()
                if self.eos in prefix_ids:
                    prefix_ids = prefix_ids[:prefix_ids.index(self.eos)]
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)
                if prefix_ids and prefix_ids[-1] == self.eos:
                    pass
                else:
                    allowed_tokens.update(self.actions)
                allowed_tokens.add(self.eos)
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask


class LexicalTransitionProcessor(LogitsProcessor):
    def __init__(self, batch, actions, tokenizer: BartTokenizer):
        self.actions = actions
        self.id_to_action = dict(zip(actions, tokenizer.convert_ids_to_tokens(actions)))
        self.tokenizer = tokenizer
        self.batch = batch
        self.eos = tokenizer.eos_token_id
        self.bos = tokenizer.bos_token_id
        self.subtoken_offsets = [1] * len(batch['FORM'])
        # self.batch['_predictions'] = [[] for _ in batch['FORM']]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        # batch = self.batch
        for batch_id, beam_sent in enumerate(input_ids.view(-1, 1, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                index = batch_id * 1 + beam_id
                allowed_tokens = set()
                prefix_ids: list = input_ids[index][1:].tolist()
                subtoken_to_token = self.batch['subtoken_to_token'][index]
                tokens = self.batch['FORM'][index]
                # ['A', 'record', 'date', 'LA-nn', 'LA-det', 'has', "n't", 'been', 'set', 'LA-auxpass', 'LA-neg',
                # 'LA-aux', 'LA-nsubjpass', '.', 'RA-punct', 'RA-root']
                text_token_ids = self.batch['text_token_ids']
                encoder_input_ids = text_token_ids[index].tolist()
                tails = [i + 1 > len(subtoken_to_token) - 1 or x != subtoken_to_token[i + 1] for i, x in
                         enumerate(subtoken_to_token)]
                if self.eos in prefix_ids:
                    prefix_ids = prefix_ids[:prefix_ids.index(self.eos)]
                else:
                    if prefix_ids and prefix_ids[-1] not in self.actions:
                        self.subtoken_offsets[index] += 1
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)
                if prefix_ids and prefix_ids[-1] == self.eos:
                    pass
                else:
                    allowed_tokens.add(encoder_input_ids[self.subtoken_offsets[index]])
                    if not prefix_ids:
                        pass
                    else:
                        if prefix_ids[-1] in self.actions:
                            allowed_tokens.update(self.actions)
                        else:
                            if tails[self.subtoken_offsets[index] - 1]:
                                allowed_tokens.update(self.actions)
                if self.subtoken_offsets[index] == len(encoder_input_ids) - 1:
                    allowed_tokens.add(self.eos)
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask


class BracketProcessor(LogitsProcessor):
    def __init__(self, batch, relations, lb, rb, tokenizer: BartTokenizer):
        self.rb = rb
        self.lb = lb
        self.relations = relations
        self.tokenizer = tokenizer
        self.batch = batch
        self.eos = tokenizer.eos_token_id
        self.bos = tokenizer.bos_token_id
        self.batch['_predictions'] = [[] for _ in batch['FORM']]
        self.last_copied_buffer = [[] for _ in batch['FORM']]
        self.unclosed_brackets = [0 for _ in batch['FORM']]
        self.subtoken_tries = []
        for form, sub2token, sub_ids in zip(batch['FORM'], batch['subtoken_to_token'],
                                            batch['text_token_ids'].tolist()):
            prev_offset = None
            buffer = []
            d = dict()
            for offset, sub_id in zip(sub2token[1:], sub_ids[1:]):
                if offset != prev_offset and buffer:
                    d[tuple(buffer)] = prev_offset
                    buffer = []
                buffer.append(sub_id)
                prev_offset = offset
            self.subtoken_tries.append(TupleTrieDict(d))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        debug = False
        for batch_id, beam_sent in enumerate(input_ids.view(-1, 1, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * 1 + beam_id
                prefix_ids: list = input_ids[index][1:].tolist()
                if self.eos in prefix_ids:
                    prefix_ids = prefix_ids[:prefix_ids.index(self.eos)]
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)
                prev_id = prefix_ids[-1] if prefix_ids else ''
                trie = self.subtoken_tries[index]
                if prev_id == self.lb:
                    self.unclosed_brackets[index] += 1
                    # expect a token
                    allowed_tokens.update(trie._children)
                elif prev_id == self.rb:
                    self.unclosed_brackets[index] -= 1
                    if self.unclosed_brackets[index]:
                        allowed_tokens.add(self.rb)
                        if len(trie):
                            allowed_tokens.update(self.relations)
                    else:
                        # assert len(trie) == 0
                        if len(trie) and False:
                            msg = f'Constrained encoding failed with sample:\n{self.batch["FORM"][index]}'
                            raise RuntimeError(msg)
                        allowed_tokens.add(self.eos)
                elif prev_id == '':
                    allowed_tokens.add(self.lb)
                elif prev_id in self.relations:
                    allowed_tokens.add(self.lb)
                else:
                    # among tokens
                    buffer = self.last_copied_buffer[index]
                    buffer.append(prev_id)
                    node = trie.transit(self.last_copied_buffer[index])
                    if node is None:
                        raise RuntimeError(f'Constrained encoding failed with sample:\n{self.batch["FORM"][index]}')
                    if node._value is not None:
                        # end of a token, expect a relation or a RB
                        del trie[buffer]
                        allowed_tokens.add(self.rb)
                        allowed_tokens.update(self.relations)
                        buffer.clear()
                    else:
                        allowed_tokens.update(node._children)
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                if debug:
                    cprint(f'[green]{len(prefix_ids)}[/green] {self.tokenizer.decode(prefix_ids)}[yellow] '
                           f'{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask


class PromptStatus(Enum):
    first_token = 0
    phrase = 1
    second_token = 2
    semicolon_maybe = 3
    semicolon = 4
    done = 5


class PromptProcessor(LogitsProcessor):
    def __init__(self, batch, tokenizer: BartTokenizer, trie: TupleTrieDict):
        self.trie = trie
        self.tokenizer = tokenizer
        self.batch = batch
        self.eos = tokenizer.eos_token_id
        self.bos = tokenizer.bos_token_id
        self.subtoken_offsets = [1] * len(batch['FORM'])
        self.token_offsets = [[] for _ in range(len(batch['FORM']))]
        self.candidate_tokens = [list(range(len(tokens) + 1)) for tokens in batch['FORM']]
        self.last_token_end = [0] * len(batch['FORM'])
        self.last_first_token = [None] * len(batch['FORM'])
        self.last_second_token = [None] * len(batch['FORM'])
        self.last_transition = [None] * len(batch['FORM'])
        self.last_token_start = [0] * len(batch['FORM'])
        self.statues = [PromptStatus.first_token] * len(batch['FORM'])
        self.semicolon = self.tokenizer.convert_tokens_to_ids(';')
        batch['arc_standard'] = [State(len(tokens)) for tokens in batch['FORM']]
        self.transitions = [[] for _ in batch['FORM']]
        self.token_to_subtoken = [[[] for _ in range(len(tokens))] for tokens in batch['FORM']]
        self.root_subtokens = self.tokenizer(' sentence', add_special_tokens=False).input_ids
        for subtoken_to_token, tokens, token_to_subtoken in zip(batch['subtoken_to_token'], batch['FORM'],
                                                                self.token_to_subtoken):
            for i, t in enumerate(subtoken_to_token):
                if 0 <= t < len(tokens):
                    token_to_subtoken[t].append(i)

        batch['arc_standard_failed'] = [False] * len(batch['FORM'])
        batch['string_matching_results'] = [DependencyTree(length=len(tokens)) for tokens in batch['FORM']]
        batch['_predictions'] = [None] * len(batch['FORM'])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        # batch = self.batch
        for batch_id, beam_sent in enumerate(input_ids.view(-1, 1, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * 1 + beam_id
                text_token_ids = self.batch['text_token_ids']
                subtoken_to_token = self.batch['subtoken_to_token'][index]
                tokens = self.batch['FORM'][index]
                full_tokens = ['sentence'] + tokens  # with root being "sentence"
                candidate_tokens = self.candidate_tokens[index]
                encoder_input_ids = text_token_ids[index].tolist()
                tails = [i + 1 > len(subtoken_to_token) - 1 or x != subtoken_to_token[i + 1] for i, x in
                         enumerate(subtoken_to_token)]
                token_to_subtoken = self.token_to_subtoken[index]
                token_offsets = self.token_offsets[index]
                arc_standard_failed = self.batch['arc_standard_failed']
                system = self.batch['arc_standard'][index]
                prefix_ids: list = input_ids[index][1:].tolist()
                if self.eos in prefix_ids:
                    prefix_ids = prefix_ids[:prefix_ids.index(self.eos)]
                # prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)
                prefix_str = self.tokenizer.decode(prefix_ids)
                # print(f'{len(prefix_ids)} {prefix_str}', end='')
                if self.statues[index] == PromptStatus.semicolon_maybe:
                    if prefix_ids[-1] == self.semicolon:
                        system = self.decode_phrase(system, arc_standard_failed, full_tokens, index)
                        self.last_token_start[index] = len(prefix_ids)
                    else:
                        self.statues[index] = PromptStatus.second_token
                if self.statues[index] == PromptStatus.first_token:
                    partial_token_offsets, matched_token_offsets = self.add_candidate_subtoken(
                        allowed_tokens, candidate_tokens, encoder_input_ids, index, prefix_ids, token_to_subtoken,
                        PromptStatus.phrase, self.last_first_token)
                    if self.statues[index] != PromptStatus.phrase:
                        if matched_token_offsets:
                            allowed_tokens.update(self.trie._children)
                            self.last_first_token[index] = matched_token_offsets
                if self.statues[index] == PromptStatus.phrase:
                    phrase_prefix = prefix_ids[self.last_token_end[index]:]
                    node = self.trie.transit(phrase_prefix)
                    children = node._children
                    if children:
                        allowed_tokens.update(children)
                    else:
                        transition = node._value
                        self.last_transition[index] = transition
                        self.statues[index] = PromptStatus.second_token
                        self.last_token_start[index] = len(prefix_ids)
                if self.statues[index] == PromptStatus.second_token:
                    partial_token_offsets, matched_token_offsets = self.add_candidate_subtoken(
                        allowed_tokens, candidate_tokens, encoder_input_ids, index, prefix_ids, token_to_subtoken,
                        PromptStatus.semicolon, self.last_second_token)
                    if self.statues[index] != PromptStatus.semicolon:
                        if matched_token_offsets:
                            allowed_tokens.add(self.semicolon)
                            self.statues[index] = PromptStatus.semicolon_maybe

                if self.statues[index] == PromptStatus.semicolon:
                    allowed_tokens.add(self.semicolon)
                    system = self.decode_phrase(system, arc_standard_failed, full_tokens, index)
                    self.last_token_start[index] = len(prefix_ids) + 1
                if system.is_terminal() or (arc_standard_failed[index] and len(self.candidate_tokens[index]) == 1):
                    allowed_tokens.add(self.eos)
                    self.statues[index] = PromptStatus.done
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f' [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask

    def decode_phrase(self, system, arc_standard_failed, full_tokens, index):
        transition: str = self.last_transition[index]
        first, second = self.last_second_token[index][0], self.last_first_token[index][0]
        first, second = full_tokens[first], full_tokens[second]
        pred_string_matching: DependencyTree = self.batch['string_matching_results'][index]
        if not arc_standard_failed[index] and not system.is_terminal():
            backup_state = copy.deepcopy(system)
            try:
                make_token_on_stack(0, first, full_tokens, system)
                make_token_on_stack(1, second, full_tokens, system)
                dependent, relation, head = system.apply(transition)
                self.candidate_tokens[index].remove(dependent)

                pred_string_matching.heads[dependent] = head
                pred_string_matching.labels[dependent] = relation
                if pred_string_matching.heads[dependent] != system.pred.heads[dependent]:
                    print('Failed to copy predictions from arc standard to string matching')
                    exit(1)
            except Exception as e:
                # print(f'Arc Standard failed due to {e}')
                arc_standard_failed[index] = True
                self.batch['arc_standard'][index] = system = backup_state
        if arc_standard_failed[index]:
            matched = False
            before_candidate_count = len(self.candidate_tokens[index])
            for f in self.last_first_token[index]:
                for s in self.last_second_token[index]:
                    if transition.startswith('LA'):
                        if not pred_string_matching.labels[f]:
                            pred_string_matching.heads[f] = s
                            pred_string_matching.labels[f] = transition.split('-', 1)[1]
                            if f in self.candidate_tokens[index]:
                                self.candidate_tokens[index].remove(f)
                            matched = True
                            break
                    else:
                        if not pred_string_matching.labels[s]:
                            pred_string_matching.heads[s] = f
                            pred_string_matching.labels[s] = transition.split('-', 1)[1]
                            if s in self.candidate_tokens[index]:
                                self.candidate_tokens[index].remove(s)
                            matched = True
                            break
            if not matched:
                if transition.startswith('LA'):
                    f = self.last_first_token[index][0]
                    if f in self.candidate_tokens:
                        self.candidate_tokens.remove(f)
                else:
                    s = self.last_second_token[index][0]
                    if s in self.candidate_tokens:
                        self.candidate_tokens.remove(s)
            if len(self.candidate_tokens[index]) == before_candidate_count:
                print('Candidates never get removed, decoding will never end')
        self.statues[index] = PromptStatus.first_token
        return system

    def add_candidate_subtoken(self, allowed_tokens, candidate_tokens, encoder_input_ids, index, prefix_ids,
                               token_to_subtoken, next_status, last_token):
        if prefix_ids:
            prefix_token = prefix_ids[self.last_token_start[index]:]
        else:
            prefix_token = []
        if candidate_tokens:
            matched_token_offsets = []
            partial_token_offsets = []
            for candidate_token in candidate_tokens:
                if candidate_token == 0:
                    candidate_token_subtokens = self.root_subtokens
                else:
                    candidate_token_subtokens = [encoder_input_ids[s] for s in
                                                 token_to_subtoken[candidate_token - 1]]
                if candidate_token_subtokens[:len(prefix_token)] == prefix_token:
                    rest = candidate_token_subtokens[len(prefix_token):]
                    if rest:
                        allowed_tokens.add(rest[0])
                        partial_token_offsets.append(candidate_token)
                    else:
                        matched_token_offsets.append(candidate_token)
            if not partial_token_offsets:
                self.statues[index] = next_status
                self.last_token_end[index] = len(prefix_ids) if matched_token_offsets else len(prefix_ids) - 1
            if matched_token_offsets:
                last_token[index] = matched_token_offsets
            return partial_token_offsets, matched_token_offsets
        return None, None
