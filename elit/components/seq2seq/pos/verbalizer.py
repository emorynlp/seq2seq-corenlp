# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-24 13:16
from abc import ABC, abstractmethod

from hanlp_common.configurable import AutoConfigurable
from transformers import BartTokenizer

from elit.components.seq2seq.ner.seq2seq_ner import tokenize


class Verbalizer(ABC, AutoConfigurable):
    def __call__(self, sample: dict):
        if 'tag' in sample:
            sample['prompt'] = self.to_prompt(sample['token'], sample['tag'])
        return sample

    def tokenize_prompt(self, prompt, tokenizer):
        return tokenize(prompt, tokenizer, '')[-1]

    @abstractmethod
    def to_prompt(self, tokens, tags):
        pass

    def recover_no_constraints(self, tokens, ids, tokenizer):
        raise NotImplementedError()


class TagVerbalizer(Verbalizer):
    def to_prompt(self, tokens, tags):
        return tags

    def recover_no_constraints(self, tokens, ids, tokenizer):
        tags = tokenizer.convert_ids_to_tokens(ids)
        if len(tags) < len(tokens):
            tags += [None] * (len(tokens) - len(tags))
        elif len(tags) > len(tokens):
            tags = tags[:len(tokens)]
        return tags


class TokenTagVerbalizer(Verbalizer):
    def to_prompt(self, tokens, tags):
        return list(sum(zip(tokens, tags), ()))

    def tokenize_prompt(self, prompt, tokenizer: BartTokenizer):
        ids = [tokenizer.bos_token_id]
        for i, token_or_tag in enumerate(prompt):
            if i % 2:
                ids.append(tokenizer.convert_tokens_to_ids(token_or_tag))
            else:
                ids.extend(tokenizer(' ' + token_or_tag, add_special_tokens=False).input_ids)
        ids.append(tokenizer.eos_token_id)
        return ids

    def recover_no_constraints(self, tokens, ids, tokenizer):
        generated_tokens = tokenizer.convert_ids_to_tokens(ids)
        print()


class IsAVerbalizer(Verbalizer):
    def __init__(self, tag_to_phrase: dict, quotation=False, is_a_tag=False) -> None:
        super().__init__()
        self.is_a_tag = is_a_tag
        self.quotation = quotation
        self.tag_to_phrase = tag_to_phrase

    def to_prompt(self, tokens, tags):
        phrases = []
        for token, tag in zip(tokens, tags):
            p = f'" {token} " is {self.tag_to_phrase[tag]};' if self.quotation else \
                f'{token} is {self.tag_to_phrase[tag]};'
            phrases.append(p)
        return ' '.join(phrases)

    def tokenize_prompt(self, prompt, tokenizer):
        return tokenizer(prompt).input_ids
