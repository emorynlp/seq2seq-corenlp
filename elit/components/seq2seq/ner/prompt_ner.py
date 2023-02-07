# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-10-22 11:01
import copy
from typing import List, Tuple

from elit.common.vocab import Vocab
from elit.datasets.ner.loaders.json_ner import JsonNERDataset
from elit.datasets.srl.ontonotes5.english import ONTONOTES5_NER_ENGLISH_DEV
from elit.metrics.chunking.sequence_labeling import get_entities
from elit.utils.log_util import logger
from hanlp_common.configurable import AutoConfigurable
from hanlp_trie.trie import Trie

SEP = '<sep>'
BEGIN = '<input>'
END = '</input>'


def to_exclusive_offset(sample: dict):
    ner = sample.get('', None)
    if ner:
        sample['ner'] = [(b, e + 1, l) for b, e, l in ner]
    return sample


def match_tokens(toks: List[str], needle: List[str], offset=0):
    for b in range(offset, len(toks) - len(needle) + 1):
        sub = toks[b:b + len(needle)]
        if sub == needle:
            return b, b + len(needle)
    return None


class Verbalizer(AutoConfigurable):
    def __init__(
            self,
            label_to_phrase,
            separator=';',
            delimiter=' ',
            be_word=' is ',
    ) -> None:
        super().__init__()
        self.delimiter = delimiter
        self.be_word = be_word
        self.separator = separator
        self.label_to_phrase = label_to_phrase
        self._phrase_to_label = dict((f'{v}{separator}', k) for k, v in label_to_phrase.items())
        self._phrase_trie = Trie(self._phrase_to_label)

    def to_prompt(self, ners: List[Tuple[int, int, str]], toks: List[str], inclusive_offset=False, spaces=None):
        sents = []
        for begin, end, label in ners:
            entity = ''.join(sum(list(zip(toks[begin:end + inclusive_offset], spaces[begin:end + inclusive_offset])),
                                 ())).strip() if spaces else self.delimiter.join(toks[begin:end + inclusive_offset])
            text = f'{self.delimiter}{entity}{self.be_word}{self.label_to_phrase[label]}'
            sents.append(text)
        return sents

    def to_prompt_tokens(self, ners: List[Tuple[int, int, str]], toks: List[str], inclusive_offset=False):
        sents = []
        for begin, end, label in ners:
            entity = toks[begin:end + inclusive_offset]
            text = [self.delimiter]
            text.extend(entity)
            text.append(self.be_word)
            text.append(self.label_to_phrase[label])
            text.append(self.separator)
            sents.append(text)
        return sents

    def prompt_to_entities(self, prompt: str, toks: List[str]):
        last_pe = 0
        last_ner_char_e = 0
        ners = []
        chars = []
        char_idx_to_token_idx = []
        for i, token in enumerate(toks):
            for c in token:
                chars.append(c)
                char_idx_to_token_idx.append(i)
            if self.delimiter:
                chars.append(self.delimiter)  # Add space to indicate word boundary
                char_idx_to_token_idx.append(i + 1)
        char_idx_to_token_idx.append(len(toks))
        chars = ''.join(chars)

        for pb, pe, label in self._phrase_trie.parse_longest(prompt):
            text_contains_ner = prompt[last_pe:pb]
            try:
                non_entity = text_contains_ner.rindex("isn't an entity; ")
                text_contains_ner = prompt[last_pe + non_entity + len("isn't an entity; "):pb]
            except ValueError:
                pass
            text_contains_ner = text_contains_ner[:-len(self.be_word)]
            if text_contains_ner.startswith(self.delimiter):
                text_contains_ner = text_contains_ner[len(self.delimiter):]
            # needle = text_contains_ner + self.delimiter
            needle = text_contains_ner
            if not self.delimiter:
                needle = needle.replace(' ', '')
            try:
                index = chars.index(needle, last_ner_char_e)
                last_ner_char_e = index + len(needle)
                span = (char_idx_to_token_idx[index], char_idx_to_token_idx[last_ner_char_e])
                b, e = span
                ners.append((b, e, label))
                last_pe = pe
            except ValueError:
                logger.warn(f'Cannot find "{needle}" in the text')

        return ners

    def sample_to_prompt(self, sample: dict):
        separator = self.separator
        sample['prompt'] = prompt = separator.join(self.to_prompt(sample['ner'], sample['token'])) + separator
        ner_back = self.prompt_to_entities(prompt, sample['token'])
        if not sample['ner'] == ner_back:
            print(sample)
            print(ner_back)
        return sample


class VerboseVerbalizer(Verbalizer):
    def to_prompt(self, ners: List[Tuple[int, int, str]], toks: List[str], inclusive_offset=False, spaces=None):
        sents = []
        offset = 0
        for begin, end, label in ners:
            if begin > offset:
                sents.append(f"{self.delimiter}{self.delimiter.join(toks[offset:begin])} isn't an entity")
            entity = ''.join(sum(list(zip(toks[begin:end + inclusive_offset], spaces[begin:end + inclusive_offset])),
                                 ())).strip() if spaces else self.delimiter.join(toks[begin:end + inclusive_offset])
            text = f'{self.delimiter}{entity}{self.be_word}{self.label_to_phrase[label]}'
            sents.append(text)
            offset = end
        if offset < len(toks):
            sents.append(f'{self.delimiter}{self.delimiter.join(toks[offset:len(toks)])} is not an entity')
        return sents


class PairOfTagsVerbalizer(AutoConfigurable):

    def __init__(self, labels) -> None:
        super().__init__()
        self.labels = labels

    def to_prompt_tokens(self, ners: List[Tuple[int, int, str]], toks: List[str], *args, **kwargs) -> List[str]:
        prompt = copy.copy(toks)
        offset = 0
        for b, e, l in ners:
            prompt.insert(b + offset, self.left_label(l))
            offset += 1
            prompt.insert(e + offset, self.right_label(l))
            offset += 1
        return prompt

    def left_label(self, label):
        return f'<{label}>'

    def right_label(self, label):
        return f'</{label}>'

    def sample_to_prompt(self, sample: dict):
        sample['prompt'] = ' '.join(self.to_prompt_tokens(sample['ner'], sample['token']))
        # ner_back = self.prompt_to_entities(prompt, sample['token'])
        # if not sample['ner'] == ner_back:
        #     print(sample)
        #     print(ner_back)
        return sample

    def get_additional_tokens(self):
        tokens = set()
        for each in self.labels:
            tokens.add(self.left_label(each))
            tokens.add(self.right_label(each))
        return sorted(tokens)

    def prompt_to_entities(self, prompt: str, toks: List[str]):
        return []


class TagCountVerbalizer(AutoConfigurable):
    def __init__(self, labels, **kwargs) -> None:
        super().__init__()
        self.labels = labels
        self.counts = [f'<count:{i}>' for i in range(1, 31)]

    def to_prompt_tokens(self, ners: List[Tuple[int, int, str]], toks: List[str], *args, **kwargs) -> List[str]:
        prompt = copy.copy(toks)
        offset = 0
        for b, e, l in ners:
            length = e - b
            del prompt[b + offset:e + offset]
            prompt.insert(b + offset, self.label_token(l))
            prompt.insert(b + offset + 1, self.counts[length - 1])
            offset += 2 - length
        return prompt

    def label_token(self, label):
        return f'<{label}>'

    def prompt_to_entities(self, prompt: str, toks: List[str]):
        return []

    def sample_to_prompt(self, sample: dict):
        sample['prompt'] = prompt = ' '.join(self.to_prompt_tokens(sample['ner'], sample['token']))
        ner_back = self.prompt_to_entities(prompt, sample['token'])
        # if not sample['ner'] == ner_back:
        #     print(sample)
        #     print(ner_back)
        return sample

    def get_additional_tokens(self):
        tokens = set()
        for each in self.labels:
            tokens.add(self.label_token(each))
        for each in self.counts:
            tokens.add(each)
        return sorted(tokens)


class TagVerbalizer(AutoConfigurable):
    def __init__(self, labels, training=True, **kwargs) -> None:
        super().__init__()
        self.labels = sum([[f'S-{x}', f'B-{x}', f'I-{x}', f'E-{x}'] for x in labels], ['O']) if training else labels
        self.training = False

    def to_prompt_tokens(self, ners: List[Tuple[int, int, str]], toks: List[str], *args, **kwargs) -> List[str]:
        tags = ['O'] * len(toks)
        for start, end, label in ners:
            if start + 1 == end:
                tags[start] = 'S-' + label
            else:
                tags[start] = 'B-' + label
                for i in range(start + 1, end):
                    tags[i] = 'I-' + label
                tags[end - 1] = 'E-' + label
        decoded = get_entities(tags)
        decoded = [(x[1], x[2], x[0]) for x in decoded]
        assert decoded == ners
        return tags

    def prompt_to_entities(self, prompt: str, toks: List[str]):
        return []

    def sample_to_prompt(self, sample: dict):
        sample['prompt'] = prompt = ' '.join(self.to_prompt_tokens(sample['ner'], sample['token']))
        # ner_back = self.prompt_to_entities(prompt, sample['token'])
        return sample

    def get_additional_tokens(self):
        tokens = set()
        tokens.update(self.labels)
        return sorted(tokens)


def main():
    dataset = JsonNERDataset(ONTONOTES5_NER_ENGLISH_DEV)
    dataset.append_transform(to_exclusive_offset)
    verb = Verbalizer(
        label_to_phrase={
            'PERSON': 'a person',
            'NORP': 'a nationality',
            'FAC': 'a facility',
            'ORG': 'an organization',
            'GPE': 'a geopolitical entity',
            'LOC': 'a location',
            'PRODUCT': 'a product',
            'EVENT': 'an event',
            'WORK_OF_ART': 'a work of art',
            'LAW': 'a law',
            'DATE': 'a date',
            'TIME': 'a time',
            'PERCENT': 'a percent',
            'MONEY': 'a monetary value or a unit',
            'QUANTITY': 'a quantity',
            'ORDINAL': 'an ordinal',
            'CARDINAL': 'a cardinal',
            'LANGUAGE': 'a language',
        },
    )
    dataset.append_transform(verb.sample_to_prompt)
    for each in dataset:
        pass


if __name__ == '__main__':
    main()
