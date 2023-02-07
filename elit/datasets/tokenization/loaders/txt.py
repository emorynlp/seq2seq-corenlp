# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-01 12:35
from typing import Union, List, Callable

from elit.common.dataset import TransformableDataset
from elit.utils.io_util import TimingFileIterator
from elit.utils.span_util import words_to_bmes, words_to_bi
from elit.utils.string_util import split_long_sentence_into


class TextTokenizingDataset(TransformableDataset):
    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 generate_idx=None,
                 delimiter=None,
                 max_seq_len=None,
                 sent_delimiter=None,
                 char_level=False,
                 hard_constraint=False,
                 ) -> None:
        """A dataset for tagging tokenization tasks.

        Args:
            data: The local or remote path to a dataset, or a list of samples where each sample is a dict.
            transform: Predefined transform(s).
            cache: ``True`` to enable caching, so that transforms won't be called twice.
            generate_idx: Create a :const:`~hanlp_common.constants.IDX` field for each sample to store its order in dataset. Useful for prediction when
                samples are re-ordered by a sampler.
            delimiter: Delimiter between tokens used to split a line in the corpus.
            max_seq_len: Sentences longer than ``max_seq_len`` will be split into shorter ones if possible.
            sent_delimiter: Delimiter between sentences, like period or comma, which indicates a long sentence can
                be split here.
            char_level: Whether the sequence length is measured at char level.
            hard_constraint: Whether to enforce hard length constraint on sentences. If there is no ``sent_delimiter``
                in a sentence, it will be split at a token anyway.
        """
        self.hard_constraint = hard_constraint
        self.char_level = char_level
        self.sent_delimiter = sent_delimiter
        self.max_seq_len = max_seq_len
        self.delimiter = delimiter
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath: str):
        """Load tokenized corpus. The format is one sentence per line, where each line consisits of tokens seperated
        by a delimiter (usually space).

        .. highlight:: bash
        .. code-block:: bash

            $ head train.txt
            上海 浦东 开发 与 法制 建设 同步
            新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）

        Args:
            filepath: The path to the corpus.
        """
        f = TimingFileIterator(filepath)
        # longest_sent = 0
        for line in f:
            line = line.rstrip('\n')
            tokens = line.split(self.delimiter)
            if not tokens:
                continue
            if self.max_seq_len and sum(len(t) for t in tokens) > self.max_seq_len:
                # debug = []
                for short_sents in split_long_sentence_into(tokens, self.max_seq_len, self.sent_delimiter,
                                                            char_level=self.char_level,
                                                            hard_constraint=self.hard_constraint):
                    # debug.extend(short_sents)
                    # longest_sent = max(longest_sent, len(''.join(short_sents)))
                    yield {'token': short_sents}
                # assert debug == tokens
            else:
                # longest_sent = max(longest_sent, len(''.join(tokens)))
                yield {'token': tokens}
            f.log(line[:20])
        f.erase()
        # print(f'Longest sent: {longest_sent} in {filepath}')


def generate_tags_for_subtokens(sample: dict, tagging_scheme='BMES'):
    """
    Create a sequence of x for tokenization task. Each x is an atomic subtoken that will be tagged with BMES or BI tags.

    Args:
        sample: During prediction, it is a dict with 'token' being the input text, 'token_subtoken_offsets' being
         incremental offsets per each subtoken. During training, it is a dict with 'token' being a sequence of tokens,
         'token_subtoken_offsets' being non-incremental offsets per each subtoken, 'token_subtoken_offsets_group' being
         subtoken offsets grouped by each token.
        tagging_scheme:

    Returns:

    """
    # We could use token_token_span but we don't want token_token_span in the batch
    subtokens_group = sample.get('token_subtoken_offsets_group', None)
    sample['raw_token'] = sample['token']
    tokens = sample.get('token_') or sample['token']

    if subtokens_group:
        sample['token'] = subtokens_group_to_subtokens(tokens, subtokens_group)
        if tagging_scheme == 'BMES':
            sample['tag'] = words_to_bmes(subtokens_group)
        elif tagging_scheme == 'BI':
            sample['tag'] = words_to_bi(subtokens_group)
        else:
            raise NotImplementedError(f'Unsupported tagging scheme {tagging_scheme}.')
    else:
        sample['token'] = subtoken_offsets_to_subtokens(tokens, sample['token_subtoken_offsets'])
    return sample


def subtoken_offsets_to_subtokens(text, token_subtoken_offsets):
    results = []
    for b, e in token_subtoken_offsets:
        results.append(text[b:e])
    return results


def subtokens_group_to_subtokens(tokens, subtoken_offsets_group):
    results = []
    for subtoken_offsets, token in zip(subtoken_offsets_group, tokens):
        for b, e in subtoken_offsets:
            results.append(token[b:e])
    return results
