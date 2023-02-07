# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-01-05 18:46
import logging

from elit.common.vocab import Vocab
from transformers import BartTokenizer

from elit.components.seq2seq.ner.dynamic_oracle_bart import DynamicOracleBart
from elit.components.seq2seq.ner.prompt_ner import to_exclusive_offset, Verbalizer
from elit.components.seq2seq.ner.seq2seq_ner import Seq2seqNamedEntityRecognizer, tokenize
from hanlp_trie import Trie
from hanlp_trie.dictionary import TupleTrieDict


class DynamicOracleSeq2seqNamedEntityRecognizer(Seq2seqNamedEntityRecognizer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: DynamicOracleBart = None

    def _get_model_cls(self, transformer: str, oracle=False):
        return DynamicOracleBart

    def finalize_dataset(
            self,
            dataset,
            logger: logging.Logger = None,
            use_detokenization=False,
            src_lang: str = None,
            tgt_lang: str = None,
            max_seq_len=None,
            gazetteer_verbalizer=None,
            doc_context=0,
            oracle=False,
            **kwargs
    ):
        dataset.append_transform(to_exclusive_offset)
        dataset.append_transform(
            lambda x: create_and_tokenize_prompt(
                x,
                self._tokenizer, use_detokenization=use_detokenization,
                src_lang=src_lang, tgt_lang=tgt_lang,
                verbalizer=self.config.verbalizer,
                delimiter_id=self._transformer_config.delimiter,
                valid_label_token_trie=self._transformer_config.label_token_ids_trie,
                vocab=self.vocabs['label'],
                gazetteer=self.gazetteer,
                gazetteer_verbalizer=gazetteer_verbalizer,
                doc_context=doc_context,
            ),
        )

    def on_config_ready(self, gazetteer=None, doc_context=0, oracle=False, **kwargs):
        super().on_config_ready(gazetteer, doc_context, oracle, **kwargs)
        self._transformer_config.label_token_ids_trie = TupleTrieDict(dict(
            zip([tuple(x) for x in self._transformer_config.valid_label_token_ids],
                self.config.verbalizer.label_to_phrase)))
        self._transformer_config.tokenizer = self._tokenizer

    def feed_batch(self, batch, oracle=False, **kwargs):
        return self.model.fit(batch)


def create_and_tokenize_prompt(
        sample,
        tokenizer: BartTokenizer,
        verbalizer: Verbalizer,
        delimiter_id: int,
        valid_label_token_trie: Trie,
        vocab: Vocab,
        use_detokenization=False,
        src_lang: str = None,
        tgt_lang: str = None,
        gazetteer: TupleTrieDict = None,
        gazetteer_verbalizer: Verbalizer = None,
        doc_context=0,
):
    spaces = sample.get('spaces', None)
    ner = sample.get('ner', None)
    tokens = sample['token']
    delimiter = verbalizer.delimiter
    if ner is None:
        prompt = None
    else:
        if delimiter:
            prompt = verbalizer.separator.join(
                verbalizer.to_prompt(ner, tokens, spaces=spaces if use_detokenization else None))
            if ner:
                prompt += verbalizer.separator
            sample['prompt'] = prompt
        else:
            _p_tokens = verbalizer.to_prompt_tokens(ner, tokens)
            prompt = [x for x in sum(_p_tokens, []) if x]  # remove delimiter
            sample['prompt'] = ''.join(prompt)

    if prompt is not None:
        if delimiter:
            sample['prompt_token_ids'] = tokenizer(prompt).input_ids
        else:
            sample['prompt_token_ids'] = tokenize(prompt, tokenizer, delimiter)[-1]

    normalized_tokens, subtoken_to_token, text_token_ids = tokenize(tokens, tokenizer, delimiter)
    if src_lang:
        text_token_ids.append(tokenizer.lang_code_to_id[src_lang])

    subtoken_to_token.append(len(normalized_tokens))

    sample['text_token_ids'] = text_token_ids
    sample['subtoken_to_token'] = subtoken_to_token
    sample['text'] = tokenizer.decode(text_token_ids[:-1], clean_up_tokenization_spaces=False)
    sample['normalized_tokens'] = normalized_tokens
    prompt_token_ids = sample['prompt_token_ids']

    x_non_gen = []
    x_gen = []  # for gold label if it's not generated
    y_non_gen = []  # for teacher forcing
    # y_gen = []  # not used at all
    offset = 0
    label_ids = []
    for b, e, l in valid_label_token_trie.parse_longest(prompt_token_ids):
        b += 2  # is a
        e -= 1  # ;
        if b > offset:
            x_non_gen.append(prompt_token_ids[offset:b])
            y_non_gen.append(prompt_token_ids[1:][offset:b - 1])
        x_gen.append(prompt_token_ids[b:e])
        # y_gen.append(prompt_token_ids[1:][b:e])
        offset = e
        label_ids.append(vocab[l])

    x_non_gen.append(prompt_token_ids[offset:offset + 1])
    y_non_gen.append(prompt_token_ids[offset + 1:offset + 2])

    # if isdebugging():
    #     print('x_non_gen:')
    #     for each in x_non_gen:
    #         print(tokenizer.convert_ids_to_tokens(each))
    #
    #     print('y_non_gen:')
    #     for each in y_non_gen:
    #         print(tokenizer.convert_ids_to_tokens(each))
    #
    #     print('x_gen:')
    #     for each in x_gen:
    #         print(tokenizer.convert_ids_to_tokens(each))

    # assert sum([x[0] + x[1] for x in zip(x_non_gen, x_gen)], []) + x_non_gen[-1] == prompt_token_ids[:-1]
    sample['x_non_gen'] = x_non_gen
    sample['x_gen'] = x_gen
    sample['y_non_gen'] = y_non_gen
    sample['label'] = label_ids
    # sample['y_gen_ids'] = y_gen
    return sample


def split_gen(valid_label_token_trie, prompt_token_ids):
    prompt_non_gen = []
    prompt_gen = []
    offset = 0
    for b, e, l in valid_label_token_trie.parse_longest(prompt_token_ids):
        b += 2  # is a
        if b > offset:
            prompt_non_gen.append(prompt_token_ids[offset:b])
        prompt_gen.append(prompt_token_ids[b:e])
        offset = e
    if offset < len(prompt_token_ids):
        prompt_non_gen.append(prompt_token_ids[offset:])
    return prompt_gen, prompt_non_gen
