# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-09-14 14:50
import logging
from typing import Union

import torch.utils.checkpoint

from elit.common.vocab import Vocab
from elit.components.amr.seq2seq.dataset.dataset import dfs_linearize_tokenize_with_linguistic_structures
from elit.components.amr.seq2seq.linguistic_bart import LinguisticBartForConditionalGeneration
from elit.components.amr.seq2seq.seq2seq_amr_parser import Seq2seq_AMR_Parser
from hanlp_common.util import merge_locals_kwargs


class Seq2seq_Embedding_AMR_Parser(Seq2seq_AMR_Parser):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: LinguisticBartForConditionalGeneration = None

    def build_vocabs(self, trn: torch.utils.data.Dataset, logger: logging.Logger):
        super().build_vocabs(trn, logger)
        if self.config.pos_layer is not None:
            self.vocabs['pos'] = Vocab(unk_token=None)
        if self.config.ner_layer is not None:
            self.vocabs['ner'] = Vocab(unk_token=None)
        if self.config.dep_layer is not None:
            self.vocabs['dep_rel'] = Vocab(unk_token=None)

    def finalize_dataset(self, dataset):
        dataset.append_transform(
            lambda x: dfs_linearize_tokenize_with_linguistic_structures(x, tokenizer=self._tokenizer))
        dataset.append_transform(self.vocabs)

    def _get_model_cls(self, transformer: str):
        self._transformer_config.pos_layer = self.config.pos_layer
        self._transformer_config.pos_vocab_size = len(self.vocabs['pos']) if self.config.pos_layer is not None else None
        self._transformer_config.ner_layer = self.config.ner_layer
        self._transformer_config.ner_vocab_size = len(self.vocabs['ner']) if self.config.ner_layer is not None else None
        self._transformer_config.dep_layer = self.config.dep_layer
        self._transformer_config.dep_rel_vocab_size = len(
            self.vocabs['dep_rel']) if self.config.dep_layer is not None else None
        return LinguisticBartForConditionalGeneration

    def fit(self, trn_data, dev_data, save_dir, batch_size=32, epochs=30, transformer='facebook/bart-base', lr=5e-05,
            grad_norm=2.5, weight_decay=0.004, warmup_steps=1, dropout=0.25, attention_dropout=0.0, pred_min=5,
            eval_after=0.5, collapse_name_ops=False, use_pointer_tokens=True, raw_graph=False, gradient_accumulation=1,
            pos_layer=None, dep_layer=None, ner_layer=None,
            recategorization_tokens=(
                    'PERSON', 'COUNTRY', 'QUANTITY', 'ORGANIZATION', 'DATE_ATTRS', 'NATIONALITY', 'LOCATION', 'ENTITY',
                    'CITY',
                    'MISC', 'ORDINAL_ENTITY', 'IDEOLOGY', 'RELIGION', 'STATE_OR_PROVINCE', 'URL', 'CAUSE_OF_DEATH', 'O',
                    'TITLE', 'DATE', 'NUMBER', 'HANDLE', 'SCORE_ENTITY', 'DURATION', 'ORDINAL', 'MONEY', 'SET',
                    'CRIMINAL_CHARGE', '_1', '_2', '_3', '_4', '_2', '_5', '_6', '_7', '_8', '_9', '_10', '_11', '_12',
                    '_13',
                    '_14', '_15'), additional_tokens=(
                    'date-entity', 'government-organization', 'temporal-quantity', 'amr-unknown', 'multi-sentence',
                    'political-party', 'monetary-quantity', 'ordinal-entity', 'religious-group', 'percentage-entity',
                    'world-region', 'url-entity', 'political-movement', 'et-cetera', 'at-least', 'mass-quantity',
                    'have-org-role-91', 'have-rel-role-91', 'include-91', 'have-concession-91', 'have-condition-91',
                    'be-located-at-91', 'rate-entity-91', 'instead-of-91', 'hyperlink-91', 'request-confirmation-91',
                    'have-purpose-91', 'be-temporally-at-91', 'regardless-91', 'have-polarity-91', 'byline-91',
                    'have-manner-91', 'have-part-91', 'have-quant-91', 'publication-91', 'be-from-91', 'have-mod-91',
                    'have-frequency-91', 'score-on-scale-91', 'have-li-91', 'be-compared-to-91', 'be-destined-for-91',
                    'course-91', 'have-subevent-91', 'street-address-91', 'have-extent-91', 'statistical-test-91',
                    'have-instrument-91', 'have-name-91', 'be-polite-91', '-00', '-01', '-02', '-03', '-04', '-05',
                    '-06',
                    '-07', '-08', '-09', '-10', '-11', '-12', '-13', '-14', '-15', '-16', '-17', '-18', '-19', '-20',
                    '-21',
                    '-22', '-23', '-24', '-25', '-26', '-27', '-28', '-29', '-20', '-31', '-32', '-33', '-34', '-35',
                    '-36',
                    '-37', '-38', '-39', '-40', '-41', '-42', '-43', '-44', '-45', '-46', '-47', '-48', '-49', '-50',
                    '-51',
                    '-52', '-53', '-54', '-55', '-56', '-57', '-58', '-59', '-60', '-61', '-62', '-63', '-64', '-65',
                    '-66',
                    '-67', '-68', '-69', '-70', '-71', '-72', '-73', '-74', '-75', '-76', '-77', '-78', '-79', '-80',
                    '-81',
                    '-82', '-83', '-84', '-85', '-86', '-87', '-88', '-89', '-90', '-91', '-92', '-93', '-94', '-95',
                    '-96',
                    '-97', '-98', '-of'), devices=None, logger=None, seed=None, finetune: Union[bool, str] = False,
            eval_trn=True, _device_placeholder=False, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def feed_batch(self, batch):
        input_ids, labels = batch['text_token_ids'], batch.get('graph_token_ids')
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        if labels is not None:
            decoder_input_ids = labels[:, :-1]
            labels = labels[:, 1:].contiguous()
        else:
            decoder_input_ids = None
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels,
            pos_ids=batch.get('pos_id', None),
            ner_ids=batch.get('ner_id', None),
            dep_arc=batch.get('dep_arc', None),
            dep_rel=batch.get('dep_rel_id', None),
        )

    def _model_generate(self, batch, beam_size):
        input_ids = batch['text_token_ids']
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1024,
            decoder_start_token_id=0,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            pos_ids=batch.get('pos_id', None),
            ner_ids=batch.get('ner_id', None),
            dep_arc=batch.get('dep_arc', None),
            dep_rel=batch.get('dep_rel_id', None),
        )
        return out

    def _get_pad_dict(self):
        pad_dict = super()._get_pad_dict()
        pad_dict['dep_arc'] = 0
        return pad_dict
