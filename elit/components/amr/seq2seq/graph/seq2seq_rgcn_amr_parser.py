# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-28 17:33
import logging
from typing import Union

import torch
from alnlp.modules.util import lengths_to_mask
from transformers import get_constant_schedule_with_warmup

from elit.common.dataset import PadSequenceDataLoader
from elit.common.transform import FieldLength
from elit.common.vocab import Vocab
from elit.components.amr.seq2seq.dataset.dataset import dfs_linearize_rgcn, AMRPickleDataset
from elit.components.amr.seq2seq.dataset.tokenization_bart import PENMANBartTokenizer
from elit.components.amr.seq2seq.graph.graph_bart import GraphBartForConditionalGeneration
from elit.components.amr.seq2seq.optim import RAdam
from elit.components.amr.seq2seq.seq2seq_amr_parser import Seq2seq_AMR_Parser
from elit.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp_common.util import merge_list_of_dict, merge_locals_kwargs


def collate_fn(samples, device):
    batch = merge_list_of_dict(samples)
    graphs = batch['dep_graph']
    batched_graphs = torch.zeros((len(graphs),) + max([x.shape for x in graphs]))
    for i, g in enumerate(graphs):
        n, n, r = g.shape
        batched_graphs[i, :n, :n, :r] = g
    batch['dep_graph'] = batched_graphs.to(
        device if isinstance(device, torch.device) or device > 0 else torch.device('cpu'))
    return batch


class Seq2seq_RGCN_AMR_Parser(Seq2seq_AMR_Parser):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tokenizer_transform = None
        self.model: GraphBartForConditionalGeneration = None

    def build_dataset(self, data, generate_idx):
        return AMRPickleDataset(data, generate_idx=generate_idx)

    def build_vocabs(self, trn: torch.utils.data.Dataset, logger: logging.Logger):
        super().build_vocabs(trn, logger)
        self.config.num_dep_rels = trn[0]['dep']['scores']['rel_scores'].size(-1)

    def finalize_dataset(self, dataset: AMRPickleDataset, logger: logging.Logger = None):
        dataset.append_transform(lambda x: dfs_linearize_rgcn(x, tokenizer=self._tokenizer))
        dataset.append_transform(self._tokenizer_transform)
        dataset.append_transform(FieldLength('text'))
        prune_max_seq_len = self.config.prune_max_seq_len
        if prune_max_seq_len:
            dataset.prune(lambda x: x['text_length'] > prune_max_seq_len, logger=logger)

    def _get_model_cls(self, transformer: str):
        self._transformer_config.num_dep_rels = self.config.num_dep_rels
        return GraphBartForConditionalGeneration

    def build_tokenizer(self, additional_tokens) -> PENMANBartTokenizer:
        tokenizer = super().build_tokenizer(additional_tokens)
        self._tokenizer_transform = TransformerSequenceTokenizer(tokenizer=tokenizer, input_key='text',
                                                                 cls_is_bos=True,
                                                                 output_key=['text_token_ids', 'text_token_span'])
        return tokenizer

    def feed_batch(self, batch):
        input_ids, labels = batch['text_token_ids'], batch.get('graph_token_ids')
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        token_mask = lengths_to_mask(batch['text_length'])
        dep_graph = batch['dep_graph']
        text_token_span = batch['text_token_span']
        if labels is not None:
            decoder_input_ids = labels[:, :-1]
            labels = labels[:, 1:].contiguous()
        else:
            decoder_input_ids = None
        return self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                          labels=labels, dep_graph=dep_graph, token_span=text_token_span, token_mask=token_mask)

    def _model_generate(self, batch, beam_size):
        input_ids = batch['text_token_ids']
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        token_mask = lengths_to_mask(batch['text_length'])
        dep_graph = batch['dep_graph']
        text_token_span = batch['text_token_span']
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1024,
            decoder_start_token_id=0,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            token_mask=token_mask,
            dep_graph=dep_graph,
            token_span=text_token_span
        )
        return out

    def _create_dataloader(self, dataset, batch_size, device, sampler, shuffle):
        return PadSequenceDataLoader(dataset, batch_size, shuffle, device=device, batch_sampler=sampler,
                                     pad=self._get_pad_dict(), collate_fn=lambda x: collate_fn(x, device=device))

    def build_optimizer(self, trn, lr, epochs, gradient_accumulation, warmup_steps, weight_decay, added_layer_lr=1e-3,
                        **kwargs):
        num_training_steps = len(trn) * epochs // gradient_accumulation
        if isinstance(warmup_steps, float):
            warmup_steps = int(num_training_steps * warmup_steps)
        para_all = set(self.model.parameters())
        para_added = set(self.model.model.encoder.dep_rgcn.parameters())
        grouped_para = [
            {"params": list(para_added), lr: added_layer_lr},
            {"params": list(para_all - para_added), lr: added_layer_lr}
        ]
        optimizer = RAdam(
            grouped_para,
            lr=lr,
            weight_decay=weight_decay)
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps)
        return optimizer, scheduler

    def fit(self, trn_data, dev_data, save_dir, batch_size=32, epochs=30, transformer='facebook/bart-base', lr=5e-05,
            added_layer_lr=1e-3, prune_max_seq_len=None,
            grad_norm=2.5, weight_decay=0.004, warmup_steps=1, dropout=0.25, attention_dropout=0.0, pred_min=5,
            eval_after=0.5, collapse_name_ops=False, use_pointer_tokens=True, raw_graph=False, gradient_accumulation=1,
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
