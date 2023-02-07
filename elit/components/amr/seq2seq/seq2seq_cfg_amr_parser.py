# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-28 17:33
import json

import torch
from phrasetree.tree import Tree
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, BartLearnedPositionalEmbedding

from elit.common.vocab import Vocab
from elit.components.amr.seq2seq.dataset.dataset import dfs_linearize_constituency
from elit.components.amr.seq2seq.seq2seq_amr_parser import Seq2seq_AMR_Parser


class Seq2seq_CFG_AMR_Parser(Seq2seq_AMR_Parser):
    def collect_additional_tokens(self, additional_tokens, dataset):
        super().collect_additional_tokens(additional_tokens, dataset)
        for sample in dataset:
            amr = sample['amr']
            tree = Tree.from_list(json.loads(amr.metadata['con_list']))
            for s in tree.subtrees():
                additional_tokens.add(s.label())

    def finalize_dataset(self, dataset):
        dataset.append_transform(lambda x: dfs_linearize_constituency(x, tokenizer=self._tokenizer))

    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        # noinspection PyTypeChecker
        model: BartForConditionalGeneration = super().build_model(training, **kwargs)
        config = model.config
        config.max_position_embeddings = 2048
        pos_embed = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        if training:
            with torch.no_grad():
                pos_embed.weight[:model.base_model.encoder.embed_positions.weight.size(0), :] \
                    = model.base_model.encoder.embed_positions.weight
        model.base_model.encoder.embed_positions = pos_embed
        return model
