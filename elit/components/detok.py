# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-11-26 20:32
from typing import Any

from elit.components.taggers.transformers.transformer_tagger import TransformerTagger
from elit.datasets.detokenization.detok import DetokenizationDataset


class TransformerDetokenizer(TransformerTagger):
    def build_dataset(self, data, transform=None, **kwargs):
        return DetokenizationDataset(data, transform=transform, **kwargs)

    def predict(self, tokens: Any, batch_size: int = None, ret_scores=False, ret_tags=False, **kwargs):
        tags = super().predict(tokens, batch_size, ret_scores, **kwargs)
        if not ret_tags:
            flat = self.input_is_flat(tokens)
            if flat:
                tags = [tags]
                tokens = [tokens]
            sents = [''.join(sum(list(zip(token, tag)), ())) for token, tag in zip(tokens, tags)]
            if flat:
                return sents[0]
            return sents
        return tags
