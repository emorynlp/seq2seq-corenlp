# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-01-19 10:01
import string
from typing import Callable, List, Optional

import torch
from transformers import LogitsProcessorList
from transformers.generation_logits_process import LogitsProcessor
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.tokenization_utils import PreTrainedTokenizer

from elit.components.seq2seq.ner.conditional_seq2seq import ConditionalSeq2seq


class ContentLogitsProcessor(LogitsProcessor):

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.bad_ids = sorted(
            sum(tokenizer(list(string.punctuation), add_special_tokens=False).input_ids, [tokenizer.eos_token_id]))
        self.pred_length = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.pred_length:
            return scores
        self.pred_length += 1
        mask = torch.full_like(scores, 0)
        mask[:, self.bad_ids] = float('-inf')
        return scores + mask


class ConstrainedBartForConditionalGeneration(BartForConditionalGeneration):
    def _get_logits_processor(self, repetition_penalty: float, no_repeat_ngram_size: int,
                              encoder_no_repeat_ngram_size: int, encoder_input_ids: torch.LongTensor,
                              bad_words_ids: List[List[int]], min_length: int, max_length: int, eos_token_id: int,
                              forced_bos_token_id: int, forced_eos_token_id: int,
                              prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int,
                              num_beam_groups: int, diversity_penalty: float,
                              remove_invalid_values: bool,
                              logits_processor: Optional[LogitsProcessorList]) -> LogitsProcessorList:
        logits_processor_list = super()._get_logits_processor(repetition_penalty, no_repeat_ngram_size,
                                                              encoder_no_repeat_ngram_size, encoder_input_ids,
                                                              bad_words_ids, min_length, max_length, eos_token_id,
                                                              forced_bos_token_id, forced_eos_token_id,
                                                              prefix_allowed_tokens_fn, num_beams,
                                                              num_beam_groups, diversity_penalty, remove_invalid_values,
                                                              logits_processor)
        processor = ContentLogitsProcessor(tokenizer=self.config.tokenizer)
        logits_processor_list.append(processor)
        return logits_processor_list


class ConstrainedConditionalSeq2seq(ConditionalSeq2seq):
    def model_cls(self, **kwargs):
        return ConstrainedBartForConditionalGeneration
