# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-10-28 14:00
from typing import Callable, List
import torch
from transformers import LogitsProcessorList
from transformers.generation_utils import GenerationMixin
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from elit.components.seq2seq.pos.constrained_decoding import TagProcessor, TokenTagProcessor, IsAProcessor, \
    IsAProcessorQuotation
from elit.components.seq2seq.pos.verbalizer import TagVerbalizer, TokenTagVerbalizer, IsAVerbalizer


class ConstrainedDecoding(GenerationMixin):
    def _get_logits_processor(self, repetition_penalty: float, no_repeat_ngram_size: int,
                              encoder_no_repeat_ngram_size: int, encoder_input_ids: torch.LongTensor,
                              bad_words_ids: List[List[int]], min_length: int, max_length: int, eos_token_id: int,
                              forced_bos_token_id: int, forced_eos_token_id: int,
                              prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int,
                              num_beam_groups: int, diversity_penalty: float,
                              remove_invalid_values: bool,
                              ) -> LogitsProcessorList:
        logits_processor_list = super()._get_logits_processor(repetition_penalty, no_repeat_ngram_size,
                                                              encoder_no_repeat_ngram_size, encoder_input_ids,
                                                              bad_words_ids, min_length, max_length, eos_token_id,
                                                              forced_bos_token_id, forced_eos_token_id,
                                                              prefix_allowed_tokens_fn, num_beams,
                                                              num_beam_groups, diversity_penalty, remove_invalid_values,
                                                              )
        if isinstance(self.config.verbalizer, TagVerbalizer):
            processor = TagProcessor(
                self.config.batch,
                self.config.tags,
                tokenizer=self.config.tokenizer
            )
        elif isinstance(self.config.verbalizer, TokenTagVerbalizer):
            processor = TokenTagProcessor(
                self.config.batch,
                self.config.tags,
                tokenizer=self.config.tokenizer
            )
        elif isinstance(self.config.verbalizer, IsAVerbalizer):
            if self.config.verbalizer.quotation:
                processor = IsAProcessorQuotation(
                    self.config.batch,
                    self.config.trie,
                    self.config.tokenizer,
                )
            else:
                processor = IsAProcessor(
                    self.config.batch,
                    self.config.trie,
                    self.config.tokenizer,
                )
        else:
            raise NotImplementedError()
        logits_processor_list.append(processor)
        return logits_processor_list


class BartForConditionalGenerationExtended(ConstrainedDecoding, BartForConditionalGeneration):
    pass
