# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-10-28 14:00
from typing import Callable, List
import torch

from elit.components.seq2seq.dep.constrained_decoding import HeadRelationProcessor, BracketProcessor, \
    TransitionProcessor, LexicalTransitionProcessor, PromptProcessor
from elit.components.seq2seq.dep.verbalizer import HeadRelationVerbalizer, BracketedVerbalizer, ArcEagerVerbalizer, \
    ArcStandardVerbalizer, PromptVerbalizer
from transformers import LogitsProcessorList
from transformers.generation_utils import GenerationMixin
from transformers.models.bart.modeling_bart import BartForConditionalGeneration


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
        verbalizer = self.config.verbalizer
        tokenizer = self.config.tokenizer
        if isinstance(verbalizer, HeadRelationVerbalizer):
            processor = HeadRelationProcessor(
                self.config.batch,
                verbalizer.get_head_token_ids(tokenizer),
                verbalizer.get_relation_token_ids(tokenizer),
                tokenizer=tokenizer
            )
        elif isinstance(verbalizer, BracketedVerbalizer):
            processor = BracketProcessor(
                self.config.batch,
                verbalizer.get_relation_token_ids(tokenizer),
                verbalizer.get_lb_id(tokenizer),
                verbalizer.get_rb_id(tokenizer),
                tokenizer=tokenizer
            )
        elif isinstance(verbalizer, (ArcEagerVerbalizer, ArcStandardVerbalizer)):
            if verbalizer.lexical:
                processor = LexicalTransitionProcessor(
                    self.config.batch,
                    verbalizer.get_action_token_ids(tokenizer),
                    tokenizer=tokenizer
                )
            else:
                processor = TransitionProcessor(
                    self.config.batch,
                    verbalizer.get_action_token_ids(tokenizer),
                    tokenizer=tokenizer
                )
        elif isinstance(verbalizer, PromptVerbalizer):
            processor = PromptProcessor(
                self.config.batch,
                tokenizer=tokenizer,
                trie=self.config.trie
            )
        else:
            raise NotImplementedError()
        logits_processor_list.append(processor)
        return logits_processor_list


class BartForConditionalGenerationExtended(ConstrainedDecoding, BartForConditionalGeneration):
    pass
