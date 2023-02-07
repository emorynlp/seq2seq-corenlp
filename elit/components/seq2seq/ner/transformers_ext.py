# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-10-28 14:00
import warnings
from typing import Callable, List, Optional, Union

import torch
import torch.distributed as dist
from hanlp_trie import Trie
from transformers import LogitsProcessorList, StoppingCriteriaList, BartConfig
from transformers.generation_stopping_criteria import validate_stopping_criteria
from transformers.generation_utils import GenerationMixin, GreedySearchOutput, GreedySearchEncoderDecoderOutput, \
    GreedySearchDecoderOnlyOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from elit.components.seq2seq.ner.constrained_decoding import CopyPrefixConstrainedLogitsProcessor, \
    CopyPrefixConstrainedOracleLogitsProcessor, PairOfTagsProcessor, TagCountProcessor, FirstTokenProcessor, \
    DynamicSwitchProcessor, TagProcessor
from elit.components.seq2seq.ner.prompt_ner import PairOfTagsVerbalizer, TagCountVerbalizer, TagVerbalizer
from elit.layers.transformers.utils import pick_tensor_for_each_token


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
        if isinstance(self.config.verbalizer, PairOfTagsVerbalizer):
            processor = PairOfTagsProcessor(
                encoder_input_ids.tolist(),
                self.config.left_labels,
                self.config.right_labels,
                self.config.eos_token_id,
                self.config.batch,
                tokenizer=self.config.tokenizer
            )
        elif isinstance(self.config.verbalizer, TagCountVerbalizer):
            processor = TagCountProcessor(
                encoder_input_ids.tolist(),
                self.config.labels,
                self.config.counts,
                self.config.eos_token_id,
                self.config.batch,
                tokenizer=self.config.tokenizer
            )
        elif isinstance(self.config.verbalizer, TagVerbalizer):
            processor = TagProcessor(self.config.batch, self.config.verbalizer.labels, self.config.tokenizer)
        else:
            trie = Trie(self.config.valid_label_token_ids)
            processor = CopyPrefixConstrainedLogitsProcessor(encoder_input_ids.tolist(),
                                                             trie,
                                                             self.config.separator_token_id,
                                                             self.config.eos_token_id,
                                                             self.config.pad_token_id,
                                                             self.config.sep_token_id,
                                                             self.config.delimiter,
                                                             self.config.batch,
                                                             tokenizer=self.config.tokenizer)
        logits_processor_list.append(processor)
        return logits_processor_list


class BartForConditionalGenerationExtended(ConstrainedDecoding, BartForConditionalGeneration):
    pass


class MBartForConditionalGenerationExtended(ConstrainedDecoding, MBartForConditionalGeneration):
    pass


class T5ForConditionalGenerationExtended(ConstrainedDecoding, T5ForConditionalGeneration):
    pass


class ConstrainedOracleDecoding(GenerationMixin):
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
        trie = Trie(set([tuple(x[:2]) for x in self.config.valid_label_token_ids]))
        processor = CopyPrefixConstrainedOracleLogitsProcessor(encoder_input_ids.tolist(),
                                                               trie,
                                                               self.config.separator_token_id,
                                                               self.config.eos_token_id,
                                                               self.config.pad_token_id,
                                                               self.config.sep_token_id,
                                                               self.config.delimiter,
                                                               self.config.batch,
                                                               [x[2:] for x in self.config.valid_label_token_ids],
                                                               full_labels=Trie(self.config.valid_label_token_ids),
                                                               tokenizer=self.config.tokenizer)
        logits_processor_list.append(processor)
        return logits_processor_list

    def greedy_search(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = None,
            **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using greedy decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (:obj:`StoppingCriteriaList`, `optional`):
                An instance of :class:`~transformers.StoppingCriteriaList`. List of instances of class derived from
                :class:`~transformers.StoppingCriteria` used to tell if the generation loop should stop.

            max_length (:obj:`int`, `optional`, defaults to 20):
                **DEPRECATED**. Use :obj:`logits_processor` or :obj:`stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the :obj:`forward` function of the
                model. If model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ... AutoTokenizer,
            ... AutoModelForCausalLM,
            ... LogitsProcessorList,
            ... MinLengthLogitsProcessor,
            ... )

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.greedy_search(input_ids, logits_processor=logits_processor)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # pre-process distribution
            next_label = torch.matmul(outputs.decoder_hidden_states[-1], self.get_label_reps().T).argmax(-1).squeeze(
                -1).tolist()
            for x in logits_processor:
                if isinstance(x, CopyPrefixConstrainedOracleLogitsProcessor):
                    x.next_label = next_label
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids


class LabelRepresentation(object):
    def get_label_reps(self):
        label_ids = self.label_ids[None, :]
        reps = self.get_decoder().embed_tokens.weight[None, :]
        reps = pick_tensor_for_each_token(reps, label_ids, True).squeeze(0)
        return reps


class OracleBartForConditionalGenerationExtended(ConstrainedOracleDecoding, BartForConditionalGeneration,
                                                 LabelRepresentation):
    pass


class SwitchableBartForConditionalGeneration(BartForConditionalGeneration):
    FIRST_TOKEN = 0
    FREE = 1

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.status = SwitchableBartForConditionalGeneration.FIRST_TOKEN

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
        first = FirstTokenProcessor(self.config.isa_id, self.config.pot_id)
        if self.status == SwitchableBartForConditionalGeneration.FIRST_TOKEN:
            processor = first
        else:
            trie = Trie(self.config.valid_label_token_ids)
            isa = CopyPrefixConstrainedLogitsProcessor(encoder_input_ids.tolist(),
                                                       trie,
                                                       self.config.separator_token_id,
                                                       self.config.eos_token_id,
                                                       self.config.pad_token_id,
                                                       self.config.sep_token_id,
                                                       self.config.delimiter,
                                                       self.config.batch,
                                                       tokenizer=self.config.tokenizer)
            pot = PairOfTagsProcessor(
                encoder_input_ids.tolist(),
                self.config.left_labels,
                self.config.right_labels,
                self.config.eos_token_id,
                self.config.batch,
                tokenizer=self.config.tokenizer
            )
            processor = DynamicSwitchProcessor(first, isa, pot)
        logits_processor_list.append(processor)
        return logits_processor_list
