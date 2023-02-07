# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-01-06 20:27
import warnings
from typing import List, Optional, Callable, Iterable, Union
import torch
from transformers.generation_stopping_criteria import validate_stopping_criteria
from transformers.generation_utils import GreedySearchOutput, GreedySearchEncoderDecoderOutput, \
    GreedySearchDecoderOnlyOutput
from torch import dist
from elit.components.seq2seq.ner.constrained_decoding import DynamicCopyPrefixConstrainedOracleLogitsProcessor
from hanlp_trie import Trie
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BeamSearchScorer, LogitsProcessorList, StoppingCriteriaList
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.bart.tokenization_bart import BartTokenizer

from elit.components.seq2seq.ner.transformers_ext import LabelRepresentation


def vectorize(x: List[int], device):
    return torch.tensor(x, device=device).unsqueeze(0)


class DynamicOracleBart(BartForConditionalGeneration, LabelRepresentation):
    pass

    def fit(self,
            sample,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = 4,
            window_size=5,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            encoder_no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            max_time: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            num_beam_groups: Optional[int] = None,
            diversity_penalty: Optional[float] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            forced_bos_token_id: Optional[int] = None,
            forced_eos_token_id: Optional[int] = None,
            remove_invalid_values: Optional[bool] = None,
            synced_gpus: Optional[bool] = None,
            **model_kwargs,
            ):
        tokenizer: BartTokenizer = self.config.tokenizer  # For debugging
        input_ids = sample['text_token_ids']
        device = input_ids.device
        x_non_gen = sample['x_non_gen'][0]
        x_gen = sample['x_gen'][0]
        y_non_gen = sample['y_non_gen'][0]

        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        if input_ids is None and "inputs_embeds" not in model_kwargs:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        # Storing encoder_input_ids for logits_processor that could use them
        encoder_input_ids = input_ids if self.config.is_encoder_decoder else None

        # add encoder_outputs to model_kwargs
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

        batch_size = input_ids.shape[0]

        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        loss_fct = CrossEntropyLoss()
        loss = 0
        for x, y, g, l in zip(x_non_gen, y_non_gen, x_gen + [None], sample['label'][0] + [None]):
            x = vectorize(x, device)
            y = vectorize(y, device)
            # g = vectorize(g, device)

            # interleave with `num_beams`
            x, model_kwargs = self._expand_inputs_for_generation(
                x, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            input_ids = x

            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.view((batch_size * num_beams,))
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            window_reps = []
            for i in range(window_size):
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs,
                                                                  cut_decoder_input_ids=i > 0)
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                if not i:
                    if l is None:  # trick: the last None is for generating semicolon
                        loss += loss_fct(outputs.logits.flatten(0, 1), y.expand(x.size(0), y.size(1)).reshape(-1))
                    else:
                        loss += loss_fct(outputs.logits[:, :-1].flatten(0, 1),
                                         y.expand(x.size(0), y.size(1)).reshape(-1))

                next_token_logits = outputs.logits[:, -1, :]
                window_reps.append(outputs.decoder_hidden_states[-1][:, -1, :])
                next_token_scores = nn.functional.log_softmax(
                    next_token_logits, dim=-1
                )  # (batch_size * num_beams, vocab_size)

                next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

                # reshape for beam search
                vocab_size = next_token_scores.shape[-1]
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )

                next_indices = (next_tokens / vocab_size).long()
                next_tokens = next_tokens % vocab_size

                # stateless
                beam_outputs = beam_scorer.process(
                    input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

                # for each in input_ids.tolist():
                #     print(tokenizer.convert_ids_to_tokens(each))

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )
                if model_kwargs["past"] is not None:
                    model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            # for each in input_ids.tolist():
            #     print(tokenizer.convert_ids_to_tokens(each))
            if l is None:
                break
            window_reps = torch.stack(window_reps, dim=1)
            label_reps = self.get_label_reps()
            sim, _ = torch.matmul(window_reps, label_reps.T).max(dim=1)
            pred = sim.argmax(dim=-1)
            wrong_mask = pred != l
            if torch.any(wrong_mask):
                sim = sim[wrong_mask]
                sim_loss = loss_fct(sim, torch.ones(sim.size(0), device=device, dtype=torch.long) * l)
                # if torch.all(wrong_mask):  # fill the beam with gold label
                #     print()
            else:
                sim_loss = 0
            loss += sim_loss

        return loss

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            cut_decoder_input_ids=True,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None and cut_decoder_input_ids:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def _get_logits_processor(self, repetition_penalty: float, no_repeat_ngram_size: int,
                              encoder_no_repeat_ngram_size: int, encoder_input_ids: torch.LongTensor,
                              bad_words_ids: List[List[int]], min_length: int, max_length: int, eos_token_id: int,
                              forced_bos_token_id: int, forced_eos_token_id: int,
                              prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int,
                              num_beam_groups: int, diversity_penalty: float,
                              remove_invalid_values: bool) -> LogitsProcessorList:
        logits_processor_list = super()._get_logits_processor(repetition_penalty, no_repeat_ngram_size,
                                                              encoder_no_repeat_ngram_size, encoder_input_ids,
                                                              bad_words_ids, min_length, max_length, eos_token_id,
                                                              forced_bos_token_id, forced_eos_token_id,
                                                              prefix_allowed_tokens_fn, num_beams,
                                                              num_beam_groups, diversity_penalty, remove_invalid_values)
        trie = Trie(set([tuple(x[:2]) for x in self.config.valid_label_token_ids]))
        processor = DynamicCopyPrefixConstrainedOracleLogitsProcessor(encoder_input_ids.tolist(),
                                                                      trie,
                                                                      self.config.separator_token_id,
                                                                      self.config.eos_token_id,
                                                                      self.config.pad_token_id,
                                                                      self.config.sep_token_id,
                                                                      self.config.delimiter,
                                                                      self.config.batch,
                                                                      [x[2:] for x in
                                                                       self.config.valid_label_token_ids],
                                                                      full_labels=Trie(
                                                                          self.config.valid_label_token_ids),
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
        window_reps = []
        tokenizer: BartTokenizer = self.config.tokenizer  # For debugging
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

            constrained: DynamicCopyPrefixConstrainedOracleLogitsProcessor = logits_processor[-1]
            if constrained.num_label_tokens > 1:
                window_reps.append(outputs.decoder_hidden_states[-1][0, -1, :])

            # pre-process distribution
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

            if len(window_reps) == 5:
                window_reps = torch.stack(window_reps, dim=0)
                label_reps = self.get_label_reps()
                sim, _ = torch.matmul(window_reps, label_reps.T).max(dim=0)
                pred = sim.argmax(dim=-1)
                window_reps = []
                pred_ids = self.config.valid_label_token_ids[pred][2:-1]
                # cprint(f'[magenta]{tokenizer.convert_ids_to_tokens(input_ids[0, -5:].tolist())} '
                #        f'-> {tokenizer.convert_ids_to_tokens(pred_ids)}[/magenta]')
                input_ids[0, -5:] = tokenizer.pad_token_id
                input_ids[0, -5:-5 + len(pred_ids)] = torch.tensor(pred_ids, dtype=torch.long, device=input_ids.device)

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
