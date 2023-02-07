# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-02-24 15:58
import copy
import json
import logging
import os
from typing import Union, Callable

import torch
from hanlp_common.configurable import AutoConfigurable
from hanlp_common.constant import IDX
from hanlp_common.util import merge_locals_kwargs, split_dict, reorder
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from elit.common.structure import History
from elit.components.seq2seq.ner.constrained_decoding import FirstTokenProcessor, DynamicSwitchProcessor
from elit.components.seq2seq.ner.prompt_ner import to_exclusive_offset
from elit.components.seq2seq.ner.seq2seq_ner import Seq2seqNamedEntityRecognizer, \
    create_and_tokenize_pair_of_tags_prompt, create_and_tokenize_is_a_prompt
from elit.components.seq2seq.ner.transformers_ext import SwitchableBartForConditionalGeneration
from elit.metrics.chunking.chunking_f1 import DetailedSpanF1
from elit.metrics.f1 import F1
from elit.utils.io_util import replace_ext
from elit.utils.time_util import CountdownTimer


class DynamicSeq2seqNamedEntityRecognizer(Seq2seqNamedEntityRecognizer):
    def fit(self, trn_data, dev_data, save_dir, verbalizer: AutoConfigurable = None, batch_size=32, epochs=30,
            transformer='facebook/bart-base', lr=5e-05, grad_norm=2.5, weight_decay=0.004, warmup_steps=1, dropout=0.25,
            attention_dropout=0.0, eval_after=0.5, gradient_accumulation=1, doc_level_offset=True,
            optimizer_name='radam', oracle=False, devices=None, logger=None, seed=None,
            finetune: Union[bool, str] = False, eval_trn=True, use_detokenization=False, transform=None,
            pot_verbalizer=None, is_a_verbalizer=None, switch_by_f1=False, switch_after_epoch=1,
            src_lang: str = None, tgt_lang: str = None, max_seq_len=None, gazetteer: str = None, doc_context=0,
            _device_placeholder=False, **kwargs):
        verbalizer = is_a_verbalizer
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def finalize_dataset(self, dataset, logger: logging.Logger = None, use_detokenization=False, src_lang: str = None,
                         tgt_lang: str = None, max_seq_len=None, gazetteer_verbalizer=None, doc_context=0, oracle=False,
                         verbalizer=None,
                         pot_verbalizer=None, is_a_verbalizer=None,
                         **kwargs):
        dataset.append_transform(to_exclusive_offset)
        # Now we want to process the data in two ways
        dataset.append_transform(lambda x: process_two_ways(x, pot_verbalizer, is_a_verbalizer, self._tokenizer))

    def _get_model_cls(self, transformer: str, oracle=False):
        # return BartForConditionalGeneration
        return SwitchableBartForConditionalGeneration

    def collect_additional_tokens(self, pot_verbalizer=None, **kwargs):
        additional_tokens = pot_verbalizer.get_additional_tokens()
        additional_tokens.append(FirstTokenProcessor.ISA)
        additional_tokens.append(FirstTokenProcessor.POT)
        return additional_tokens

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, dev_data=None, eval_after=None,
                              **kwargs):
        best_epoch, best_metric = 0, -1
        if isinstance(eval_after, float):
            eval_after = int(epochs * eval_after)
        timer = CountdownTimer(epochs)
        history = History()
        self.config.training = True
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger, history=history, ratio_width=ratio_width,
                                **self.config, epoch=epoch)
            # TODO: Remove this
            # self.save_weights(save_dir)
            if epoch > eval_after:
                dev_metric = self.evaluate_dataloader(dev, criterion, logger=logger, ratio_width=ratio_width,
                                                      # output=os.path.join(save_dir, 'dev.pred.txt'),
                                                      input=dev_data, use_fast=True)
            timer.update()
            report = f"{timer.elapsed_human} / {timer.total_time_human} ETA: {timer.eta_human}"
            if epoch > eval_after:
                if dev_metric > best_metric:
                    best_epoch, best_metric = epoch, dev_metric
                    self.save_weights(save_dir)
                    report += ' [red](saved)[/red]'
                else:
                    report += f' ({epoch - best_epoch})'
                # if epoch - best_epoch >= patience:
                #     report += ' early stop'
            logger.info(report)
            # if epoch - best_epoch >= patience:
            #     break
        if not best_epoch:
            self.save_weights(save_dir)
        elif best_epoch != epoch:
            self.load_weights(save_dir)
        logger.info(f"Max score of dev is {best_metric} at epoch {best_epoch}")
        logger.info(f"Average time of each epoch is {timer.elapsed_average_human}")
        logger.info(f"{timer.elapsed_human} elapsed")
        return best_metric

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, epoch=1,
                       history: History = None, gradient_accumulation=1, ratio_percentage=None, oracle=False,
                       switch_by_f1=False, switch_after_epoch=1, **kwargs):
        optimizer, scheduler = optimizer
        self.model.train()
        self.model.status = SwitchableBartForConditionalGeneration.FIRST_TOKEN
        timer = CountdownTimer(history.num_training_steps(len(trn), gradient_accumulation=gradient_accumulation))
        total_loss = 0
        num_isa = 0
        num_total = 0
        for batch in trn:
            num_total += len(batch['token'])
            if epoch <= switch_after_epoch:
                isa_loss = self.feed_batch(self._pick_batch(batch, 'isa_'), oracle)
                pot_loss = self.feed_batch(self._pick_batch(batch, 'pot_'), oracle)
                loss = (isa_loss + pot_loss) / 2
            else:
                # run generation first
                if switch_by_f1:
                    isa_mask = self.switch_by_f1(batch)
                    self.config.decoder_input_ids = None
                else:
                    isa_mask = self.switch_by_first_token(batch)

                isa_loss = self.feed_batch(self._pick_batch_tensor(batch, 'isa_', isa_mask),
                                           oracle) if isa_mask.any() else 0
                num_isa += isa_mask.sum().item()
                pot_mask = ~isa_mask
                pot_loss = self.feed_batch(self._pick_batch_tensor(batch, 'pot_', pot_mask),
                                           oracle) if pot_mask.any() else 0
                loss = (isa_loss + pot_loss) / 2
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            if history.step(gradient_accumulation):
                self._step(optimizer, scheduler)
                timer.log(f'loss: {total_loss / (timer.current + 1):.4f} isa: {num_isa / num_total:.2%}',
                          ratio_percentage=ratio_percentage, logger=logger)
            del loss
        return total_loss / max(timer.total, 1)

    def switch_by_first_token(self, batch):
        input_ids = batch['isa_text_token_ids']
        self._transformer_config.batch = batch
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=3,
            )
        self.model.train()
        first_token = out[:, 1]
        isa_mask = first_token == self._transformer_config.isa_id
        return isa_mask

    def _pick_batch_tensor(self, batch, prefix, isa_mask):
        return DynamicSwitchProcessor.pick_batch_tensor(batch, prefix, isa_mask)

    def _pick_batch(self, batch, prefix):
        isa_batch = copy.copy(batch)
        for k, v in batch.items():
            if k.startswith(prefix):
                isa_batch[k[len(prefix):]] = v
        return isa_batch

    def on_config_ready(self, gazetteer=None, doc_context=0, oracle=False, verbalizer=None, pot_verbalizer=None,
                        **kwargs):
        super().on_config_ready(gazetteer, doc_context, oracle, verbalizer, **kwargs)
        self._transformer_config.isa_id = self._tokenizer.convert_tokens_to_ids(FirstTokenProcessor.ISA)
        self._transformer_config.pot_id = self._tokenizer.convert_tokens_to_ids(FirstTokenProcessor.POT)
        self._prepare_pot_config(pot_verbalizer)

    def _decode_enterties_per_seq(self, batch, normalized_tokens, prompt, sample_index, verbalizer):
        # print(prompt)
        # return []
        isa_mask = batch['_isa_mask']
        if not isa_mask[sample_index].item():
            entities_per_seq = batch['_predictions'][sample_index]
        else:
            entities_per_seq = verbalizer.prompt_to_entities(prompt, normalized_tokens)
        return entities_per_seq

    def _model_generate(self, batch, beam_size):
        self.model.status = SwitchableBartForConditionalGeneration.FREE
        input_ids = batch['text_token_ids']
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        kwargs = {}
        decoder_input_ids = self.config.get('decoder_input_ids', None)
        if decoder_input_ids is not None:
            kwargs['decoder_input_ids'] = decoder_input_ids
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1024,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            **kwargs
        )
        return out

    def _write_samples(self, samples, out):
        out.write('text\tpred\tgold\tP\tR\tF1\n')
        for sample in samples:
            if not sample['pred_entity'] and not sample['ner']:
                continue
            p, r, f = sample['score'].prf
            text = ' '.join(sample['token'])
            out.write(f'{text}\t{sample["pred_prompt"]}\t{sample["isa_prompt"]}\t'
                      f'{p:.2%}\t{r:.2%}\t{f:.2%}\n')

    @torch.no_grad()
    def switch_by_f1(self, batch):
        self.model.eval()
        entities_isa = self.pred_force_mode(batch, 'isa_', self._transformer_config.isa_id)
        entities_pot = self.pred_force_mode(batch, 'pot_', self._transformer_config.pot_id)
        self.model.train()
        mask = []
        for isa, pot, gold in zip(entities_isa, entities_pot, batch['ner']):
            isa_f1 = self.evaluate_entities(isa, gold)
            pot_f1 = self.evaluate_entities(pot, gold)
            mask.append(isa_f1 > pot_f1)
        return torch.tensor(mask, device=self.device)

    def pred_force_mode(self, batch, prefix, prefix_id):
        batch_size = batch['isa_text_token_ids'].size(0)
        isa_batch = self._pick_batch_tensor(batch, prefix,
                                            torch.ones((batch_size,), device=self.device, dtype=torch.bool))
        self.config.decoder_input_ids = torch.tensor([0, prefix_id], device=self.device,
                                                     dtype=torch.long).tile([batch_size, 1])
        entities_isa, pred_prompts = self.predict_ners(isa_batch)
        return [x[0] for x in entities_isa]

    def evaluate_entities(self, pp, gg):
        score = F1()
        score(set([tuple(x) for x in pp]), set([tuple(x) for x in gg]))
        return score

    @torch.no_grad()
    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, ratio_width=None,
                            logger=None, input=None, use_fast=False, save_dir=None, filename=None,
                            **kwargs):
        self.model.eval()
        timer = CountdownTimer(len(data))
        samples = []
        orders = []
        metric = DetailedSpanF1(do_confusion_matrix=True) if output else F1()
        num_samples = 0
        num_isa = 0
        for idx, batch in enumerate(data):
            entities_per_batch, pred_prompts = self.predict_ners(batch)
            num_samples += len(pred_prompts)
            num_isa += len([x for x in pred_prompts if x.startswith(FirstTokenProcessor.ISA)])
            entities_per_batch = [x[0] for x in entities_per_batch]
            for gp, gg, tokens in zip(entities_per_batch, batch['ner'], batch['token']):
                gp = [tuple(x) for x in gp]
                if output:
                    metric(set(gp), set(gg), num_tokens=len(tokens))
                else:
                    metric(set(gp), set(gg))

            if output:
                batch['pred_entity'] = entities_per_batch
                batch['pred_prompt'] = pred_prompts
                samples.extend(split_dict(batch))
                orders.extend(batch[IDX])
            timer.log(f'{metric} isa: {num_isa / num_samples:.2%}', ratio_percentage=False, logger=logger)

        if output:
            samples = reorder(samples, orders)
            for sample in samples:
                _metric = F1()
                _metric(set(sample['pred_entity']), set(sample['ner']))
                sample['score'] = _metric

            if output == '.jsonlines':
                output = os.path.join(save_dir, f'pred-{filename}')
                with open(output, 'w') as out:
                    for sample in samples:
                        out.write(json.dumps(
                            {'sentences': [sample['token']],
                             'ner': [[(b, e - 1, l) for b, e, l in sample['pred_entity']]]}) + '\n')
            else:
                output = replace_ext(output, '.tsv')
                with open(output, 'w') as out, open(replace_ext(output, '-sorted.tsv'), 'w') as out_sorted:
                    samples = sorted(samples, key=lambda x: x['score'].prf[1])
                    self._write_samples(samples, out)

                    samples = sorted(samples, key=lambda x: tuple(x['token']))
                    self._write_samples(samples, out_sorted)
        return metric


def process_two_ways(sample: dict, pot_verbalizer=None, is_a_verbalizer=None, tokenizer: BartTokenizer = None):
    sample_both = copy.copy(sample)
    sample_pot = create_and_tokenize_pair_of_tags_prompt(copy.copy(sample), pot_verbalizer, tokenizer)
    for key in sample_pot.keys() - sample.keys():
        sample_both[f'pot_{key}'] = sample_pot[key]
    sample_isa = create_and_tokenize_is_a_prompt(
        copy.copy(sample),
        tokenizer, use_detokenization=False,
        src_lang=None, tgt_lang=None,
        verbalizer=is_a_verbalizer,
        delimiter_id=None,
        gazetteer=None,
        gazetteer_verbalizer=None,
        doc_context=False,
    )
    for key in sample_isa.keys() - sample.keys():
        sample_both[f'isa_{key}'] = sample_isa[key]
    if 'ner' in sample:
        sample_both['pot_prompt_token_ids'].insert(1, tokenizer.convert_tokens_to_ids(FirstTokenProcessor.POT))
        sample_both['isa_prompt_token_ids'].insert(1, tokenizer.convert_tokens_to_ids(FirstTokenProcessor.ISA))
    assert sample_pot['text_token_ids'] == sample_isa['text_token_ids']
    sample_both['text_token_ids'] = sample_isa['text_token_ids']
    sample_both['normalized_tokens'] = sample_isa['normalized_tokens']
    return sample_both
