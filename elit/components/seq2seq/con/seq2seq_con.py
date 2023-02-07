# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-28 17:33
import json
import logging
import os
from typing import Union, List, Callable

import torch
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, BartTokenizer, AutoTokenizer, BertTokenizer, \
    BartForConditionalGeneration, optimization

from elit.common.dataset import SamplerBuilder, SortingSamplerBuilder, PadSequenceDataLoader
from elit.common.structure import History
from elit.common.torch_component import TorchComponent
from elit.components.amr.seq2seq.optim import RAdam
from elit.components.seq2seq.con.transformers_ext import BartForConditionalGenerationExtended
from elit.components.seq2seq.con.verbalizer import Verbalizer, ShiftReduceVerbalizer
from elit.components.seq2seq.ner.seq2seq_ner import tokenize
from elit.datasets.parsing.loaders.constituency_dataset import ConstituencyDataset, factorize
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING
from elit.layers.transformers.pt_imports import PretrainedConfig, AutoConfig_
from elit.layers.transformers.resource import get_model_mirror, get_tokenizer_mirror
from elit.layers.transformers.utils import build_optimizer_for_pretrained
from elit.metrics.parsing.evalb_bracketing_scorer import EvalbBracketingScorer
from elit.metrics.parsing.span import SpanMetric
from elit.utils.time_util import CountdownTimer
from hanlp_common.constant import IDX
from hanlp_common.util import merge_locals_kwargs, reorder, split_dict


class Seq2seqConstituencyParser(TorchComponent):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._transformer_config: PretrainedConfig = None
        self._tokenizer: BartTokenizer = None
        self.model: BartForConditionalGenerationExtended = None

    def build_dataloader(self, data, batch_size,
                         gradient_accumulation=1,
                         shuffle=False,
                         sampler_builder: SamplerBuilder = None,
                         device=None,
                         logger: logging.Logger = None,
                         transform=None,
                         max_seq_len=None,
                         max_prompt_len=None,
                         verbalizer: Verbalizer = None,
                         **kwargs) -> DataLoader:
        dataset = self.build_dataset(data, not shuffle, **self.config)
        dataset.append_transform(verbalizer)
        if isinstance(data, str) and shuffle:
            for each in dataset:
                pass
            self.build_tokenizer(additional_tokens=verbalizer.get_special_tokens())

        self.finalize_dataset(dataset, verbalizer)
        if isinstance(data, str):
            dataset.purge_cache()
            timer = CountdownTimer(len(dataset))
            _max_num_tokens = 0
            _max_prompt_len = 0
            for each in dataset:
                _max_num_tokens = max(_max_num_tokens, len(each['text_token_ids']))
                if _max_prompt_len < len(each['prompt_token_ids']):
                    _max_prompt_len = len(each['prompt_token_ids'])
                    # print(each['prompt'])
                timer.log(f'Preprocessing and caching samples (longest sequence {_max_num_tokens}, '
                          f'longest prompt {_max_prompt_len}) '
                          f'[blink][yellow]...[/yellow][/blink]')
            if max_seq_len:
                dataset.prune(lambda x: len(x['text_token_ids']) > max_seq_len, logger)
            if max_prompt_len:
                dataset.prune(lambda x: len(x['prompt_token_ids']) > max_prompt_len, logger)
            # dataset.prune(lambda x: not x['prompt'], logger)

        if not sampler_builder:
            sampler_builder = SortingSamplerBuilder(batch_max_tokens=6000, use_effective_tokens=True)
        sampler = sampler_builder.build([len(x['text_token_ids']) for x in dataset], shuffle, gradient_accumulation)
        return self._create_dataloader(dataset, batch_size, device, sampler, shuffle)

    def _create_dataloader(self, dataset, batch_size, device, sampler, shuffle):
        return PadSequenceDataLoader(dataset, batch_size, shuffle, device=device, batch_sampler=sampler,
                                     pad=self._get_pad_dict())

    def _get_pad_dict(self):
        return {'text_token_ids': self._transformer_config.pad_token_id,
                'graph_token_ids': self._transformer_config.pad_token_id}

    def finalize_dataset(
            self,
            dataset,
            verbalizer=None
    ):
        dataset.append_transform(lambda x: tokenize_prompt(
            x,
            self._tokenizer,
            verbalizer,
        ))

    def build_dataset(self, data, generate_idx, **kwargs):
        dataset = ConstituencyDataset(data, generate_idx=generate_idx)
        return dataset

    def build_tokenizer(self, additional_tokens=None) -> BartTokenizer:
        transformer = self.config.transformer
        # if 't5-' in transformer:
        #     cls = T5TokenizerFast
        # elif 'bart-' in transformer:
        #     cls = BartTokenizer
        # else:
        #     raise NotImplemented(f'Unsupported transformer {transformer}')
        if transformer == 'fnlp/bart-large-chinese':
            cls = BertTokenizer
        else:
            cls = AutoTokenizer
        transformer = get_tokenizer_mirror(transformer)
        self._tokenizer = self._transformer_config.tokenizer = cls.from_pretrained(
            transformer,
            config=self._transformer_config,
        )
        if additional_tokens:
            self._tokenizer.add_tokens(additional_tokens)
        if 'tag' in self.vocabs:
            tags = self.vocabs['tag'].idx_to_token
            self._transformer_config.tags = dict(zip(self._tokenizer.convert_tokens_to_ids(tags), tags))

        verbalizer = self.config.verbalizer

        return self._tokenizer

    def build_optimizer(self, trn, lr, epochs, gradient_accumulation, warmup_steps, weight_decay,
                        optimizer_name='radam', lr_scheduler='constant',
                        **kwargs):
        num_training_steps = len(trn) * epochs // gradient_accumulation
        if isinstance(warmup_steps, float):
            warmup_steps = int(num_training_steps * warmup_steps)
        if optimizer_name == 'radam':
            optimizer = RAdam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay)
        else:
            optimizer = build_optimizer_for_pretrained(self.model, None, lr, weight_decay,
                                                       no_decay=('layer_norm', '.bias'))
        if lr_scheduler == 'constant':
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps)
        elif lr_scheduler == 'linear':
            scheduler = optimization.get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
        else:
            raise NotImplemented(f'Unsupported schedule {lr_scheduler}')
        return optimizer, scheduler

    def build_criterion(self, **kwargs):
        pass

    def build_metric(self, **kwargs):
        pass

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, dev_data=None, eval_after=None,
                              save_every_epoch=False, **kwargs):
        best_epoch, best_metric = 0, -1
        if isinstance(eval_after, float):
            eval_after = int(epochs * eval_after)
        timer = CountdownTimer(epochs)
        history = History()
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger, history=history, ratio_width=ratio_width,
                                **self.config)
            if save_every_epoch:
                self.save_weights(save_dir)
            elif epoch > eval_after:
                dev_metric = self.evaluate_dataloader(dev, criterion, logger=logger, ratio_width=ratio_width,
                                                      # output=os.path.join(save_dir, 'dev.pred.txt'),
                                                      input=dev_data, use_fast=True)
            timer.update()
            report = f"{timer.elapsed_human} / {timer.total_time_human} ETA: {timer.eta_human}"
            if epoch > eval_after and not save_every_epoch:
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

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger,
                       history: History = None, gradient_accumulation=1, ratio_percentage=None, oracle=False, **kwargs):
        optimizer, scheduler = optimizer
        self.model.train()
        timer = CountdownTimer(history.num_training_steps(len(trn), gradient_accumulation=gradient_accumulation))
        total_loss = 0
        for batch in trn:
            loss = self.feed_batch(batch)
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            if history.step(gradient_accumulation):
                self._step(optimizer, scheduler)
                timer.log(self.report_metrics(total_loss / (timer.current + 1)),
                          ratio_percentage=ratio_percentage, logger=logger)
            del loss
        return total_loss / max(timer.total, 1)

    def _step(self, optimizer, scheduler):
        if self.config.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

    def report_metrics(self, loss):
        return f'loss: {loss:.4f}'

    def feed_batch(self, batch, text_token_ids='text_token_ids', prompt_token_ids='prompt_token_ids',
                   **kwargs):
        input_ids, labels = batch[text_token_ids], batch.get(prompt_token_ids)
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        if labels is not None:
            if isinstance(self.model,
                          (BartForConditionalGenerationExtended, BartForConditionalGeneration)) \
                    and self.model.config.name_or_path != 'fnlp/bart-large-chinese':
                decoder_input_ids = labels[:, :-1]
                labels = labels[:, 1:].contiguous()
            else:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
        else:
            decoder_input_ids = None
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                             labels=labels)

        return outputs.loss

    @torch.no_grad()
    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, ratio_width=None,
                            logger=None, input=None, use_fast=False, save_dir=None, filename=None,
                            delete=('', ':', '``', "''", '.', '?', '!', '-NONE-', 'TOP', ',', 'S1'),
                            equal=(('ADVP', 'PRT'),), official=False,
                            **kwargs):
        if isinstance(equal, (list, tuple)):
            equal = dict(equal)
        self.model.eval()
        timer = CountdownTimer(len(data))
        samples = []
        orders = []
        metric = EvalbBracketingScorer() if official else SpanMetric()
        for idx, batch in enumerate(data):
            trees, pred_prompts = self.predict_con(batch)
            trees = [x[0] for x in trees]
            # Make sure token and pos match. It's OK to use gold since these are not part of evalb scores
            gold_trees = batch['constituency']
            # gold_trees = batch['raw_gold']
            for p, g in zip(trees, gold_trees):
                for pt, gt in zip(p.subtrees(lambda x: x.height() == 2), g.subtrees(lambda x: x.height() == 2)):
                    pt.set_label(gt.label())
                    pt[0] = gt[0]
            if official:
                metric(trees, gold_trees)
            else:
                metric([factorize(tree, delete, equal) for tree in trees],
                       [factorize(tree, delete, equal) for tree in gold_trees])

            if output:
                batch['metric'] = metrics = []
                for p, g in zip(trees, gold_trees):
                    if official:
                        _metric = EvalbBracketingScorer()
                        _metric([p], [g])
                    else:
                        _metric = SpanMetric()
                        _metric([factorize(p, delete, equal)],
                                [factorize(g, delete, equal)])
                    sample_score = float(_metric)
                    metrics.append(sample_score)
                batch['pred'] = trees
                batch['pred_prompt'] = pred_prompts
                samples.extend(split_dict(batch))
                orders.extend(batch[IDX])
            timer.log(f'{metric}', ratio_percentage=False, logger=logger)

        if output:
            samples = reorder(samples, orders)
            output = os.path.join(save_dir, f'pred-{filename}.jsonl')
            with open(output, 'w') as out:
                for sample in samples:
                    out.write(json.dumps(
                        {
                            'token': sample['token'],
                            'gold': sample['constituency'].to_list(),
                            'pred': sample['pred'].to_list(),
                            'gold_prompt': sample['prompt'],
                            'pred_prompt': sample['pred_prompt'],
                            'f': sample['metric'],
                        }, ensure_ascii=False) + '\n')
        return metric

    def _write_samples(self, samples, out):
        out.write('text\tpred\tgold\tP\tR\tF1\n')
        for sample in samples:
            if not sample['pred_entity'] and not sample['ner']:
                continue
            p, r, f = sample['score'].prf
            text = ' '.join(sample['token'])
            out.write(f'{text}\t{sample["pred_prompt"]}\t{sample["prompt"]}\t'
                      f'{p:.2%}\t{r:.2%}\t{f:.2%}\n')

    def predict_con(self, batch, beam_size=1):
        self.model.config.batch = batch  # Bad practise but convenient to pass arguments
        out = self._model_generate(batch, beam_size)
        tokens = []
        for i1 in range(0, out.size(0), beam_size):
            tokens_same_source = []
            tokens.append(tokens_same_source)
            for i2 in range(i1, i1 + beam_size):
                tokk = out[i2].tolist()
                tokens_same_source.append(tokk)
        tokens = [t for tt in tokens for t in tt]
        tags = []
        pred_prompts = []
        tokenizer = self._tokenizer
        verbalizer: Verbalizer = self.config.verbalizer
        for i1 in range(0, len(tokens), beam_size):
            sample_index = i1 // beam_size
            tags_same_source = []
            tags.append(tags_same_source)
            for i2 in range(i1, i1 + beam_size):
                tokk = tokens[i2]
                prompt: str = tokenizer.decode(tokk, clean_up_tokenization_spaces=False)
                if isinstance(tokenizer, BertTokenizer):
                    prompt = prompt[len('[SEP]'):-len('[SEP]')]
                    prompt = prompt.replace(' ', '')
                prompt = prompt.replace('<pad>', '')
                if prompt.startswith('<s>'):
                    prompt = prompt[len('<s>'):]
                if prompt.endswith('</s>'):
                    prompt = prompt[:-len('</s>')]
                if prompt.startswith('</s>'):
                    prompt = prompt[len('</s>'):]
                original_prompt = None
                if i2 == i1:
                    pred_prompts.append(original_prompt if original_prompt is not None else prompt)
                normalized_tokens = batch['normalized_tokens'][sample_index]
                pred_per_seq = self._decode_enterties_per_seq(batch, normalized_tokens, prompt, sample_index,
                                                              verbalizer)
                tags_same_source.append(pred_per_seq)
        return tags, pred_prompts

    def _decode_enterties_per_seq(self, batch, normalized_tokens, prompt, sample_index, verbalizer: Verbalizer):
        return verbalizer.decode(batch, sample_index, prompt)

    def _model_generate(self, batch, beam_size):
        input_ids = batch['text_token_ids']
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1024,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            decoder_start_token_id=None
        )
        return out

    def build_model(self, training=True, gazetteer: str = None, doc_context=0, oracle=False,
                    verbalizer=None,
                    **kwargs) -> torch.nn.Module:
        # noinspection PyTypeChecker
        transformer = self.config.transformer
        cls = self._get_model_cls(transformer, oracle)
        transformer = get_model_mirror(self.config.transformer)
        model: cls = cls.from_pretrained(
            transformer,
            config=self._transformer_config) if training else cls(self._transformer_config)
        # if cls == MBartForConditionalGenerationExtended:
        #     model.config.forced_bos_token_id = self._tokenizer.lang_code_to_id[self.config.tgt_lang]
        if model.get_input_embeddings().weight.size(0) != len(self._tokenizer):
            model.resize_token_embeddings(len(self._tokenizer))
        return model

    def _get_model_cls(self, transformer: str, save_every_epoch=False):
        constrained_decoding = self.config.get('constrained_decoding', None)
        if constrained_decoding:
            return BartForConditionalGenerationExtended
        return BartForConditionalGeneration

    def predict(self, data: Union[List[str], List[List[str]]], **kwargs):
        if not data:
            return []
        flat = self.input_is_flat(data)
        if flat:
            data = [data]
        dataloader = self.build_dataloader([{'FORM': x} for x in data], **self.config, device=self.device)
        orders = []
        results = []
        for batch in dataloader:
            tags, _ = self.predict_con(batch)
            tags = [x[0] for x in tags]
            results.extend(tags)
            orders.extend(batch[IDX])
        results = reorder(results, orders)
        if flat:
            results = results[0]
        return results

    def fit(self, trn_data, dev_data, save_dir,
            verbalizer: Verbalizer,
            batch_size=32,
            epochs=30,
            transformer='facebook/bart-large',
            lr=1e-05,
            grad_norm=2.5,
            weight_decay=0.004,
            warmup_steps=1,
            dropout=0.25,
            attention_dropout=0.0,
            eval_after=0.5,
            gradient_accumulation=1,
            optimizer_name='radam',
            lr_shcedular='constant',
            oracle=False,
            devices=None,
            logger=None,
            seed=None,
            finetune: Union[bool, str] = False,
            eval_trn=True,
            max_seq_len=None,
            save_every_epoch=False,
            max_prompt_len=None,
            delete=('', ':', '``', "''", '.', '?', '!', '-NONE-', 'TOP', ',', 'S1'),
            equal=(('ADVP', 'PRT'),),
            _device_placeholder=False,
            **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def on_config_ready(self, gazetteer=None, doc_context=0, oracle=False, verbalizer=None, **kwargs):
        super().on_config_ready(**kwargs)
        config = AutoConfig_.from_pretrained(self.config.transformer)
        config.output_past = False
        config.no_repeat_ngram_size = 0
        config.prefix = " "
        # config.output_attentions = True
        if hasattr(config, 'dropout'):
            config.dropout = self.config.dropout
        else:
            config.dropout_rate = self.config.dropout
        config.attention_dropout = self.config.attention_dropout
        self._transformer_config = config
        self._transformer_config.forced_bos_token_id = None
        self._transformer_config.verbalizer = self.config.verbalizer

    def collect_additional_tokens(self, verbalizer, **kwargs):
        additional_tokens = None
        return additional_tokens

    def input_is_flat(self, data):
        return isinstance(data[0], str)

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        super().load_vocabs(save_dir, filename)
        self.build_tokenizer(additional_tokens=self.config.verbalizer.get_special_tokens())
        self._transformer_config.tokenizer = self._tokenizer
        verbalizer = self.config.verbalizer
        if isinstance(verbalizer, ShiftReduceVerbalizer):
            self._transformer_config.ls = set(self._tokenizer.convert_tokens_to_ids(
                ['Ġ' + x for x in verbalizer.vocabs['labels'].idx_to_token if x.startswith('[')]))
            self._transformer_config.sh = self._tokenizer.convert_tokens_to_ids('ĠXX')
            self._transformer_config.rs = self._tokenizer.convert_tokens_to_ids('Ġ]')


def tokenize_prompt(
        sample,
        tokenizer: BartTokenizer,
        verbalizer: Verbalizer,
):
    prompt = sample.get('prompt', None)
    tokens = verbalizer.get_tokens(sample)
    tokens = [PTB_TOKEN_MAPPING.get(x, x) for x in tokens]

    if prompt is not None:
        sample['prompt_token_ids'] = verbalizer.tokenize_prompt(prompt, tokenizer)

    normalized_tokens, subtoken_to_token, text_token_ids = tokenize(tokens, tokenizer, ' ')
    subtoken_to_token.append(len(normalized_tokens))

    sample['text_token_ids'] = text_token_ids
    sample['subtoken_to_token'] = subtoken_to_token
    sample['text'] = tokenizer.decode(text_token_ids[:-1], clean_up_tokenization_spaces=False)
    sample['normalized_tokens'] = normalized_tokens
    return sample
