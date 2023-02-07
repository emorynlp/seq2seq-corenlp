# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-28 17:33
import logging
from typing import Union, List, Callable

import torch
from hanlp_common.constant import IDX
from hanlp_common.io import save_json
from hanlp_common.util import merge_locals_kwargs, reorder, split_dict
from hanlp_trie.dictionary import TupleTrieDict
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, BartTokenizer, AutoTokenizer, BertTokenizer, \
    BartForConditionalGeneration

from elit.common.dataset import SamplerBuilder, SortingSamplerBuilder, PadSequenceDataLoader
from elit.common.structure import History
from elit.common.torch_component import TorchComponent
from elit.common.vocab import Vocab
from elit.components.amr.seq2seq.optim import RAdam
from elit.components.seq2seq.pos.transformers_ext import BartForConditionalGenerationExtended
from elit.components.seq2seq.pos.verbalizer import Verbalizer, TokenTagVerbalizer, IsAVerbalizer
from elit.components.seq2seq.ner.seq2seq_ner import tokenize
from elit.datasets.ner.loaders.tsv import TSVTaggingDataset
from elit.layers.transformers.pt_imports import PretrainedConfig, AutoConfig_
from elit.layers.transformers.resource import get_model_mirror, get_tokenizer_mirror
from elit.layers.transformers.utils import build_optimizer_for_pretrained
from elit.metrics.accuracy import ScalarAccuracy
from elit.utils.io_util import replace_ext
from elit.utils.time_util import CountdownTimer


class Seq2seqTagger(TorchComponent):

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
                         verbalizer=None,
                         **kwargs) -> DataLoader:
        dataset = self.build_dataset(data, not shuffle, **self.config)
        if isinstance(data, str) and self.vocabs.mutable:
            self.vocabs['tag'] = Vocab(unk_token=None, pad_token=None)
            dataset.append_transform(self.vocabs)
            for each in dataset:
                pass
            self.vocabs.lock()
            self.vocabs.summary(logger)
            self.build_tokenizer(additional_tokens=self.vocabs['tag'].idx_to_token)
        if transform:
            dataset.append_transform(transform)
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
        dataset.append_transform(verbalizer)
        dataset.append_transform(lambda x: tokenize_prompt(
            x,
            self._tokenizer,
            verbalizer,
        ))

    def build_dataset(self, data, generate_idx, **kwargs):
        dataset = TSVTaggingDataset(data, generate_idx=generate_idx)
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
        if isinstance(verbalizer, IsAVerbalizer):
            valid_label_token_ids = self._tokenizer([f' is {x};' for x in verbalizer.tag_to_phrase.values()],
                                                    add_special_tokens=False).input_ids
            self._transformer_config.trie = TupleTrieDict(dict(
                zip([tuple(x) for x in valid_label_token_ids],
                    verbalizer.tag_to_phrase)))

        return self._tokenizer

    def build_optimizer(self, trn, lr, epochs, gradient_accumulation, warmup_steps, weight_decay,
                        optimizer_name='radam',
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
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps)
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
                dev_metric = -1
            elif epoch > eval_after:
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
                            **kwargs):
        self.model.eval()
        timer = CountdownTimer(len(data))
        samples = []
        orders = []
        metric = ScalarAccuracy()
        for idx, batch in enumerate(data):
            tags, pred_prompts = self.predict_tags(batch)
            tags = [x[0] for x in tags]
            for gp, gg, tokens in zip(tags, batch['tag'], batch['token']):
                metric(gp, gg)

            if output:
                batch['pred_tags'] = tags
                batch['pred_prompt'] = pred_prompts
                samples.extend(split_dict(batch))
                orders.extend(batch[IDX])
            timer.log(f'{metric}', ratio_percentage=False, logger=logger)

        if output:
            output = replace_ext(output, '.json')
            samples = reorder(samples, orders)
            items = []
            for sample in samples:
                _metric = ScalarAccuracy()
                _metric(sample['tag'], sample['pred_tags'])
                item = {
                    'token': sample['token'],
                    'gold': sample['tag'],
                    'pred': sample['pred_tags'],
                    'prompt': sample['prompt'],
                    'prompt_token_ids': sample['prompt_token_ids_'],
                    'pred_prompt': sample['pred_prompt'],
                    'score': float(_metric)
                }
                items.append(item)
            save_json(items, output)
        return metric

    def predict_tags(self, batch, beam_size=1):
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
        text_token_ids = batch['text_token_ids'].tolist()
        text_token_ids = [x[1:l] for x, l in zip(text_token_ids, torch.sum(batch['text_token_ids'] != 1, 1).tolist())]
        all_special_ids = set(tokenizer.all_special_ids)
        all_special_ids.remove(tokenizer.unk_token_id)  # remove unk
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
                pred_per_seq = self._decode_per_seq(batch, normalized_tokens, prompt, sample_index,
                                                    verbalizer, text_token_ids[sample_index],
                                                    [x for x in tokk if x not in all_special_ids], tokenizer)
                tags_same_source.append(pred_per_seq)
        return tags, pred_prompts

    def _decode_per_seq(self, batch, tokens, prompt, sample_index, verbalizer, input_ids, output_ids,
                        tokenizer: BartTokenizer):
        constrained_decoding = self.config.get('constrained_decoding', True)
        if not constrained_decoding:
            tags = []
            if isinstance(verbalizer, TokenTagVerbalizer):
                for t in output_ids:
                    if not input_ids:
                        break
                    if t == input_ids[0]:
                        input_ids = input_ids[1:]
                    else:
                        tags.append(tokenizer.convert_ids_to_tokens(t))
            elif isinstance(verbalizer, IsAVerbalizer):
                trie = self._transformer_config.trie
                tags = [x[-1] for x in trie.parse_longest(output_ids)]
            else:
                raise NotImplementedError()
            if len(tags) < len(tokens):
                tags += [None] * (len(tokens) - len(tags))
            elif len(tags) > len(tokens):
                tags = tags[:len(tokens)]
            return tags
        pred_per_seq = batch['_predictions'][sample_index]
        return pred_per_seq

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
        # with torch.no_grad():
        #     model.base_model.decoder.embed_tokens  = copy.deepcopy(model.base_model.decoder.embed_tokens)
        #     model.config.torchscript = True
        #     model._tie_or_clone_weights(model.base_model.decoder.embed_tokens, model.base_model.encoder.embed_tokens)
        #     model.config.torchscript = False
        return model

    def _get_model_cls(self, transformer: str, save_every_epoch=False):
        constrained_decoding = self.config.get('constrained_decoding', True)
        if save_every_epoch or not constrained_decoding:
            return BartForConditionalGeneration
        if 'bart-' in transformer:
            cls = BartForConditionalGenerationExtended
        else:
            raise NotImplemented(f'Unsupported transformer {transformer}')
        return cls

    def predict(self, data: Union[List[str], List[List[str]]], **kwargs):
        if not data:
            return []
        flat = self.input_is_flat(data)
        if flat:
            data = [data]
        dataloader = self.build_dataloader([{'token': x} for x in data], **self.config, device=self.device)
        orders = []
        results = []
        for batch in dataloader:
            tags, _ = self.predict_tags(batch)
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
            oracle=False,
            devices=None,
            logger=None,
            seed=None,
            finetune: Union[bool, str] = False,
            eval_trn=True,
            transform=None,
            max_seq_len=None,
            save_every_epoch=False,
            max_prompt_len=None,
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
        if 'tag' in self.vocabs:
            tags = self.vocabs['tag'].idx_to_token
            self.build_tokenizer(additional_tokens=tags)
        self._transformer_config.tokenizer = self._tokenizer


def tokenize_prompt(
        sample,
        tokenizer: BartTokenizer,
        verbalizer: Verbalizer,
):
    tag = sample.get('tag', None)
    tokens = sample['token']
    if tag is None:
        prompt = None
    else:
        prompt = sample['prompt']

    if prompt is not None:
        sample['prompt_token_ids'] = verbalizer.tokenize_prompt(prompt, tokenizer)
        sample['prompt_token_ids_'] = sample['prompt_token_ids']

    normalized_tokens, subtoken_to_token, text_token_ids = tokenize(tokens, tokenizer, ' ')
    subtoken_to_token.append(len(normalized_tokens))

    sample['text_token_ids'] = text_token_ids
    sample['subtoken_to_token'] = subtoken_to_token
    sample['text'] = tokenizer.decode(text_token_ids[:-1], clean_up_tokenization_spaces=False)
    sample['normalized_tokens'] = normalized_tokens
    return sample
