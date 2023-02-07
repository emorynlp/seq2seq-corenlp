# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-28 17:33
import json
import logging
import os
from typing import Union, List, Callable

import torch
from elit.components.seq2seq.dep.arc_standard import LA, RA, State

from hanlp_common.constant import IDX
from hanlp_common.util import merge_locals_kwargs, reorder, split_dict
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, BartTokenizer, AutoTokenizer, BertTokenizer, \
    BartForConditionalGeneration

from elit.common.dataset import SamplerBuilder, SortingSamplerBuilder, PadSequenceDataLoader
from elit.common.structure import History
from elit.common.torch_component import TorchComponent
from elit.components.amr.seq2seq.optim import RAdam
from elit.components.seq2seq.dep.transformers_ext import BartForConditionalGenerationExtended
from elit.components.seq2seq.dep.verbalizer import Verbalizer, ArcStandardVerbalizer, PromptVerbalizer, \
    make_index_on_stack
from elit.components.seq2seq.ner.seq2seq_ner import tokenize
from elit.datasets.parsing.loaders.conll_dataset import CoNLLParsingDataset
from elit.layers.transformers.pt_imports import PretrainedConfig, AutoConfig_
from elit.layers.transformers.resource import get_model_mirror, get_tokenizer_mirror
from elit.layers.transformers.utils import build_optimizer_for_pretrained
from elit.metrics.parsing.attachmentscore import AttachmentScore
from elit.utils.io_util import replace_ext
from elit.utils.time_util import CountdownTimer
from hanlp_trie.dictionary import TupleTrieDict, TrieDict


class Seq2seqDependencyParser(TorchComponent):

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
        if transform:
            dataset.append_transform(transform)
        dataset.append_transform(verbalizer)
        if isinstance(data, str) and shuffle:
            for each in dataset:
                pass
            self.build_tokenizer(additional_tokens=verbalizer.get_special_tokens())
            self.build_trie()

        self.finalize_dataset(dataset, verbalizer)
        if isinstance(data, str):
            dataset.purge_cache()
            timer = CountdownTimer(len(dataset))
            _max_num_tokens = 0
            _max_prompt_len = 0
            for each in dataset:
                _max_num_tokens = max(_max_num_tokens, len(each['text_token_ids']))
                prompt_token_ids = each.get('prompt_token_ids', None) or []
                if _max_prompt_len < len(prompt_token_ids):
                    _max_prompt_len = len(prompt_token_ids)
                    # print(each['prompt'])
                timer.log(f'Preprocessing and caching samples (longest sequence {_max_num_tokens}, '
                          f'longest prompt {_max_prompt_len}) '
                          f'[blink][yellow]...[/yellow][/blink]')
            # dataset.prune(lambda x: x['prompt'] is None, logger)
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
        dataset.append_transform(lambda x: tokenize_prompt(
            x,
            self._tokenizer,
            verbalizer,
        ))

    def build_dataset(self, data, generate_idx, **kwargs):
        dataset = CoNLLParsingDataset(data, generate_idx=generate_idx)
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
        metric = AttachmentScore()
        for idx, batch in enumerate(data):
            dep_rel, pred_prompts = self.predict_dep_rel(batch)
            dep_rel = [x[0] for x in dep_rel]
            for gp, gh, gr in zip(dep_rel, batch['HEAD'], batch['DEPREL']):
                gg = list(zip(gh, gr))
                ignore = [i for i, r in enumerate(gr) if r == 'punct']
                gp = [x for i, x in enumerate(gp) if i not in ignore]
                gg = [x for i, x in enumerate(gg) if i not in ignore]
                metric.update_lists(gp, gg)

            if output:
                batch['pred'] = dep_rel
                batch['pred_prompt'] = pred_prompts
                batch['uas'] = uas = []
                batch['las'] = las = []
                for gp, gh, gr in zip(dep_rel, batch['HEAD'], batch['DEPREL']):
                    _metric = AttachmentScore()
                    gg = list(zip(gh, gr))
                    ignore = [i for i, r in enumerate(gr) if r == 'punct']
                    gp = [x for i, x in enumerate(gp) if i not in ignore]
                    gg = [x for i, x in enumerate(gg) if i not in ignore]
                    _metric.update_lists(gp, gg)
                    uas.append(_metric.uas)
                    las.append(_metric.las)
                samples.extend(split_dict(batch))
                orders.extend(batch[IDX])
            timer.log(f'{metric}', ratio_percentage=False, logger=logger)

        if output:
            samples = reorder(samples, orders)
            if output == '.jsonlines':
                output = os.path.join(save_dir, f'pred-{filename}')
                with open(output, 'w') as out:
                    for sample in samples:
                        out.write(json.dumps(
                            {
                                'token': sample['FORM'],
                                'gold': list(zip(sample['HEAD'], sample['DEPREL'])),
                                'pred': sample['pred'],
                                'pred_prompt': sample['pred_prompt'],
                                'prompt': sample['prompt'],
                                'uas': sample['uas'],
                                'las': sample['las'],
                            }) + '\n')
            else:
                output = replace_ext(output, '.tsv')
                with open(output, 'w') as out, open(replace_ext(output, '-sorted.tsv'), 'w') as out_sorted:
                    samples = sorted(samples, key=lambda x: x['score'].prf[1])
                    self._write_samples(samples, out)

                    samples = sorted(samples, key=lambda x: tuple(x['token']))
                    self._write_samples(samples, out_sorted)
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

    def predict_dep_rel(self, batch, beam_size=1):
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
                    if isinstance(verbalizer, ArcStandardVerbalizer):
                        pred_prompt = ' '.join(
                            tokenizer.convert_ids_to_tokens(x for x in tokk if x not in all_special_ids))
                    else:
                        pred_prompt = original_prompt if original_prompt is not None else prompt
                    pred_prompts.append(pred_prompt)
                normalized_tokens = batch['normalized_tokens'][sample_index]
                y_tokens = tokenizer.convert_ids_to_tokens(tokk)
                if y_tokens[0] == '</s>':
                    y_tokens = y_tokens[1:]
                if '</s>' in y_tokens:
                    y_tokens = y_tokens[:y_tokens.index('</s>')]
                pred_per_seq = self._decode_enterties_per_seq(batch, normalized_tokens, prompt, y_tokens, sample_index,
                                                              verbalizer)
                tags_same_source.append(pred_per_seq)
        return tags, pred_prompts

    def _decode_enterties_per_seq(self, batch, normalized_tokens, prompt, prompt_tokens, sample_index,
                                  verbalizer: Verbalizer):
        constrained_decoding = self.config.get('constrained_decoding', True)
        if constrained_decoding:
            return verbalizer.decode_head_rel(batch, sample_index, prompt, prompt_tokens)
        else:
            if isinstance(verbalizer, ArcStandardVerbalizer):
                return verbalizer.decode_head_rel(batch, sample_index, prompt, prompt_tokens)
            elif isinstance(verbalizer, PromptVerbalizer):
                full_tokens = ['sentence'] + normalized_tokens  # with root being "sentence"
                system = State(len(normalized_tokens))

                def match_token(token):
                    for i in system.stack + system.buffer:
                        if full_tokens[i] == token:
                            return i

                offset = 0
                for b, e, transition in self._transformer_config.action_trie.parse_longest(prompt):
                    first = prompt[offset:b].strip()
                    colon = prompt.index(' ;', e)
                    if colon == -1:
                        break
                    if colon + 3 < len(prompt) and prompt[colon + 3] == ';':
                        colon = colon + 3
                    offset = colon + 3
                    second = prompt[e:colon].strip()

                    fid = match_token(first)
                    sid = match_token(second)
                    if fid is None or sid is None:
                        break

                    try:
                        make_index_on_stack(0, sid, system)
                        make_index_on_stack(1, fid, system)
                        system.apply(transition)
                    except:
                        break
                return list(zip(system.pred.heads[1:], system.pred.labels[1:]))
            else:
                raise NotImplementedError()

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
        constrained_decoding = self.config.get('constrained_decoding', True)
        if save_every_epoch:
            return BartForConditionalGeneration
        if 'bart-' in transformer:
            if constrained_decoding:
                cls = BartForConditionalGenerationExtended
            else:
                cls = BartForConditionalGeneration
        else:
            raise NotImplemented(f'Unsupported transformer {transformer}')
        return cls

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
            tags, _ = self.predict_dep_rel(batch)
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
        if isinstance(self.config.verbalizer, PromptVerbalizer):
            phrases = dict()
            for r, d in verbalizer.relations.items():
                phrases[f'is {d} of'] = f'{LA}-{r}'
                phrases[f'has {d}'] = f'{RA}-{r}'
            self._transformer_config.action_trie = TrieDict(phrases)

    def collect_additional_tokens(self, verbalizer, **kwargs):
        additional_tokens = None
        return additional_tokens

    def input_is_flat(self, data):
        return isinstance(data[0], str)

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        super().load_vocabs(save_dir, filename)
        self.build_tokenizer(additional_tokens=self.config.verbalizer.get_special_tokens())
        self.build_trie()
        self._transformer_config.tokenizer = self._tokenizer

    def build_trie(self):
        if isinstance(self.config.verbalizer, PromptVerbalizer):
            phrases = self.config.verbalizer.vocabs['action'].idx_to_token
            trie = TupleTrieDict()
            relations = self.config.verbalizer.relations
            rev_relations = dict((v, k) for k, v in relations.items())
            for p in phrases:
                tokens = tuple(p.split())
                if tokens[0] == 'is' and tokens[-1] == 'of':
                    action = 'LA-' + rev_relations[' '.join(tokens[1:-1])]
                else:
                    action = 'RA-' + rev_relations[' '.join(tokens[1:])]
                # trie[p] = action
                trie[tuple(tokenize(tokens, self._tokenizer, ' ')[-1][1:-1])] = action
            self._transformer_config.trie = trie


def tokenize_prompt(
        sample,
        tokenizer: BartTokenizer,
        verbalizer: Verbalizer,
):
    tokens = verbalizer.get_tokens(sample)
    normalized_tokens, subtoken_to_token, text_token_ids = tokenize(tokens, tokenizer, ' ')
    subtoken_to_token.append(len(normalized_tokens))

    prompt = sample.get('prompt', None)
    if prompt is not None:
        sample['prompt_token_ids'] = prompt_token_ids = verbalizer.tokenize_prompt(prompt, tokenizer)
        if isinstance(verbalizer, ArcStandardVerbalizer) and verbalizer.lexical:
            tokens_in_prompt = [x for x in tokenizer.convert_ids_to_tokens(prompt_token_ids) if
                                x not in verbalizer.vocabs['action']]
            if tokens_in_prompt != tokenizer.convert_ids_to_tokens(text_token_ids):
                print('Unable to reverse tokens back from prompts')

    sample['text_token_ids'] = text_token_ids
    sample['subtoken_to_token'] = subtoken_to_token
    sample['text'] = tokenizer.decode(text_token_ids[:-1], clean_up_tokenization_spaces=False)
    sample['normalized_tokens'] = normalized_tokens
    return sample
