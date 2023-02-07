# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-28 17:33
import json
import logging
import os
import re
from collections import Counter
from typing import Union, List, Callable

import torch

from elit.metrics.chunking.sequence_labeling import get_entities
from hanlp_common.configurable import AutoConfigurable
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, BartTokenizer, AutoTokenizer, BertTokenizer, \
    BartTokenizerFast, BartForConditionalGeneration

from elit.common.dataset import SamplerBuilder, SortingSamplerBuilder, PadSequenceDataLoader
from elit.common.structure import History
from elit.common.torch_component import TorchComponent
from elit.common.vocab import Vocab
from elit.components.amr.seq2seq.dataset.tokenization_bart import PENMANBartTokenizer
from elit.components.amr.seq2seq.optim import RAdam
from elit.components.seq2seq.ner.constrained_decoding import FirstTokenProcessor, first_index_of
from elit.components.seq2seq.ner.dataset import JsonDocumentDataset
from elit.components.seq2seq.ner.prompt_ner import to_exclusive_offset, Verbalizer, SEP, BEGIN, END, \
    PairOfTagsVerbalizer, TagCountVerbalizer, TagVerbalizer
from elit.components.seq2seq.ner.transformers_ext import BartForConditionalGenerationExtended, \
    MBartForConditionalGenerationExtended, T5ForConditionalGenerationExtended, \
    OracleBartForConditionalGenerationExtended
from elit.layers.transformers.pt_imports import PretrainedConfig, AutoConfig_
from elit.layers.transformers.resource import get_model_mirror, get_tokenizer_mirror
from elit.layers.transformers.utils import build_optimizer_for_pretrained, pick_tensor_for_each_token
from elit.metrics.chunking.chunking_f1 import DetailedSpanF1
from elit.metrics.f1 import F1
from elit.utils.io_util import replace_ext, get_resource
from elit.utils.time_util import CountdownTimer
from hanlp_common.constant import IDX
from hanlp_common.io import load_json
from hanlp_common.util import merge_locals_kwargs, reorder, split_dict
from hanlp_trie.dictionary import TupleTrieDict, TrieDict

REG_POT = re.compile('<(.*?)>(.*?)</(.*?)>')


class Seq2seqNamedEntityRecognizer(TorchComponent):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._transformer_config: PretrainedConfig = None
        self._tokenizer: PENMANBartTokenizer = None
        self.model: BartForConditionalGenerationExtended = None
        self.gazetteer: TupleTrieDict = None

    def build_dataloader(self, data, batch_size,
                         gradient_accumulation=1,
                         shuffle=False,
                         sampler_builder: SamplerBuilder = None,
                         device=None,
                         logger: logging.Logger = None,
                         use_detokenization=False,
                         doc_level_offset=True,
                         transform=None,
                         max_seq_len=None,
                         oracle=False,
                         verbalizer=None,
                         **kwargs) -> DataLoader:
        if self.vocabs.mutable:
            if oracle:
                verbalizer = self.config.verbalizer
                self.vocabs['label'] = Vocab(list(verbalizer.label_to_phrase), pad_token=None, unk_token=None)
                self.vocabs.lock()

        dataset = self.build_dataset(data, not shuffle, **self.config)
        if transform:
            dataset.append_transform(transform)
        self.finalize_dataset(dataset, logger, **self.config)
        if isinstance(data, str):
            dataset.purge_cache()
            timer = CountdownTimer(len(dataset))
            max_num_tokens = 0
            for each in dataset:
                max_num_tokens = max(max_num_tokens, len(each['text_token_ids']))
                timer.log(f'Preprocessing and caching samples (longest sequence {max_num_tokens})'
                          f'[blink][yellow]...[/yellow][/blink]')
            if max_seq_len:
                dataset.prune(lambda x: len(x['text_token_ids']) > max_seq_len, logger)
                # if shuffle:
            #     with open('ner_train.jsonlines', 'w') as out:
            #         for each in dataset:
            #             out.write(json.dumps({'text': each['text'], 'prompt': each['prompt']}, ensure_ascii=False))
            #             out.write('\n')

        if not sampler_builder:
            sampler_builder = SortingSamplerBuilder(batch_max_tokens=500)
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
            logger: logging.Logger = None,
            use_detokenization=False,
            src_lang: str = None,
            tgt_lang: str = None,
            max_seq_len=None,
            gazetteer_verbalizer=None,
            doc_context=0,
            oracle=False,
            verbalizer=None,
            **kwargs
    ):
        dataset.append_transform(to_exclusive_offset)
        if isinstance(verbalizer, PairOfTagsVerbalizer):
            dataset.append_transform(lambda x: create_and_tokenize_pair_of_tags_prompt(
                x,
                verbalizer,
                self._tokenizer,
            ))
        elif isinstance(verbalizer, TagCountVerbalizer):
            dataset.append_transform(lambda x: create_and_tokenize_tag_count_prompt(
                x,
                verbalizer,
                self._tokenizer,
            ))
        elif isinstance(verbalizer, TagVerbalizer):
            dataset.append_transform(lambda x: create_and_tokenize_tag_prompt(
                x,
                verbalizer,
                self._tokenizer,
            ))
        else:
            dataset.append_transform(
                lambda x: create_and_tokenize_is_a_prompt(
                    x,
                    self._tokenizer, use_detokenization=use_detokenization,
                    src_lang=src_lang, tgt_lang=tgt_lang,
                    verbalizer=self.config.verbalizer,
                    delimiter_id=self._transformer_config.delimiter,
                    gazetteer=self.gazetteer,
                    gazetteer_verbalizer=gazetteer_verbalizer,
                    doc_context=doc_context,
                ),
            )
        if oracle:
            dataset.append_transform(
                lambda x: mask_oracle(x, label_token_ids_trie=self._transformer_config.label_token_ids_trie,
                                      vocab=self.vocabs['label']))

    def build_dataset(self, data, generate_idx, doc_level_offset=True, doc_context=0, **kwargs):
        dataset = JsonDocumentDataset(data, generate_idx=generate_idx, doc_level_offset=doc_level_offset,
                                      doc_context=doc_context)
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
        self._tokenizer = cls.from_pretrained(
            transformer,
            config=self._transformer_config,
        )
        if additional_tokens:
            self._tokenizer.add_tokens(additional_tokens)
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
                              save_every_epoch=False,
                              **kwargs):
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
            if epoch > eval_after or save_every_epoch:
                if dev_metric > best_metric or save_every_epoch:
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
            loss = self.feed_batch(batch, oracle)
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

    def feed_batch(self, batch, oracle=False, text_token_ids='text_token_ids', prompt_token_ids='prompt_token_ids',
                   **kwargs):
        input_ids, labels = batch[text_token_ids], batch.get(prompt_token_ids)
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        if labels is not None:
            if isinstance(self.model,
                          (BartForConditionalGenerationExtended, OracleBartForConditionalGenerationExtended,
                           BartForConditionalGeneration)) \
                    and self.model.config.name_or_path != 'fnlp/bart-large-chinese':
                decoder_input_ids = labels[:, :-1]
                labels = labels[:, 1:].contiguous()
            else:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
        else:
            decoder_input_ids = None
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                             labels=labels)
        if oracle:
            loss_fct = CrossEntropyLoss()
            lm_logits = outputs.logits
            mask = batch['teacher_forcing_mask']
            tf_loss = loss_fct(lm_logits[mask].view(-1, self._transformer_config.vocab_size), labels[mask].view(-1))
            # batch['oracle_subtoken_offset']
            oracle_subtoken_offset = batch['oracle_subtoken_offset']
            if oracle_subtoken_offset.numel():
                phrase_rep = pick_tensor_for_each_token(outputs.decoder_hidden_states[-1],
                                                        oracle_subtoken_offset, False)  # Use the first token
                label_reps = self.model.get_label_reps()
                sim_scores = torch.matmul(phrase_rep, label_reps.T)
                pred_label_ids = sim_scores.argmax(-1)
                gold_label_ids = batch['label_id']
                wrong_mask = pred_label_ids != gold_label_ids
                label_mask = batch['label_mask']
                wrong_mask &= label_mask
                if torch.any(wrong_mask):
                    sim_loss = loss_fct(sim_scores[wrong_mask], gold_label_ids[wrong_mask])
                else:
                    sim_loss = 0
            else:
                sim_loss = 0
            outputs.loss = tf_loss + sim_loss

        return outputs.loss

    @torch.no_grad()
    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, ratio_width=None,
                            logger=None, input=None, use_fast=False, save_dir=None, filename=None,
                            **kwargs):
        self.model.eval()
        timer = CountdownTimer(len(data))
        samples = []
        orders = []
        metric = DetailedSpanF1(do_confusion_matrix=True) if output else F1()
        for idx, batch in enumerate(data):
            entities_per_batch, pred_prompts = self.predict_ners(batch)
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
            timer.log(f'{metric}', ratio_percentage=False, logger=logger)

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
                        p, r, f = sample['score'].prf
                        out.write(json.dumps(
                            {
                                'sentences': [sample['token']],
                                'ner': [[(b, e - 1, l) for b, e, l in sample['pred_entity']]],
                                'gold': [[(b, e - 1, l) for b, e, l in sample['ner']]],
                                'pred_prompt': sample['pred_prompt'],
                                'prompt': sample['prompt'],
                                'p': p,
                                'r': r,
                                'f': f,
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

    def predict_ners(self, batch, beam_size=1):
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
        entities = []
        pred_prompts = []
        tokenizer = self._tokenizer
        verbalizer: Verbalizer = self.config.verbalizer
        text_token_ids = batch['text_token_ids'].tolist()
        text_token_ids = [x[1:l] for x, l in zip(text_token_ids, torch.sum(batch['text_token_ids'] != 1, 1).tolist())]
        all_special_ids = set(tokenizer.all_special_ids)
        all_special_ids.remove(tokenizer.unk_token_id)  # remove unk
        tgt_lang = self.config.get('tgt_lang', None)
        for i1 in range(0, len(tokens), beam_size):
            sample_index = i1 // beam_size
            entities_same_source = []
            entities.append(entities_same_source)
            for i2 in range(i1, i1 + beam_size):
                tokk = tokens[i2]
                prompt: str = tokenizer.decode(tokk, clean_up_tokenization_spaces=False)
                if isinstance(tokenizer, BertTokenizer):
                    prompt = prompt[len('[SEP]'):-len('[SEP]')]
                    prompt = prompt.replace(' ', '')
                if tgt_lang and prompt.startswith(tgt_lang):
                    prompt = prompt[len(tgt_lang):]
                prompt = prompt.replace('<pad>', '')
                if prompt.startswith('<s>'):
                    prompt = prompt[len('<s>'):]
                if prompt.endswith('</s>'):
                    prompt = prompt[:-len('</s>')]
                if prompt.startswith('</s>'):
                    prompt = prompt[len('</s>'):]
                original_prompt = None
                for prefix in [FirstTokenProcessor.ISA, FirstTokenProcessor.POT]:
                    if prompt.startswith(prefix):
                        original_prompt = prompt
                        prompt = prompt[len(prefix):]
                        break
                if i2 == i1:
                    if isinstance(verbalizer, TagVerbalizer):
                        pred_prompt = ' '.join(
                            tokenizer.convert_ids_to_tokens(x for x in tokk if x not in all_special_ids))
                    else:
                        pred_prompt = original_prompt if original_prompt is not None else prompt
                    pred_prompts.append(pred_prompt)
                normalized_tokens = batch['normalized_tokens'][sample_index]
                entities_per_seq = self._decode_enterties_per_seq(batch, normalized_tokens, prompt, sample_index,
                                                                  verbalizer, text_token_ids[sample_index],
                                                                  [x for x in tokk if x not in all_special_ids])
                entities_same_source.append([tuple(x) for x in entities_per_seq])
        return entities, pred_prompts

    def _decode_enterties_per_seq(self, batch, normalized_tokens, prompt, sample_index, verbalizer, input_ids,
                                  output_ids):
        tokenizer: BartTokenizer = self._tokenizer
        constrained_decoding = self.config.get('constrained_decoding', True)
        if isinstance(verbalizer, (PairOfTagsVerbalizer, TagVerbalizer)):
            if constrained_decoding:
                entities_per_seq = batch['_predictions'][sample_index]
            elif isinstance(verbalizer, TagVerbalizer):
                tags = tokenizer.convert_ids_to_tokens(output_ids)
                entities = get_entities(tags)
                entities_per_seq = [(x[1], x[2], x[0]) for x in entities]
            elif isinstance(verbalizer, PairOfTagsVerbalizer):
                offset = 0
                entities_per_seq = []
                for label, form, _ in REG_POT.findall(prompt):
                    _tokens = form.split()
                    b = first_index_of(normalized_tokens, _tokens, offset)
                    if b is not None:
                        entities_per_seq.append((b, b + len(_tokens), label))
                        offset = b + len(_tokens)
        else:
            entities_per_seq = verbalizer.prompt_to_entities(prompt, normalized_tokens)
        return entities_per_seq

    def _model_generate(self, batch, beam_size):
        input_ids = batch['text_token_ids']
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1024,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            decoder_start_token_id=(self._tokenizer.lang_code_to_id[self.config.tgt_lang] if isinstance(self.model,
                                                                                                        MBartForConditionalGenerationExtended) else None),
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
        if oracle:
            label_ids = [x[2:-1] for x in self._transformer_config.valid_label_token_ids]
            label_ids = PadSequenceDataLoader.pad_data(label_ids, 0, torch.long)
            model.register_buffer('label_ids', label_ids)
        return model

    def _get_model_cls(self, transformer: str, oracle=False):
        # return BartForConditionalGeneration
        constrained_decoding = self.config.get('constrained_decoding', True)
        if 't5-' in transformer:
            cls = T5ForConditionalGenerationExtended
        elif 'mbart-' in transformer:
            cls = MBartForConditionalGenerationExtended
        elif 'bart-' in transformer:
            if oracle:
                cls = OracleBartForConditionalGenerationExtended
            else:
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
        verbalizer: Verbalizer = self.config.verbalizer
        flat = self.input_is_flat(data)
        if flat:
            data = [data]
        dataloader = self.build_dataloader([{'token': x} for x in data], **self.config, device=self.device)
        orders = []
        results = []
        delimiter = verbalizer.delimiter if hasattr(verbalizer, 'delimiter') else ' '
        for batch in dataloader:
            entities, _ = self.predict_ners(batch)
            entities = [x[0] for x in entities]
            entities = [[(delimiter.join(x[b:e]), l, b, e,) for b, e, l in y] for x, y in zip(batch['token'], entities)]
            results.extend(entities)
            orders.extend(batch[IDX])
        results = reorder(results, orders)
        if flat:
            results = results[0]
        return results

    def fit(self, trn_data, dev_data, save_dir, verbalizer: AutoConfigurable,
            batch_size=32,
            epochs=30,
            transformer='facebook/bart-base',
            lr=5e-05,
            grad_norm=2.5,
            weight_decay=0.004,
            warmup_steps=1,
            dropout=0.25,
            attention_dropout=0.0,
            eval_after=0.5,
            gradient_accumulation=1,
            doc_level_offset=True,
            optimizer_name='radam',
            oracle=False,
            devices=None,
            logger=None,
            seed=None,
            finetune: Union[bool, str] = False,
            eval_trn=True,
            use_detokenization=False,
            transform=None,
            src_lang: str = None,
            tgt_lang: str = None,
            max_seq_len=None,
            gazetteer: str = None,
            doc_context=0,
            save_every_epoch=False,
            constrained_decoding=True,
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
        additional_tokens = self.collect_additional_tokens(**self.config)
        self.build_tokenizer(additional_tokens=additional_tokens)
        # valid_phrases = [f' is {x}' for x in phrase_to_label.keys()] + [';', ' ;', ' ; ']
        # valid_phrase_token_ids = self._tokenizer(valid_phrases, add_special_tokens=False).input_ids
        # self._transformer_config.valid_phrase_token_ids = set(sum(valid_phrase_token_ids, []))
        self._transformer_config.tokenizer = self._tokenizer
        self._transformer_config.forced_bos_token_id = None
        if not verbalizer:
            self.config.verbalizer = verbalizer = Verbalizer(
                label_to_phrase={
                    'PERSON': 'a person',
                    'NORP': 'a nationality',
                    'FAC': 'a facility',
                    'ORG': 'an organization',
                    'GPE': 'a geopolitical entity',
                    'LOC': 'a location',
                    'PRODUCT': 'a product',
                    'EVENT': 'an event',
                    'WORK_OF_ART': 'a work of art',
                    'LAW': 'a law',
                    'DATE': 'a date',
                    'TIME': 'a time',
                    'PERCENT': 'a percent',
                    'MONEY': 'a monetary value or a unit',
                    'QUANTITY': 'a quantity',
                    'ORDINAL': 'an ordinal',
                    'CARDINAL': 'a cardinal',
                    'LANGUAGE': 'a language',
                },
            )
        elif isinstance(verbalizer, Verbalizer):
            valid_labels = [f'{verbalizer.be_word}{x}{verbalizer.separator}' for x in
                            verbalizer.label_to_phrase.values()]
            if verbalizer.delimiter:
                self._transformer_config.valid_label_token_ids = self._tokenizer(valid_labels,
                                                                                 add_special_tokens=False).input_ids
            else:
                self._transformer_config.valid_label_token_ids = \
                    [tokenize([x], self._tokenizer, verbalizer.delimiter, add_special_tokens=False)[-1] for x in
                     valid_labels]
            self._transformer_config.phrase_trie = TrieDict(dict(zip(valid_labels, verbalizer.label_to_phrase)))
            self._transformer_config.separator_token_id = self._transformer_config.valid_label_token_ids[0][-1]
            self._transformer_config.delimiter = \
                Counter([x[0] for x in self._transformer_config.valid_label_token_ids]).most_common(1)[0][
                    0] if verbalizer.delimiter else None
            if gazetteer:
                self._tokenizer.add_tokens(SEP)
                self.gazetteer = TupleTrieDict.from_config(load_json(get_resource(gazetteer)))
                config.sep = self._tokenizer.convert_tokens_to_ids(SEP)
            else:
                config.sep = None
            if doc_context:
                self._tokenizer.add_tokens(BEGIN)
                self._tokenizer.add_tokens(END)
            if oracle:
                self._transformer_config.output_hidden_states = True
                self._transformer_config.label_token_ids_trie = TupleTrieDict(dict(
                    zip([tuple(x) for x in self._transformer_config.valid_label_token_ids],
                        self.config.verbalizer.label_to_phrase)))
        elif isinstance(verbalizer, PairOfTagsVerbalizer):
            self._prepare_pot_config(verbalizer)
        elif isinstance(verbalizer, TagCountVerbalizer):
            self._transformer_config.labels = self._tokenizer.convert_tokens_to_ids(
                [verbalizer.label_token(x) for x in verbalizer.labels])
            self._transformer_config.counts = self._tokenizer.convert_tokens_to_ids(verbalizer.counts)
        elif isinstance(verbalizer, TagVerbalizer):
            self._transformer_config.delimiter = None
        self._transformer_config.verbalizer = verbalizer

    def collect_additional_tokens(self, verbalizer, **kwargs):
        additional_tokens = verbalizer.get_additional_tokens() if isinstance(verbalizer, (
            PairOfTagsVerbalizer, TagCountVerbalizer, TagVerbalizer)) else None
        return additional_tokens

    def _prepare_pot_config(self, verbalizer):
        self._transformer_config.left_labels = []
        self._transformer_config.right_labels = []
        for l in verbalizer.labels:
            self._transformer_config.left_labels.append(
                self._tokenizer.convert_tokens_to_ids(verbalizer.left_label(l)))
            self._transformer_config.right_labels.append(
                self._tokenizer.convert_tokens_to_ids(verbalizer.right_label(l)))

    def input_is_flat(self, data):
        return isinstance(data[0], str)


def create_and_tokenize_is_a_prompt(
        sample,
        tokenizer: BartTokenizer,
        verbalizer: Verbalizer,
        delimiter_id: int,
        use_detokenization=False,
        src_lang: str = None,
        tgt_lang: str = None,
        gazetteer: TupleTrieDict = None,
        gazetteer_verbalizer: Verbalizer = None,
        doc_context=0,
):
    spaces = sample.get('spaces', None)
    ner = sample.get('ner', None)
    tokens = sample['token']
    delimiter = verbalizer.delimiter
    if ner is None:
        prompt = None
    else:
        if delimiter:
            prompt = verbalizer.separator.join(
                verbalizer.to_prompt(ner, tokens, spaces=spaces if use_detokenization else None))
            if ner:
                prompt += verbalizer.separator
            sample['prompt'] = prompt
        else:
            _p_tokens = verbalizer.to_prompt_tokens(ner, tokens)
            prompt = [x for x in sum(_p_tokens, []) if x]  # remove delimiter
            sample['prompt'] = ''.join(prompt)

    if src_lang or tgt_lang:
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        if prompt is not None:
            if delimiter:
                with tokenizer.as_target_tokenizer():
                    sample['prompt_token_ids'] = tokenizer(prompt).input_ids
            else:
                sample['prompt_token_ids'] = tokenize(prompt, tokenizer, delimiter)[-1] + [
                    tokenizer.lang_code_to_id[tgt_lang]]
    else:
        if prompt is not None:
            if delimiter:
                sample['prompt_token_ids'] = tokenizer(prompt).input_ids
            else:
                sample['prompt_token_ids'] = tokenize(prompt, tokenizer, delimiter)[-1]

    normalized_tokens, subtoken_to_token, text_token_ids = tokenize(tokens, tokenizer, delimiter)
    if src_lang:
        text_token_ids.append(tokenizer.lang_code_to_id[src_lang])

    subtoken_to_token.append(len(normalized_tokens))
    if gazetteer_verbalizer:
        # matches = gazetteer.tokenize(tokens)
        matches = gazetteer.parse(tokens)
        if matches:
            unpacked_matches = []
            for b, e, lf in matches:
                for l, f in lf.items():
                    unpacked_matches.append((b, e, l))
            hint_text = ''.join(sum(gazetteer_verbalizer.to_prompt_tokens(unpacked_matches, tokens), []))
            hint_token_ids = tokenizer(hint_text, add_special_tokens=False).input_ids
            if text_token_ids[-1] == tokenizer.eos_token_id:
                eos = text_token_ids[-1]
                text_token_ids = text_token_ids[:-1]
            else:
                eos = None
            text_token_ids = text_token_ids + [tokenizer.convert_tokens_to_ids(SEP)] + hint_token_ids
            if eos is not None:
                text_token_ids = text_token_ids + [eos]

    if doc_context:
        sample['text_token_ids_'] = text_token_ids  # backup for copying
        prev_tokens = sum(sample['left_context'], [])
        next_tokens = sum(sample['right_context'], [])
        if text_token_ids[0] == tokenizer.bos_token_id:
            bos = text_token_ids[0]
            text_token_ids = text_token_ids[1:]
        else:
            bos = None
        if text_token_ids[-1] == tokenizer.eos_token_id:
            eos = text_token_ids[-1]
            text_token_ids = text_token_ids[:-1]
        else:
            eos = None
        prev_token_ids = tokenizer(' '.join(prev_tokens), add_special_tokens=False).input_ids
        next_token_ids = tokenizer(' '.join(next_tokens), add_special_tokens=False).input_ids
        text_token_ids = prev_token_ids + [tokenizer.convert_tokens_to_ids(BEGIN)] + text_token_ids + [
            tokenizer.convert_tokens_to_ids(END)] + next_token_ids
        if bos is not None:
            text_token_ids = [bos] + text_token_ids
        if eos is not None:
            text_token_ids = text_token_ids + [eos]

    sample['text_token_ids'] = text_token_ids
    sample['subtoken_to_token'] = subtoken_to_token
    sample['text'] = tokenizer.decode(text_token_ids[:-1], clean_up_tokenization_spaces=False)
    sample['normalized_tokens'] = normalized_tokens
    return sample


def create_and_tokenize_pair_of_tags_prompt(sample: dict, verbalizer: PairOfTagsVerbalizer, tokenizer: BartTokenizer):
    ner = sample.get('', None)
    tokens = sample['token']
    if ner is None:
        prompt = None
    else:
        sample['prompt'] = prompt = ' ' + ' '.join(verbalizer.to_prompt_tokens(ner, tokens))
        prompt_input_ids = tokenizer(prompt).input_ids
        space_id = tokenizer(' ').input_ids[1]
        prompt_input_ids = list(filter(lambda x: x != space_id, prompt_input_ids))
        sample['prompt_token_ids'] = prompt_input_ids

    normalized_tokens, subtoken_to_token, text_token_ids = tokenize(tokens, tokenizer, ' ')
    subtoken_to_token.append(len(normalized_tokens))
    sample['text_token_ids'] = text_token_ids
    sample['subtoken_to_token'] = subtoken_to_token
    sample['text'] = tokenizer.decode(text_token_ids[:-1], clean_up_tokenization_spaces=False)
    sample['normalized_tokens'] = normalized_tokens
    return sample


def create_and_tokenize_tag_count_prompt(sample: dict, verbalizer: TagCountVerbalizer, tokenizer: BartTokenizer):
    ner = sample.get('', None)
    tokens = sample['token']
    if ner is None:
        prompt = None
    else:
        sample['prompt'] = prompt = ' ' + ' '.join(verbalizer.to_prompt_tokens(ner, tokens))
        prompt_input_ids = tokenizer(prompt).input_ids
        space_id = tokenizer(' ').input_ids[1]
        prompt_input_ids = list(filter(lambda x: x != space_id, prompt_input_ids))
        sample['prompt_token_ids'] = prompt_input_ids

    normalized_tokens, subtoken_to_token, text_token_ids = tokenize(tokens, tokenizer, ' ')
    subtoken_to_token.append(len(normalized_tokens))
    sample['text_token_ids'] = text_token_ids
    sample['subtoken_to_token'] = subtoken_to_token
    sample['text'] = tokenizer.decode(text_token_ids[:-1], clean_up_tokenization_spaces=False)
    sample['normalized_tokens'] = normalized_tokens
    return sample


def create_and_tokenize_tag_prompt(sample: dict, verbalizer: TagCountVerbalizer, tokenizer: BartTokenizer):
    ner = sample.get('', None)
    tokens = sample['token']
    if ner is None:
        prompt = None
    else:
        sample['prompt'] = prompt = verbalizer.to_prompt_tokens(ner, tokens)
        prompt_input_ids = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(prompt) + [tokenizer.eos_token_id]
        sample['prompt_token_ids'] = prompt_input_ids
        assert len(tokens) == len(prompt_input_ids) - 2

    normalized_tokens, subtoken_to_token, text_token_ids = tokenize(tokens, tokenizer, ' ')
    subtoken_to_token.append(len(normalized_tokens))
    sample['text_token_ids'] = text_token_ids
    sample['subtoken_to_token'] = subtoken_to_token
    sample['text'] = tokenizer.decode(text_token_ids[:-1], clean_up_tokenization_spaces=False)
    sample['normalized_tokens'] = normalized_tokens
    return sample


def tokenize(tokens, tokenizer: BartTokenizer, delimiter, add_special_tokens=True):
    text_token_ids = []
    subtoken_to_token = []
    normalized_tokens = []
    text_token_ids.append(tokenizer.bos_token_id)
    subtoken_to_token.append(-1)
    if isinstance(tokenizer, BartTokenizerFast) and delimiter:
        tokens = [delimiter + x for x in tokens]
    if tokens:
        encoded = tokenizer(tokens, add_special_tokens=False)
        if not delimiter:
            ids = tokenizer.convert_tokens_to_ids(tokens)
            mod_ids = tokenizer([f'_{x}' for x in tokens], add_special_tokens=False).input_ids
            mod_ids = [x[1:] for x in mod_ids] if all(len(x) >= 2 for x in mod_ids) else None
            prefix_id = tokenizer.convert_tokens_to_ids('_')
        if encoded.encodings:
            for i, each in enumerate(encoded.encodings):
                subtoken_ids = each.ids
                # leading space and real subtokens
                if not delimiter:
                    if ids[i] != tokenizer.unk_token_id:
                        subtoken_ids = [ids[i]]
                    elif len(each.offsets) >= 2 and each.offsets[0] == (0, 1) and each.offsets[1][0] == 0:
                        subtoken_ids = subtoken_ids[1:]  # remove delimiter
                    elif mod_ids and mod_ids[i][0] == prefix_id:
                        subtoken_ids = mod_ids[i][1:]

                text_token_ids.extend(subtoken_ids)
                subtoken_to_token.extend([i] * len(subtoken_ids))
                normalized = ''.join(tokenizer.convert_ids_to_tokens(subtoken_ids))
                if isinstance(tokenizer, BartTokenizerFast):
                    normalized = normalized.lstrip('Ġ')
                else:
                    normalized = normalized.lstrip('▁')
                normalized_tokens.append(normalized)
        else:
            for i, subtoken_ids in enumerate(encoded.data['input_ids']):  # Must be BERT
                text_token_ids.extend(subtoken_ids)
                subtoken_to_token.extend([i] * len(subtoken_ids))
                text = ''.join(tokenizer.convert_ids_to_tokens(subtoken_ids))
                text = text.replace('##', '')
                normalized_tokens.append(text)

    if add_special_tokens:
        if tokenizer.eos_token_id:  # BART
            text_token_ids.append(tokenizer.eos_token_id)
        else:  # BERT
            text_token_ids.append(tokenizer.sep_token_id)

    # if -1 in text_token_ids:
    #     cprint('[red]-1 in text_token_ids[/red]')
    #     exit(1)
    return normalized_tokens, subtoken_to_token, text_token_ids


def mask_oracle(sample: dict, label_token_ids_trie: TupleTrieDict, vocab: Vocab) -> dict:
    prompt_token_ids = sample.get('prompt_token_ids', None)
    if prompt_token_ids is None:
        return sample
    label_token_ids = prompt_token_ids[1:]
    mask = [True] * len(label_token_ids)
    subtoken_offsets = []
    label_ids = []
    for b, e, l in label_token_ids_trie.parse_longest(label_token_ids):
        b += 2  # is a
        e -= 1  # ;
        mask[b:e] = [False] * (e - b)
        subtoken_offsets.append(list(range(b, e)))
        label_ids.append(vocab[l])
    sample['teacher_forcing_mask'] = mask
    sample['oracle_subtoken_offset'] = subtoken_offsets
    sample['label_id'] = label_ids
    sample['label_mask'] = [True] * len(label_ids)
    return sample
