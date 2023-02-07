# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 11:25
import os
import warnings
if os.environ.get('USE_TF', None) is None:
    os.environ["USE_TF"] = 'NO'  # saves time loading transformers
if os.environ.get('TOKENIZERS_PARALLELISM', None) is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import BertTokenizer, BertConfig, PretrainedConfig, \
    AutoConfig, AutoTokenizer, PreTrainedTokenizer, BertTokenizerFast, AlbertConfig, BertModel, AutoModel, \
    PreTrainedModel, get_linear_schedule_with_warmup, AdamW, AutoModelForSequenceClassification, \
    AutoModelForTokenClassification, optimization, BartModel
from elit.layers.transformers.longformer.long_models import LongBertModel, LongElectraModel
from elit.layers.transformers.resource import get_tokenizer_mirror, get_model_mirror


class AutoModel_(AutoModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, training=True, **kwargs):
        # noinspection PyMethodFirstArgAssignment
        cls = AutoModel
        if pretrained_model_name_or_path == 'ValkyriaLenneth/longformer_zh' or \
                pretrained_model_name_or_path == 'schen/longformer-chinese-base-4096':
            cls = LongBertModel
        elif pretrained_model_name_or_path == 'ernie-gram-zh-4096':
            pretrained_model_name_or_path = 'peterchou/ernie-gram'
            cls = LongBertModel
            kwargs['convert_to_length'] = 4096
        elif pretrained_model_name_or_path == 'mengzi-bert-base-4096':
            pretrained_model_name_or_path = 'Langboat/mengzi-bert-base'
            cls = LongBertModel
            kwargs['convert_to_length'] = 4096
        elif pretrained_model_name_or_path == 'chinese-electra-180g-base-discriminator-4096':
            pretrained_model_name_or_path = 'hfl/chinese-electra-180g-base-discriminator'
            cls = LongElectraModel
            kwargs['convert_to_length'] = 4096
        elif pretrained_model_name_or_path == 'chinese-electra-180g-large-discriminator-4096':
            pretrained_model_name_or_path = 'hfl/chinese-electra-180g-large-discriminator'
            cls = LongElectraModel
            kwargs['convert_to_length'] = 4096
        pretrained_model_name_or_path = get_model_mirror(pretrained_model_name_or_path)
        if training:
            return cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        else:
            if isinstance(pretrained_model_name_or_path, str):
                pretrained_model_name_or_path = get_tokenizer_mirror(pretrained_model_name_or_path)
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
                # config.training = False  # Mark this model is for loading
                # Pass additional config to longformer
                for k, v in kwargs.items():
                    if not hasattr(config, k):
                        setattr(config, k, v)
                return cls.from_config(config)
            else:
                assert not kwargs
                return cls.from_config(pretrained_model_name_or_path)


class AutoConfig_(AutoConfig):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        pretrained_model_name_or_path = get_tokenizer_mirror(pretrained_model_name_or_path)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


class AutoTokenizer_(AutoTokenizer):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, use_fast=True,
                        do_basic_tokenize=True) -> PreTrainedTokenizer:
        if isinstance(pretrained_model_name_or_path, str):
            transformer = pretrained_model_name_or_path
        else:
            transformer = pretrained_model_name_or_path.transformer
        additional_config = dict()
        if transformer.startswith('voidful/albert_chinese_') or transformer.startswith('uer/albert'):
            cls = BertTokenizer
        elif transformer == 'cl-tohoku/bert-base-japanese-char':
            # Since it's char level model, it's OK to use char level tok instead of fugashi
            # from elit.utils.lang.ja.bert_tok import BertJapaneseTokenizerFast
            # cls = BertJapaneseTokenizerFast
            from transformers import BertJapaneseTokenizer
            cls = BertJapaneseTokenizer
            # from transformers import BertTokenizerFast
            # cls = BertTokenizerFast
            additional_config['word_tokenizer_type'] = 'basic'
        elif transformer == "Langboat/mengzi-bert-base":
            cls = BertTokenizerFast if use_fast else BertTokenizer
        elif transformer == 'ernie-gram-zh-4096':
            transformer = 'peterchou/ernie-gram'
        elif transformer == 'mengzi-bert-base-4096':
            transformer = 'Langboat/mengzi-bert-base'
        elif transformer == 'chinese-electra-180g-base-discriminator-4096':
            transformer = 'hfl/chinese-electra-180g-base-discriminator'
        elif transformer == 'chinese-electra-180g-large-discriminator-4096':
            transformer = 'hfl/chinese-electra-180g-large-discriminator'
        elif transformer == "Langboat/mengzi-bert-base":
            cls = BertTokenizerFast if use_fast else BertTokenizer
        else:
            cls = AutoTokenizer
        if use_fast and not do_basic_tokenize:
            warnings.warn('`do_basic_tokenize=False` might not work when `use_fast=True`')
        tokenizer = cls.from_pretrained(get_tokenizer_mirror(transformer), use_fast=use_fast,
                                        do_basic_tokenize=do_basic_tokenize,
                                        **additional_config)
        tokenizer.name_or_path = transformer
        return tokenizer
