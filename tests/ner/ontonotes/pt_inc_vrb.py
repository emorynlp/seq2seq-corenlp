# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-10-22 17:19
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.seq2seq.ner.prompt_ner import VerboseVerbalizer
from elit.components.seq2seq.ner.seq2seq_ner import Seq2seqNamedEntityRecognizer
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING
from elit.datasets.srl.ontonotes5.english import ONTONOTES5_NER_ENGLISH_TRAIN, ONTONOTES5_NER_ENGLISH_DEV, \
    ONTONOTES5_NER_ENGLISH_TEST
from elit.utils.log_util import cprint
from tests import cdroot
cdroot()
for i in range(3):
    save_dir = f'data/model/ner/ontonotes/pt_inc_vrb/{i}'
    ner = Seq2seqNamedEntityRecognizer()
    cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
    ner.fit(
        ONTONOTES5_NER_ENGLISH_TRAIN,
        ONTONOTES5_NER_ENGLISH_DEV,
        save_dir,
        epochs=30,
        eval_after=25,
        transformer='facebook/bart-large',
        sampler_builder=SortingSamplerBuilder(batch_max_tokens=6000, use_effective_tokens=True),
        gradient_accumulation=6,
        fp16=False,
        transform=NormalizeToken(PTB_TOKEN_MAPPING, 'token'),
        verbalizer=VerboseVerbalizer(
            label_to_phrase={
                'PERSON': 'a person',
                'NORP': 'an ethnicity',
                'FAC': 'a facility',
                'ORG': 'an organization',
                'GPE': 'a geopolitical entity',
                'LOC': 'a location',
                'PRODUCT': 'a product',
                'EVENT': 'an event',
                'WORK_OF_ART': 'an art work',
                'LAW': 'a law',
                'DATE': 'a date',
                'TIME': 'a time',
                'PERCENT': 'a percentage',
                'MONEY': 'a monetary value',
                'QUANTITY': 'a quantity',
                'ORDINAL': 'an ordinal',
                'CARDINAL': 'a cardinal',
                'LANGUAGE': 'a language',
            }
        ),
        _device_placeholder=True,
        constrained_decoding=False,
    )
    ner.load(save_dir)
    test_score = ner.evaluate(ONTONOTES5_NER_ENGLISH_TEST, save_dir)[-1]
    cprint(f'Official score on testset: [red]{test_score.score:.2%}[/red]')
    cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
