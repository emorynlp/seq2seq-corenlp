# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-10-22 17:19
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.seq2seq.ner.prompt_ner import TagVerbalizer
from elit.components.seq2seq.ner.seq2seq_ner import Seq2seqNamedEntityRecognizer
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING
from elit.datasets.srl.ontonotes5.english import ONTONOTES5_NER_ENGLISH_DEV, ONTONOTES5_NER_ENGLISH_TRAIN, \
    ONTONOTES5_NER_ENGLISH_TEST
from elit.utils.log_util import cprint
from tests import cdroot

cdroot()

save_dir = 'data/model/ner/ontonotes/ls/0'
ner = Seq2seqNamedEntityRecognizer()
cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
ner.fit(
    ONTONOTES5_NER_ENGLISH_TRAIN,
    ONTONOTES5_NER_ENGLISH_DEV,
    save_dir,
    # lr=3e-5,
    epochs=30,
    eval_after=25,
    transformer='facebook/bart-large',
    sampler_builder=SortingSamplerBuilder(batch_max_tokens=6000, use_effective_tokens=True),
    gradient_accumulation=10,
    fp16=False,
    transform=NormalizeToken(PTB_TOKEN_MAPPING, 'token'),
    verbalizer=TagVerbalizer(
        ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG',
         'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']),
    _device_placeholder=True,
    save_every_epoch=False
    # max_seq_len=600
)
ner.load(save_dir, constrained_decoding=True)
test_score = ner.evaluate(ONTONOTES5_NER_ENGLISH_TEST, save_dir, output='.jsonlines')[-1]
cprint(f'Official score on testset: [red]{test_score.score:.2%}[/red]')
cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
