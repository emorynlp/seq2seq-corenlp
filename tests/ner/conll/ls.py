# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-10-22 17:19
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.seq2seq.ner.prompt_ner import TagVerbalizer
from elit.components.seq2seq.ner.seq2seq_ner import Seq2seqNamedEntityRecognizer
from elit.datasets.ner.conll03_json import CONLL03_EN_JSON_TRAIN, CONLL03_EN_JSON_TEST, CONLL03_EN_JSON_DEV
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING
from elit.utils.log_util import cprint
from tests import cdroot

cdroot()

save_dir = 'data/model/ner/conll/ls/0'
ner = Seq2seqNamedEntityRecognizer()
cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
ner.fit(
    CONLL03_EN_JSON_TRAIN,
    CONLL03_EN_JSON_DEV,
    save_dir,
    epochs=30,
    eval_after=25,
    transformer='facebook/bart-large',
    sampler_builder=SortingSamplerBuilder(batch_max_tokens=6000, use_effective_tokens=True),
    gradient_accumulation=1,
    fp16=False,
    transform=NormalizeToken(PTB_TOKEN_MAPPING, 'token'),
    verbalizer=TagVerbalizer(['LOC', 'PER', 'ORG', 'MISC']),
    _device_placeholder=True,
    save_every_epoch=False
)
ner.load(save_dir, constrained_decoding=True)
test_score = ner.evaluate(CONLL03_EN_JSON_TEST, save_dir)[-1]
cprint(f'Official score on testset: [red]{test_score.score:.2%}[/red]')
cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
