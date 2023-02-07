# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-29 22:10
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.seq2seq.dep.seq2seq_dep import Seq2seqDependencyParser
from elit.components.seq2seq.dep.verbalizer import ArcStandardVerbalizer
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING
from elit.datasets.srl.ontonotes5.english import ONTONOTES5_DEP_ENGLISH_TEST, ONTONOTES5_DEP_ENGLISH_DEV, \
    ONTONOTES5_DEP_ENGLISH_TRAIN
from elit.utils.log_util import cprint
from tests import cdroot

cdroot()
scores = []
for i in range(3):
    save_dir = f'data/model/dep/ontonotes/lt/{i}'
    cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
    dep = Seq2seqDependencyParser()
    dep.fit(
        ONTONOTES5_DEP_ENGLISH_TRAIN,
        ONTONOTES5_DEP_ENGLISH_DEV,
        save_dir,
        ArcStandardVerbalizer(lexical=True),
        transform=NormalizeToken(PTB_TOKEN_MAPPING, 'FORM'),
        epochs=30,
        eval_after=25,
        save_every_epoch=False,
        gradient_accumulation=4,
        sampler_builder=SortingSamplerBuilder(batch_size=32, use_effective_tokens=True),
    )
    dep.load(save_dir)
    test_score = dep.evaluate(ONTONOTES5_DEP_ENGLISH_TEST, save_dir)[-1]
    cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
    scores.append(test_score)

print(f'Scores on {len(scores)} runs:')
for metric in scores:
    print(metric)
