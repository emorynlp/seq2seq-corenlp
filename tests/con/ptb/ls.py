# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-29 22:10
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.seq2seq import Seq2seqConstituencyParser
from elit.components.seq2seq.con.verbalizer import ShiftReduceVerbalizer
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING, PTB_DEV, PTB_TEST, PTB_TRAIN
from elit.utils.log_util import cprint
from tests import cdroot

cdroot()
scores = []
for i in range(3):
    save_dir = f'data/model/con/ptb/ls/{i}'
    cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
    con = Seq2seqConstituencyParser()
    con.fit(
        PTB_TRAIN,
        PTB_DEV,
        save_dir,
        ShiftReduceVerbalizer(flatten_pos=True, anonymize_token=True),
        transform=NormalizeToken(PTB_TOKEN_MAPPING, 'token'),
        epochs=30,
        eval_after=28,
        save_every_epoch=False,
        gradient_accumulation=2,
        sampler_builder=SortingSamplerBuilder(batch_size=32, use_effective_tokens=True),
    )
    con.load(save_dir)
    test_score = con.evaluate(PTB_TEST, save_dir, official=True)[-1]
    cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
    scores.append(test_score)

print(f'Scores on {len(scores)} runs:')
for metric in scores:
    print(metric)
