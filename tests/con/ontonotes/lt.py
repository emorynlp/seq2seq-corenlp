# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-29 22:10
from elit.common.dataset import SortingSamplerBuilder
from elit.components.seq2seq import Seq2seqConstituencyParser
from elit.components.seq2seq.con.verbalizer import BracketedVerbalizer
from elit.datasets.srl.ontonotes5.english import ONTONOTES5_CON_ENGLISH_TRAIN, ONTONOTES5_CON_ENGLISH_DEV, \
    ONTONOTES5_CON_ENGLISH_TEST
from elit.utils.log_util import cprint
from tests import cdroot

cdroot()
scores = []
for i in range(3):
    save_dir = f'data/model/con/ontonotes/lt/{i}'
    cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
    con = Seq2seqConstituencyParser()
    con.fit(
        ONTONOTES5_CON_ENGLISH_TRAIN,
        ONTONOTES5_CON_ENGLISH_DEV,
        save_dir,
        BracketedVerbalizer(flatten_pos=True),
        epochs=30,
        eval_after=25,
        gradient_accumulation=8,
        sampler_builder=SortingSamplerBuilder(batch_size=32, use_effective_tokens=True),
    )
    con.load(save_dir)
    test_score = con.evaluate(ONTONOTES5_CON_ENGLISH_TEST, save_dir, official=True)[-1]
    cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
    scores.append(test_score)

print(f'Scores on {len(scores)} runs:')
for metric in scores:
    print(metric)
