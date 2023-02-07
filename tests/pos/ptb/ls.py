# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-23 16:30
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.seq2seq.pos.seq2seq_pos import Seq2seqTagger
from elit.components.seq2seq.pos.verbalizer import TagVerbalizer
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING
from elit.utils.log_util import cprint
from tests import cdroot

cdroot()
scores = []
for i in range(3):
    save_dir = f'data/model/pos/ptb/ls/{i}'
    cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
    pos = Seq2seqTagger()
    pos.fit(
        'data/pos/wsj-pos/train.tsv',
        'data/pos/wsj-pos/dev.tsv',
        save_dir,
        verbalizer=TagVerbalizer(),
        transform=NormalizeToken(PTB_TOKEN_MAPPING, 'token'),
        epochs=30,
        eval_after=25,
        gradient_accumulation=1,
        sampler_builder=SortingSamplerBuilder(batch_max_tokens=6000, use_effective_tokens=True),
    )
    pos.load(save_dir, constrained_decoding=True)
    test_score = pos.evaluate('data/pos/wsj-pos/test.tsv', save_dir)[-1]
    scores.append(test_score)
    cprint(f'Model saved in [cyan]{save_dir}[/cyan]')

print(f'Scores on {len(scores)} runs:')
for metric in scores:
    print(metric)
