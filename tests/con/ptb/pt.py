# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-05-22 14:36
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.seq2seq.con.seq2seq_con import Seq2seqConstituencyParser
from elit.components.seq2seq.con.verbalizer import IsAPhraseVerbalizerVerbose
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING, PTB_DEV, PTB_TEST, PTB_TRAIN
from elit.utils.log_util import cprint
from tests import cdroot

cdroot()
scores = []
for i in range(3):
    save_dir = f'data/model/con/ptb/pt/{i}'
    cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
    con = Seq2seqConstituencyParser()
    con.fit(
        PTB_TRAIN,
        PTB_DEV,
        save_dir,
        verbalizer=IsAPhraseVerbalizerVerbose(
            label_to_phrase={
                'ADJP': 'an adjective phrase', 'ADVP': 'an adverb phrase', 'CONJP': 'a conjunction phrase',
                'FRAG': 'a fragment phrase', 'INTJ': 'an interjection', 'LST': 'a list marker',
                'NAC': 'a non-constituent', 'NP': 'a noun phrase', "NX": "a head noun phrase",
                'PP': 'a prepositional phrase', 'PRN': 'a parenthetical.', 'PRT': 'a particle',
                'QP': 'a quantifier phrase', 'RRC': 'a reduced relative clause',
                'UCP': 'an unlike coordinated phrase', 'VP': 'a verb phrase',
                'WHADJP': 'a wh-adjective phrase', 'WHADVP': 'a wh-adverb phrase',
                'WHNP': 'a wh-noun phrase',
                'WHPP': 'a wh-prepositional phrase', 'X': 'an unknown phrase',
                'S': 'a simple clause',
                "SBAR": "a subordinating clause",
                "SBARQ": "a wh-subordinating clause",
                "SINV": "an inverted clause", "SQ": "an interrogative clause",
            }),
        transform=NormalizeToken(PTB_TOKEN_MAPPING, 'token'),
        epochs=30,
        eval_after=28,
        gradient_accumulation=4,
        sampler_builder=SortingSamplerBuilder(batch_size=32, use_effective_tokens=True),
        max_seq_len=1024,
        max_prompt_len=1024,
    )
    con.load(save_dir)
    test_score = con.evaluate(PTB_TEST, save_dir, official=True)[-1]
    cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
    scores.append(test_score)

print(f'Scores on {len(scores)} runs:')
for metric in scores:
    print(metric)
