# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-23 16:30
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.seq2seq.pos.seq2seq_pos import Seq2seqTagger
from elit.components.seq2seq.pos.verbalizer import IsAVerbalizer
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING
from elit.utils.log_util import cprint
from tests import cdroot

cdroot()
scores = []
for i in range(3):
    save_dir = f'data/model/pos/ptb/pt_dec_lex/{i}'
    cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
    pos = Seq2seqTagger()
    mapper = {
        'NNP': 'a singular proper noun',
        ',': 'a comma',
        'CD': 'a cardinal number',
        'NNS': 'a plural noun',
        'JJ': 'an adjective',
        'MD': 'a modal',
        'VB': 'a verb',
        'DT': 'a determiner',
        'NN': 'a singular noun',
        'IN': 'a preposition or subordinating conjunction',
        '.': 'a period',
        'VBZ': 'a 3rd person singular present verb',
        'VBG': 'a gerund or present participle verb',
        'CC': 'a coordinating conjunction',
        'VBD': 'a past tense verb',
        'VBN': 'a past participle verb',
        'RB': 'an adverb',
        'TO': 'a to word',
        'PRP': 'a personal pronoun',
        'RBR': 'a comparative adverb',
        'WDT': 'a wh-determiner',
        'VBP': 'a non-3rd person singular present verb',
        'RP': 'a particle',
        'PRP$': 'a possessive pronoun',
        'JJS': 'a superlative adjective',
        'POS': 'a possessive ending',
        '``': 'a quotation mark',
        'EX': 'an existential there',
        "''": 'a back quotation mark',
        "WP": 'a wh-pronoun',
        ":": 'a colon',
        "JJR": 'a comparative adjective',
        "WRB": 'a wh-adverb',
        "$": 'a dollar sign',
        "NNPS": 'a plural proper noun',
        "WP$": 'a possessive wh-pronoun',
        "-LRB-": 'a left round bracket',
        "-RRB-": 'a right round bracket',
        "PDT": 'a predeterminer',
        "RBS": 'a superlative adverb',
        "FW": 'a foreign word',
        "UH": 'an interjection',
        "SYM": 'a symbol',
        "LS": 'a list item marker',
        "#": 'a pound sign',
    }
    for k in list(mapper):
        mapper[k] = f'a {k}'
    pos.fit(
        'data/pos/wsj-pos/train.tsv',
        'data/pos/wsj-pos/dev.tsv',
        save_dir,
        verbalizer=IsAVerbalizer(
            mapper,
            quotation=True,
        ),
        transform=NormalizeToken(PTB_TOKEN_MAPPING, 'token'),
        sampler_builder=SortingSamplerBuilder(batch_max_tokens=6000, use_effective_tokens=True),
        gradient_accumulation=8,
        max_prompt_len=1024,
        epochs=30,
        eval_after=25,
        save_every_epoch=True
    )
    pos.load(save_dir, constrained_decoding=False)
    score = pos.evaluate('data/pos/wsj-pos/test.tsv', save_dir)[-1]
    scores.append(score)
    cprint(f'Model saved in [cyan]{save_dir}[/cyan]')

print(f'Scores on {len(scores)} runs:')
for metric in scores:
    print(metric)
