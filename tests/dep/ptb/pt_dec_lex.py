# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-29 22:10
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.seq2seq.dep.seq2seq_dep import Seq2seqDependencyParser
from elit.components.seq2seq.dep.verbalizer import PromptVerbalizer
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING, PTB_SD330_DEV, PTB_SD330_TEST, PTB_SD330_TRAIN
from elit.utils.log_util import cprint
from tests import cdroot
cdroot()
scores = []
for i in range(3):
    save_dir = f'data/model/dep/ptb/pt_dec_lex/{i}'
    cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
    dep = Seq2seqDependencyParser()
    mapper = {'acomp': 'an adjectival complement',
              'advcl': 'an adverbial clause',
              'advmod': 'an adverbial',
              'amod': 'an adjective',
              'appos': 'an apposition',
              'aux': 'an auxiliary',
              'auxpass': 'a passive auxiliary',
              'cc': 'a coordination',
              'ccomp': 'a clausal complement',
              'conj': 'a conjunct',
              'cop': 'a copula',
              'csubj': 'a clausal subject',
              'csubjpass': 'a clausal passive subject',
              'dep': 'a dependent',
              'det': 'a determiner',
              'discourse': 'a discourse',
              'dobj': 'a direct object',
              'expl': 'an expletive',
              'infmod': 'an infinitival modifier',
              'iobj': 'an indirect object',
              'mark': 'a marker',
              'mwe': 'a multi-word expression',
              'neg': 'a negation modifier',
              'nn': 'a compound noun',
              'npadvmod': 'an adverbial noun phrase',
              'nsubj': 'a subject',
              'nsubjpass': 'a passive subject',
              'num': 'a numeral',
              'number': 'a compound number',
              'parataxis': 'a parataxis',
              'partmod': 'a participle',
              'pcomp': 'a prepositional complement',
              'pobj': 'a preposition object',
              'poss': 'a possession',
              'possessive': 'a possessive',
              'preconj': 'a preconjunct',
              'predet': 'a predeterminer',
              'prep': 'a preposition',
              'prt': 'a particle',
              'punct': 'a punctuation',
              'quantmod': 'a quantifier phrase',
              'rcmod': 'a relative clause',
              'root': 'a root',
              'tmod': 'a time',
              'xcomp': 'an open clausal complement'}
    for k in list(mapper):
        mapper[k] = f'a {k}'
    dep.fit(
        PTB_SD330_TRAIN,
        PTB_SD330_DEV,
        save_dir,
        PromptVerbalizer(
            mapper,
            is_a_tag=True
        ),
        transform=NormalizeToken(PTB_TOKEN_MAPPING, 'FORM'),
        epochs=30,
        eval_after=25,
        save_every_epoch=True,
        gradient_accumulation=4,
        sampler_builder=SortingSamplerBuilder(batch_size=32, use_effective_tokens=True),
        max_prompt_len=1024,
    )
    dep.load(save_dir)
    test_score = dep.evaluate(PTB_SD330_TEST, save_dir)[-1]
    cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
    scores.append(test_score)
print(f'Scores on {len(scores)} runs:')
for metric in scores:
    print(metric)
