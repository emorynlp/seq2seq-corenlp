# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-29 22:10
from elit.common.dataset import SortingSamplerBuilder
from elit.common.transform import NormalizeToken
from elit.components.seq2seq.dep.seq2seq_dep import Seq2seqDependencyParser
from elit.components.seq2seq.dep.verbalizer import PromptVerbalizer
from elit.datasets.parsing.ptb import PTB_TOKEN_MAPPING
from elit.datasets.srl.ontonotes5.english import ONTONOTES5_DEP_ENGLISH_TRAIN, ONTONOTES5_DEP_ENGLISH_DEV, \
    ONTONOTES5_DEP_ENGLISH_TEST
from elit.utils.log_util import cprint
from tests import cdroot

cdroot()
scores = []
for i in range(3):
    save_dir = f'data/model/dep/ontonotes/pt_dec_lex/{i}'
    cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
    dep = Seq2seqDependencyParser()
    mapper = {'acomp': 'an adjectival complement', 'advcl': 'an adverbial clause modifier',
              'advmod': 'an adverbial modifier', 'amod': 'an adjectival modifier',
              'appos': 'an appositional modifier',
              'aux': 'an auxiliary', 'auxpass': 'a passive auxiliary', 'cc': 'a coordination',
              'ccomp': 'a clausal complement', 'conj': 'a conjunct', 'cop': 'a copula', 'csubj': 'a clausal subject',
              'csubjpass': 'a clausal passive subject', 'dep': 'a dependent', 'det': 'a determiner',
              'discourse': 'a discourse element', 'dobj': 'a direct object', 'expl': 'an expletive',
              'iobj': 'an indirect object', 'mark': 'a marker', 'mwe': 'a multi-word expression',
              'neg': 'a negation modifier', 'nn': 'a noun compound modifier',
              'npadvmod': 'a noun phrase as adverbial modifier', 'nsubj': 'a nominal subject',
              'nsubjpass': 'a passive nominal subject', 'num': 'a numeric modifier',
              'number': 'an element of compound number', 'parataxis': 'a parataxis',
              'pcomp': 'a prepositional complement', 'pobj': 'an object of a preposition',
              'poss': 'a possession modifier', 'possessive': 'a possessive modifier', 'preconj': 'a preconjunct',
              'predet': 'a predeterminer', 'prep': 'a prepositional modifier', 'prt': 'a phrasal verb particle',
              'punct': 'a punctuation', 'quantmod': 'a quantifier phrase modifier',
              'rcmod': 'a relative clause modifier', 'root': 'a root', 'tmod': 'a temporal modifier',
              'xcomp': 'an open clausal complement', 'infmod': 'an infinitival modifier',
              'partmod': 'a participial modifier'}
    for k in list(mapper):
        mapper[k] = f'a {k}'
    dep.fit(
        ONTONOTES5_DEP_ENGLISH_TRAIN,
        ONTONOTES5_DEP_ENGLISH_DEV,
        save_dir,
        PromptVerbalizer(
            mapper,
            is_a_tag=True,
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
    test_score = dep.evaluate(ONTONOTES5_DEP_ENGLISH_TEST, save_dir)[-1]
    cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
    scores.append(test_score)

print(f'Scores on {len(scores)} runs:')
for metric in scores:
    print(metric)
