# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-12-13 19:49
import glob
import gzip
import os.path
from collections import defaultdict, Counter

from elit.datasets.ner.conll03_json import CONLL03_EN_JSON_TRAIN
from elit.datasets.ner.json_ner import JsonNERDataset
from elit.utils.io_util import get_resource
from elit.utils.time_util import CountdownTimer
from hanlp_common.io import save_json, load_json

GAZETTERS_UIUC_HOME = 'https://file.hankcs.com/corpus/gazetteers_UIUC.zip'
GAZETTERS_UIUC = GAZETTERS_UIUC_HOME + '#uiuic.json'
GAZETTERS_UIUC_FOR_CONLL03 = GAZETTERS_UIUC_HOME + '#conll03.json'

GAZETTERS_WIKIDATA_HOME = 'https://raw.githubusercontent.com/hltcoe/gazetteer-collection/master/gazetteers.tgz'
GAZETTERS_WIKIDATA = GAZETTERS_WIKIDATA_HOME + '#wikidata-eng.json'


def make_conll03_if_needed():
    home = get_resource(GAZETTERS_UIUC_HOME)
    json_path = os.path.join(home, GAZETTERS_UIUC_FOR_CONLL03[len(GAZETTERS_UIUC_HOME) + 1:])
    if os.path.isfile(json_path):
        return
    train = JsonNERDataset(CONLL03_EN_JSON_TRAIN)
    ne_conll = defaultdict(Counter)
    for sample in train:
        token = sample['token']
        for b, e, t in sample['ner']:
            ne_conll[' '.join(token[b:e + 1])][t] += 1
    ne_conll = dict((k, v.most_common(1)[0][0]) for k, v in ne_conll.items())

    gazetteer = defaultdict(Counter)
    files = sorted(glob.glob(f'{home}/*.gz'))
    timer = CountdownTimer(len(files))
    for g in files:
        tag = os.path.basename(g)[:-len('.gz')]

        with gzip.open(g, 'rt') as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                gazetteer[line][tag] += 1
        timer.log(f'Loaded {tag}')
    gazetteer = dict((k, v.most_common(1)[0][0]) for k, v in gazetteer.items())

    match_freq = defaultdict(Counter)
    for ne, tag in ne_conll.items():
        gtag = gazetteer.get(ne)
        if not gtag:
            print(ne)
        match_freq[tag][gtag] += 1
    match_freq = dict(
        (tag, dict((k, v / sum(freq.values())) for k, v in freq.most_common())) for tag, freq in match_freq.items())
    print()


def make_wikidata_if_needed():
    home = get_resource(GAZETTERS_WIKIDATA_HOME)
    json_path = os.path.join(home, GAZETTERS_WIKIDATA[len(GAZETTERS_WIKIDATA_HOME) + 1:])
    if os.path.isfile(json_path):
        return
    files = sorted(glob.glob(f'{home}/eng-*.txt'))
    gazetteer = defaultdict(Counter)
    timer = CountdownTimer(len(files))
    for g in files:
        tag = os.path.basename(g).split('-')[1]
        with open(g) as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                gazetteer[line][tag] += 1
        timer.log(f'Loaded {tag}')
    gazetteer = dict((k, dict(v.most_common())) for k, v in gazetteer.items())
    save_json(gazetteer, json_path)


def make_uiuc_if_needed():
    home = get_resource(GAZETTERS_UIUC_HOME)
    json_path = os.path.join(home, GAZETTERS_UIUC[len(GAZETTERS_UIUC_HOME) + 1:])
    if os.path.isfile(json_path):
        return
    files = sorted(glob.glob(f'{home}/*.gz'))
    gazetteer = defaultdict(Counter)
    timer = CountdownTimer(len(files))
    for g in files:
        tag = os.path.basename(g)[:-len('.gz')]
        with gzip.open(g, 'rt') as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                gazetteer[line][tag] += 1
        timer.log(f'Loaded {tag}')
    gazetteer = dict((k, v.most_common(1)[0][0]) for k, v in gazetteer.items())
    save_json(gazetteer, json_path)


make_wikidata_if_needed()
make_uiuc_if_needed()
