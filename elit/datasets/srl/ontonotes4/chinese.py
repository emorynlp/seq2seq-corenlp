# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-11-26 16:07
import glob
import json
import re
from urllib.error import HTTPError
import os

from hanlp_common.io import load_json

from elit.datasets.srl.ontonotes4 import ONTONOTES4_TASKS_HOME, ONTONOTES4_HOME
from elit.datasets.srl.ontonotes5._utils import convert_jsonlines_to_IOBES
from elit.utils.io_util import get_resource, path_from_url
from elit.utils.log_util import cprint
from elit.utils.time_util import CountdownTimer

_ONTONOTES4_CHINESE_HOME = ONTONOTES4_HOME + 'files/data/chinese/annotations/'
_ONTONOTES4_CHINESE_TASKS_HOME = ONTONOTES4_TASKS_HOME + 'ner/chinese/'

ONTONOTES4_NER_4TYPES_CHINESE_TRAIN = _ONTONOTES4_CHINESE_TASKS_HOME + 'train.chinese.v4.ner.jsonlines'
ONTONOTES4_NER_4TYPES_CHINESE_DEV = _ONTONOTES4_CHINESE_TASKS_HOME + 'development.chinese.v4.ner.jsonlines'
ONTONOTES4_NER_4TYPES_CHINESE_TEST = _ONTONOTES4_CHINESE_TASKS_HOME + 'test.chinese.v4.ner.jsonlines'

ONTONOTES4_NER_4TYPES_CHINESE_TSV_TRAIN = _ONTONOTES4_CHINESE_TASKS_HOME + 'train.chinese.v4.ner.tsv'
ONTONOTES4_NER_4TYPES_CHINESE_TSV_DEV = _ONTONOTES4_CHINESE_TASKS_HOME + 'development.chinese.v4.ner.tsv'
ONTONOTES4_NER_4TYPES_CHINESE_TSV_TEST = _ONTONOTES4_CHINESE_TASKS_HOME + 'test.chinese.v4.ner.tsv'

ENAMEX = re.compile('<ENAMEX TYPE=\"(\w+?)\".*?>(.+?)</ENAMEX>')

try:
    get_resource(ONTONOTES4_HOME, verbose=False)
except HTTPError:
    intended_file_path = path_from_url(ONTONOTES4_HOME)
    cprint('Ontonotes 4.0 is a [red][bold]copyright[/bold][/red] dataset owned by LDC which we cannot re-distribute. '
           f'Please apply for a licence from LDC (https://catalog.ldc.upenn.edu/LDC2011T03) '
           f'then download it to {intended_file_path}')
    exit(1)


def load_ner(name_file):
    with open(name_file) as src:
        lines = src.readlines()
        lines = lines[1:-1]  # remove <DOC> and </DOC>
        sentences = []
        ner = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line == 'ＥＭＰＴＹ':
                continue
            tokens = []
            entities = []
            offset = 0
            for entity in ENAMEX.finditer(line):
                b, e = entity.span()
                if b > offset:
                    form = line[offset:b]
                    words = tokenize(form)
                    tokens.extend(words)
                tag = entity.group(1).strip()
                form = entity.group(2).strip()
                words = tokenize(form)
                # According to Named Entity Recognition with Bilingual Constraints:
                # In this paper, we selected the four most common named entity types,
                # i.e., PER (Person), LOC (Location), ORG (Organization) and GPE (Geo-Political Entities)
                if tag in ('PERSON', 'LOC', 'ORG', 'GPE'):
                    entities.append((len(tokens), len(tokens) + len(words) - 1, tag))
                tokens.extend(words)
                offset = e
            if offset < len(line):
                form = line[offset:]
                words = tokenize(form)
                tokens.extend(words)

            sentences.append(tokens)
            ner.append(entities)
            # print(tokens)
            # print([(tokens[b: e + 1], l) for b, e, l in entities])
        return {'doc_key': '/'.join(name_file.split(os.path.sep)[-4:])[:-len('.name')], 'sentences': sentences,
                'ner': ner}


def tokenize(form):
    words = [x.strip() for x in form.split()]
    words = [x for x in words if x]
    return words


def make_ner_jsonlines_if_needed():
    root = get_resource(ONTONOTES4_HOME)
    jsonfiles = dict(zip(['train', 'dev', 'test'], [os.path.join(root, x[len(ONTONOTES4_HOME):]) for x in
                                                    [ONTONOTES4_NER_4TYPES_CHINESE_TRAIN,
                                                     ONTONOTES4_NER_4TYPES_CHINESE_DEV,
                                                     ONTONOTES4_NER_4TYPES_CHINESE_TEST]]))
    if all(os.path.isfile(x) for x in jsonfiles.values()):
        return

    for file in jsonfiles.values():
        os.makedirs(os.path.dirname(file), exist_ok=True)

    zh_root = get_resource(_ONTONOTES4_CHINESE_HOME)
    name_files = glob.glob(f'{zh_root}/**/*.name', recursive=True)
    fps = dict((k, open(v, 'w', encoding='utf-8')) for k, v in jsonfiles.items())
    # glyce = load_json('data/ner/ontonotes4/splits.json')
    # glyce_splits = dict()
    # unsure = set()
    # for portion, sents in glyce.items():
    #     for sent in sents:
    #         sent = sent.replace('TYPE=\"PRODUCT\"E_OFF=\"1\">', '')
    #         exsisting = glyce_splits.get(sent, None)
    #         if exsisting and exsisting != portion:
    #             print(f'Conflict {sent} in both {exsisting} and {portion}')
    #             unsure.add(sent)
    #         glyce_splits[sent] = portion
    timer = CountdownTimer(len(name_files))
    for file in sorted(name_files):
        # According to Named Entity Recognition with Bilingual Constraints:
        # This corpus includes about 400 document pairs (chtb 0001-0325, ectb 1001-1078).
        # We used odd-numbered documents as development data and even-numbered documents as test data.
        # We used all other portions of the named entity annotated corpus as training data for the monolingual systems
        basename = os.path.basename(file)
        genre, remaining = basename.split('_')
        fid = int(remaining[:-len('.name')])
        if genre in ('chtb', 'ectb') and (1 <= fid <= 325 or 1001 <= fid <= 1078):
            if fid % 2:
                portion = 'dev'
            else:
                portion = 'test'
        else:
            portion = 'train'
        timer.log(f'Pre-processing {basename} to {portion} set [blink][yellow]...[/yellow][/blink]')
        doc = load_ner(file)
        # for sent in doc['sentences']:
        #     text = ''.join(sent)
        #     expected_portion = glyce_splits.get(text, None)
        #     # from difflib import SequenceMatcher
        #     # similar_keys = sorted(glyce_splits.keys(),
        #     #                       key=lambda k: SequenceMatcher(None, k, text).ratio(),
        #     #                       reverse=True)[:5]
        #     if expected_portion != portion and text not in unsure:
        #         print(text)
        fps[portion].write(json.dumps(doc, ensure_ascii=False) + '\n')
    for f in fps.values():
        f.close()


def make_ner_tsv_if_needed():
    root = get_resource(ONTONOTES4_HOME)
    tsv_files = [os.path.join(root, x[len(ONTONOTES4_HOME):]) for x in
                 [ONTONOTES4_NER_4TYPES_CHINESE_TSV_TRAIN, ONTONOTES4_NER_4TYPES_CHINESE_TSV_DEV,
                  ONTONOTES4_NER_4TYPES_CHINESE_TSV_TEST]]

    if all(os.path.isfile(x) for x in tsv_files):
        return

    for file in tsv_files:
        os.makedirs(os.path.dirname(file), exist_ok=True)

    for j, t in zip(
            [ONTONOTES4_NER_4TYPES_CHINESE_TRAIN, ONTONOTES4_NER_4TYPES_CHINESE_DEV,
             ONTONOTES4_NER_4TYPES_CHINESE_TEST],
            tsv_files):
        convert_jsonlines_to_IOBES(j, t, doc_level_offset=False, normalize_token=True)


make_ner_jsonlines_if_needed()
make_ner_tsv_if_needed()
