# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-26 19:11
import os
import re
from glob import glob
from urllib.error import HTTPError
from elit.utils.io_util import get_resource, path_from_url

_PTB_POS_HOME = 'https://catalog.ldc.upenn.edu/LDC99T42/LDC99T42.tgz#treebank_3/tagged/pos/wsj/'

PTB_POS_TRAIN = _PTB_POS_HOME + 'train.tsv'
'''Training set for PTB PoS tagging.'''
PTB_POS_DEV = _PTB_POS_HOME + 'dev.tsv'
'''Dev set for PTB PoS tagging.'''
PTB_POS_TEST = _PTB_POS_HOME + 'test.tsv'
'''Test set for PTB PoS tagging.'''

try:
    get_resource(_PTB_POS_HOME, verbose=False)
except HTTPError:
    raise FileNotFoundError(
        'The Penn Treebank is a copyright dataset owned by LDC which we cannot re-distribute. '
        f'Please apply for a licence from LDC (https://catalog.ldc.upenn.edu/LDC99T42) '
        f'then download it to {path_from_url(_PTB_POS_HOME)}'
    ) from None

_TOKEN_TAG = re.compile(r'\S+/\S+')


def _make_ptb_pos():
    home = get_resource(_PTB_POS_HOME)
    training = list(range(0, 18 + 1))
    development = list(range(19, 21 + 1))
    test = list(range(22, 24 + 1))
    for part, ids in zip(['train', 'dev', 'test'], [training, development, test]):
        out = f'{home}{part}.tsv'
        if os.path.isfile(out):
            continue
        with open(out, 'w') as out:
            dataset = []
            for fid in ids:
                for file in sorted(glob(f'{home}{fid:02d}/*.pos')):
                    with open(file) as src:
                        sent = []
                        for line in src:
                            line = line.strip()
                            if not line:
                                if sent:
                                    dataset.append(sent)
                                    sent = []
                            elif line.startswith('=========='):
                                continue
                            else:
                                for pair in _TOKEN_TAG.findall(line):
                                    pair = pair.rsplit('/', 1)
                                    sent.append(pair)
                    if sent:
                        dataset.append(sent)

            for sent in dataset:
                for token, pos in sent:
                    out.write(f'{token}\t{pos}\n')
                out.write('\n')


_make_ptb_pos()
