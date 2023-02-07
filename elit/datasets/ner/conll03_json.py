# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-12-09 00:32
import os

from elit.datasets.ner.conll03 import CONLL03_EN_TRAIN, CONLL03_EN_DEV, CONLL03_EN_TEST
from elit.utils.io_util import get_resource, replace_ext
from elit.utils.span_util import ner_tsv_to_jsonlines

CONLL03_EN_JSON_TRAIN = 'https://file.hankcs.com/corpus/conll03_en_iobes.zip#eng.train.jsonlines'
'''Training set of CoNLL03 (:cite:`tjong-kim-sang-de-meulder-2003-introduction`)'''
CONLL03_EN_JSON_DEV = 'https://file.hankcs.com/corpus/conll03_en_iobes.zip#eng.dev.jsonlines'
'''Dev set of CoNLL03 (:cite:`tjong-kim-sang-de-meulder-2003-introduction`)'''
CONLL03_EN_JSON_TEST = 'https://file.hankcs.com/corpus/conll03_en_iobes.zip#eng.test.jsonlines'
'''Test set of CoNLL03 (:cite:`tjong-kim-sang-de-meulder-2003-introduction`)'''


def make_jsonlines_if_needed():
    for tsv in [CONLL03_EN_TRAIN, CONLL03_EN_DEV, CONLL03_EN_TEST]:
        tsv = get_resource(tsv)
        if not os.path.isfile(replace_ext(tsv, '.jsonlines')):
            ner_tsv_to_jsonlines(tsv)


make_jsonlines_if_needed()
