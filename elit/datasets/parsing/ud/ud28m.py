# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-21 20:39
import os

from elit.datasets.parsing.ud import concat_treebanks
from elit.datasets.parsing.ud.ud28 import _UD_28_HOME

_UD_28_MULTILINGUAL_HOME = concat_treebanks(_UD_28_HOME, '2.8')
UD_28_MULTILINGUAL_TRAIN = os.path.join(_UD_28_MULTILINGUAL_HOME, 'train.conllu')
"Training set of multilingual UD_28 obtained by concatenating all training sets."
UD_28_MULTILINGUAL_DEV = os.path.join(_UD_28_MULTILINGUAL_HOME, 'dev.conllu')
"Dev set of multilingual UD_28 obtained by concatenating all dev sets."
UD_28_MULTILINGUAL_TEST = os.path.join(_UD_28_MULTILINGUAL_HOME, 'test.conllu')
"Test set of multilingual UD_28 obtained by concatenating all test sets."
