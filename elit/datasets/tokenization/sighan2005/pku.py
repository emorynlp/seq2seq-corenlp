# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:42
from elit.datasets.tokenization.sighan2005 import SIGHAN2005, make

SIGHAN2005_PKU_DICT = SIGHAN2005 + "#" + "gold/pku_training_words.utf8"
'''Dictionary built on trainings set.'''
SIGHAN2005_PKU_TRAIN_ALL = SIGHAN2005 + "#" + "training/pku_training.utf8"
'''Full training set.'''
SIGHAN2005_PKU_TRAIN = SIGHAN2005 + "#" + "training/pku_training_90.txt"
'''Training set (first 90% of the full official training set).'''
SIGHAN2005_PKU_DEV = SIGHAN2005 + "#" + "training/pku_training_10.txt"
'''Dev set (last 10% of full official training set).'''
SIGHAN2005_PKU_TEST_INPUT = SIGHAN2005 + "#" + "testing/pku_test.utf8"
'''Test input.'''
SIGHAN2005_PKU_TEST = SIGHAN2005 + "#" + "gold/pku_test_gold.utf8"
'''Test set.'''

make(SIGHAN2005_PKU_TRAIN)
