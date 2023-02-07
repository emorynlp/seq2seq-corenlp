# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-08 16:34
from hanlp_common.structure import SerializableDict
from elit.datasets.glu.glue import STANFORD_SENTIMENT_TREEBANK_2_TRAIN, MICROSOFT_RESEARCH_PARAPHRASE_CORPUS_DEV
from elit.transform.table_tf import TableTransform


class StanfordSentimentTreebank2Transorm(TableTransform):
    pass


class MicrosoftResearchParaphraseCorpus(TableTransform):

    def __init__(self, config: SerializableDict = None, map_x=False, map_y=True, x_columns=(3, 4),
                 y_column=0, skip_header=True, delimiter='auto', **kwargs) -> None:
        super().__init__(config, map_x, map_y, x_columns, y_column, skip_header, delimiter, **kwargs)


def main():
    # _test_sst2()
    _test_mrpc()


def _test_sst2():
    transform = StanfordSentimentTreebank2Transorm()
    transform.fit(STANFORD_SENTIMENT_TREEBANK_2_TRAIN)
    transform.lock_vocabs()
    transform.label_vocab.summary()
    transform.build_config()
    dataset = transform.file_to_dataset(STANFORD_SENTIMENT_TREEBANK_2_TRAIN)
    for batch in dataset.take(1):
        print(batch)


def _test_mrpc():
    transform = MicrosoftResearchParaphraseCorpus()
    transform.fit(MICROSOFT_RESEARCH_PARAPHRASE_CORPUS_DEV)
    transform.lock_vocabs()
    transform.label_vocab.summary()
    transform.build_config()
    dataset = transform.file_to_dataset(MICROSOFT_RESEARCH_PARAPHRASE_CORPUS_DEV)
    for batch in dataset.take(1):
        print(batch)