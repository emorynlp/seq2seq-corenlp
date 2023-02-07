# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-10-24 22:47
import json
import os
from typing import Union, List, Callable

from elit.common.dataset import TransformableDataset
from elit.utils.io_util import TimingFileIterator


class JsonDocumentDataset(TransformableDataset):

    def __init__(self, data: Union[str, List], transform: Union[Callable, List] = None, cache=None,
                 generate_idx=None, doc_level_offset=True, tagset=None, doc_context=0) -> None:
        """A dataset for ``.jsonlines`` format NER corpora.

        Args:
            data: The local or remote path to a dataset, or a list of samples where each sample is a dict.
            transform: Predefined transform(s).
            cache: ``True`` to enable caching, so that transforms won't be called twice.
            generate_idx: Create a :const:`~hanlp_common.constants.IDX` field for each sample to store its order in dataset. Useful for prediction when
                samples are re-ordered by a sampler.
            doc_level_offset: ``True`` to indicate the offsets in ``jsonlines`` are of document level.
            tagset: Optional tagset to prune entities outside of this tagset from datasets.
        """
        self.doc_context = doc_context
        self.tagset = tagset
        self.doc_level_offset = doc_level_offset
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath: str):
        """Load ``.jsonlines`` NER corpus. Samples of this corpus can be found using the following scripts.

        .. highlight:: python
        .. code-block:: python

            import json
            from hanlp_common.document import Document
            from elit.datasets.srl.ontonotes5.chinese import ONTONOTES5_CONLL12_CHINESE_DEV
            from elit.utils.io_util import get_resource

            with open(get_resource(ONTONOTES5_CONLL12_CHINESE_DEV)) as src:
                for line in src:
                    doc = json.loads(line)
                    print(Document(doc))
                    break

        Args:
            filepath: ``.jsonlines`` NER corpus.
        """
        filename = os.path.basename(filepath)
        reader = TimingFileIterator(filepath)
        num_docs, num_sentences = 0, 0
        for line in reader:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            num_docs += 1
            num_tokens_in_doc = 0
            for i, (sentence, ner) in enumerate(zip(doc['sentences'], doc['ner'])):
                if self.doc_level_offset:
                    ner = [(x[0] - num_tokens_in_doc, x[1] - num_tokens_in_doc, x[2]) for x in ner]
                else:
                    ner = [(x[0], x[1], x[2]) for x in ner]
                if self.tagset:
                    ner = [x for x in ner if x[2] in self.tagset]
                    if isinstance(self.tagset, dict):
                        ner = [(x[0], x[1], self.tagset[x[2]]) for x in ner]
                deduplicated_ner = []
                be_set = set()
                for b, e, l in ner:
                    be = (b, e)
                    if be in be_set:
                        continue
                    be_set.add(be)
                    deduplicated_ner.append((b, e, l))
                sample = {
                    'token': sentence,
                    'ner': deduplicated_ner
                }
                if self.doc_context:
                    sample['left_context'] = doc['sentences'][i - self.doc_context:i]
                    sample['right_context'] = doc['sentences'][i + 1:i + 1 + self.doc_context]
                spaces = doc.get('spaces', None)
                if spaces:
                    sample['spaces'] = spaces[i]
                yield sample
                num_sentences += 1
                num_tokens_in_doc += len(sentence)
            reader.log(
                f'{filename} {num_docs} documents, {num_sentences} sentences [blink][yellow]...[/yellow][/blink]')
        reader.erase()
