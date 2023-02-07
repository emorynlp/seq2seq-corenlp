# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-23 12:43
from elit.metrics.mtl import MetricDict


class SemanticRoleLabelingMetrics(MetricDict):
    @property
    def score(self):
        """Obtain the end-to-end score, which is the major metric for SRL.

        Returns:
            The end-to-end score.
        """
        return self['e2e'].score
