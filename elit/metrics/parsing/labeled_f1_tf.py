# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-27 21:42
import tensorflow as tf


class LabeledF1TF(object):

    def __init__(self):
        super(LabeledF1TF, self).__init__()

        self.sum_gold_arcs_wo_punc = 0.0
        self.sum_pred_arcs_wo_punc = 0.0
        self.correct_arcs_wo_punc = 0.0
        self.correct_rels_wo_punc = 0.0

    def __repr__(self):
        return f"UF: {self.uf:6.2%} LF: {self.lf:6.2%}"

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        mask = mask.unsqueeze(-1).expand_as(arc_preds)
        mask = mask & mask.transpose(1, 2)

        mask_gold = mask & arc_golds
        mask_pred = mask & arc_preds
        correct_arcs_wo_punc = (arc_preds == arc_golds)[mask_gold & mask_pred]
        correct_rels_wo_punc = (rel_preds == rel_golds)[mask_gold & mask_pred] & correct_arcs_wo_punc

        self.sum_gold_arcs_wo_punc += float(tf.math.count_nonzero(mask_gold))
        self.sum_pred_arcs_wo_punc += float(tf.math.count_nonzero(mask_pred))
        self.correct_arcs_wo_punc += float(tf.math.count_nonzero(correct_arcs_wo_punc))
        self.correct_rels_wo_punc += float(tf.math.count_nonzero(correct_rels_wo_punc))

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return self.las

    @property
    def uas(self):
        return self.uf

    @property
    def las(self):
        return self.lf

    @property
    def ur(self):
        if not self.sum_gold_arcs_wo_punc:
            return 0
        return self.correct_arcs_wo_punc / self.sum_gold_arcs_wo_punc

    @property
    def up(self):
        if not self.sum_pred_arcs_wo_punc:
            return 0
        return self.correct_arcs_wo_punc / self.sum_pred_arcs_wo_punc

    @property
    def lr(self):
        if not self.sum_gold_arcs_wo_punc:
            return 0
        return self.correct_rels_wo_punc / self.sum_gold_arcs_wo_punc

    @property
    def lp(self):
        if not self.sum_pred_arcs_wo_punc:
            return 0
        return self.correct_rels_wo_punc / self.sum_pred_arcs_wo_punc

    @property
    def uf(self):
        rp = self.ur + self.up
        if not rp:
            return 0
        return 2 * self.ur * self.up / rp

    @property
    def lf(self):
        rp = self.lr + self.lp
        if not rp:
            return 0
        return 2 * self.lr * self.lp / rp

    def reset_states(self):
        self.sum_gold_arcs_wo_punc = 0.0
        self.sum_pred_arcs_wo_punc = 0.0
        self.correct_arcs_wo_punc = 0.0
        self.correct_rels_wo_punc = 0.0

    def to_dict(self) -> dict:
        return {'UF': self.uf, 'LF': self.lf}
