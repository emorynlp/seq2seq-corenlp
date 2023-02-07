# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-02-14 16:47
import math
import warnings

import torch
from torch import nn


class ConcreteGate(nn.Module):
    """
    A gate made of stretched concrete distribution (using experimental Stretchable Concrete™)
    Can be applied to sparsify neural network activations or weights.
    Example usage: https://gist.github.com/justheuristic/1118a14a798b2b6d47789f7e6f511abd
    :param shape: shape of gate variable. can be broadcasted.
        e.g. if you want to apply gate to tensor [batch, length, units] over units axis,
        your shape should be [1, 1, units]
    :param temperature: concrete sigmoid temperature, should be in (0, 1] range
        lower values yield better approximation to actual discrete gate but train longer
    :param stretch_limits: min and max value of gate before it is clipped to [0, 1]
        min value should be negative in order to compute l0 penalty as in https://arxiv.org/pdf/1712.01312.pdf
        however, you can also use tf.nn.sigmoid(log_a) as regularizer if min, max = 0, 1
    :param l0_penalty: coefficient on the regularizer that minimizes l0 norm of gated value
    :param l2_penalty: coefficient on the regularizer that minimizes l2 norm of gated value
    :param eps: a small additive value used to avoid NaNs
    :param hard: if True, gates are binarized to {0, 1} but backprop is still performed as if they were concrete
    :param local_rep: if True, samples a different gumbel noise tensor for each sample in batch,
        by default, noise is sampled using shape param as size.

    """

    def __init__(self, shape, temperature=0.33, stretch_limits=(-0.1, 1.1), l0_penalty=0.0, l2_penalty=0.0, eps=1e-6,
                 hard=False, local_rep=False, name='gate', init=nn.init.xavier_uniform_):
        super().__init__()
        self.name = name
        self.temperature, self.stretch_limits, self.eps = temperature, stretch_limits, eps
        self.l0_penalty, self.l2_penalty = l0_penalty, l2_penalty
        self.hard, self.local_rep = hard, local_rep
        self.log_a = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters(init)

    def reset_parameters(self, init):
        init(self.log_a)

    def get_gates(self, is_train, shape=None) -> torch.FloatTensor:
        """ samples gate activations in [0, 1] interval """
        low, high = self.stretch_limits
        if is_train:
            shape = self.log_a.size() if shape is None else shape
            noise = torch.zeros(*shape, device=self.log_a.device, dtype=torch.float).uniform_(self.eps, 1.0 - self.eps)
            concrete = torch.sigmoid((torch.log(noise) - torch.log(1 - noise) + self.log_a) / self.temperature)
        else:
            concrete = torch.sigmoid(self.log_a)

        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete, 0, 1)
        if self.hard:
            hard_concrete = torch.gt(clipped_concrete, 0.5).to(torch.float)
            clipped_concrete = clipped_concrete + (hard_concrete - clipped_concrete).detach()
        return clipped_concrete

    def get_penalty(self, values=None, axis=None):
        """
        Computes l0 and l2 penalties. For l2 penalty one must also provide the sparsified values
        (usually activations or weights) before they are multiplied by the gate
        Returns the regularizer value that should to be MINIMIZED (negative logprior)
        """
        if self.l0_penalty == self.l2_penalty == 0:
            warnings.warn("get_penalty() is called with both penalties set to 0")
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"
        # compute p(gate_is_closed) = cdf(stretched_sigmoid < 0)
        p_open = torch.sigmoid(self.log_a - self.temperature * math.log(-low / high))
        p_open = torch.clamp(p_open, self.eps, 1.0 - self.eps)

        total_reg = 0.0
        if self.l0_penalty != 0:
            if values != None and self.local_rep:
                p_open += torch.zeros_like(values)  # broadcast shape to account for values
            l0_reg = self.l0_penalty * (torch.sum(p_open) if axis is None else torch.sum(p_open, dim=axis))
            total_reg += torch.mean(l0_reg)

        if self.l2_penalty != 0:
            assert values is not None
            l2_reg = 0.5 * self.l2_penalty * p_open * torch.sum(values ** 2, dim=axis)
            total_reg += torch.mean(l2_reg)

        return total_reg

    def get_sparsity_rate(self, is_train=False):
        """ Computes the fraction of gates which are now active (non-zero) """
        is_nonzero = torch.ne(self.get_gates(is_train), 0.0)
        return torch.mean(is_nonzero.to(torch.float))

    def forward(self, values, is_train=None, axis=None):
        """ applies gate to values, if is_train, adds regularizer to reg_collection """
        is_train = is_train or self.training
        gates = self.get_gates(is_train, shape=values.size() if self.local_rep else None)

        if is_train and (self.l0_penalty != 0 or self.l2_penalty != 0):
            reg = self.get_penalty(values=values, axis=axis)
        else:
            reg = None
        return values * gates, reg
