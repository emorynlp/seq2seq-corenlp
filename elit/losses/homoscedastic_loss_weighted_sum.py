# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-22 21:14
import torch
import torch.nn as nn


class HomoscedasticLossWeightedSum(nn.Module):

    def __init__(self, num_losses):
        """Automatically weighted sum of multi-task losses described in :cite:`Kendall_2018_CVPR`.

        Args:
            num_losses: The number of losses.
        """
        super(HomoscedasticLossWeightedSum, self).__init__()
        params = torch.ones(num_losses, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *losses):
        losses = torch.stack(losses)
        norm = self.params ** 2
        return torch.sum(0.5 / norm * losses + torch.log(1 + norm))
