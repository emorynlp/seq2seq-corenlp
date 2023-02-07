# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-10-05 17:47
import torch
from torch import nn, FloatTensor
from torch.nn import Parameter

from elit.utils.torch_util import set_seed


class SoftRGCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_relations: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.weight = Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        torch.nn.init.normal_(self.weight)

    def forward(self, x: FloatTensor, adj: FloatTensor):
        batch_size, num_nodes, in_channels = x.size()
        f = torch.einsum('bni,rio->bnro', x, self.weight)
        f = f[:, :, None].expand(batch_size, num_nodes, num_nodes, self.num_relations, self.out_channels)
        f = f * adj.unsqueeze(-1)
        f = f.sum(1).sum(2)
        f /= adj.sum(1).sum(2).clamp_min(1e-16).unsqueeze(-1)  # avoid div by zero
        return f


def main():
    set_seed(1)
    batch_size = 2
    in_channels = 4
    num_nodes = 3
    x = torch.rand((batch_size, num_nodes, in_channels))
    mask = torch.tensor([[True, True, True], [True, True, False]])
    x[~mask] = 0
    num_relations = 2
    adj = torch.rand((batch_size, num_nodes, num_nodes, num_relations)).softmax(dim=-1)
    mask3d = mask.unsqueeze(-1).expand(batch_size, num_nodes, num_nodes) & mask.unsqueeze(1).expand(batch_size,
                                                                                                    num_nodes,
                                                                                                    num_nodes)
    adj[~mask3d] = 0
    out_channels = 5
    conv = SoftRGCNConv(in_channels, out_channels, num_relations)
    y = conv(x, adj)
    print(y)


if __name__ == '__main__':
    main()
