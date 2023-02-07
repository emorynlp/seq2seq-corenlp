# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2023-01-10 18:08
from typing import List

from phrasetree.tree import Tree

from elit.components.seq2seq.dep.dep_utility import LB, RB


def bracket_linearize(tree: Tree, buffer: List):
    buffer.append(LB)
    buffer.append(tree.label())
    for t in tree:
        if isinstance(t, Tree):
            bracket_linearize(t, buffer)
        else:
            buffer.append(t)
    buffer.append(RB)


def flatten_terminals(tree: Tree, anonymize_token, placeholder='XX'):
    for i, child in enumerate(tree):
        if isinstance(child, str):
            if anonymize_token:
                tree[i] = placeholder
        elif child.label() == placeholder:
            tree[i] = placeholder if anonymize_token else child[0]
        else:
            flatten_terminals(child, anonymize_token, placeholder)


def unflatten_terminals(tree: Tree, placeholder='XX'):
    for i, child in enumerate(tree):
        if isinstance(child, str):
            tree[i] = Tree(placeholder, [child])
        else:
            unflatten_terminals(child, placeholder)


def find_first(stack, label: str, start=0):
    for i in range(start, len(stack)):
        if stack[i] == label:
            return i
    return -1