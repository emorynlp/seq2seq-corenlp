# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-04-05 17:36
from collections import Counter, defaultdict
from typing import List

LB = '['
RB = ']'


def bracketed(tokens, heads, rels):
    tree = [dict() for _ in range(len(tokens) + 1)]
    for i, (h, r) in enumerate(zip(heads, rels)):
        i = i + 1
        tree[h][i] = r
    effective_tokens = get_effective_tokens(tokens)
    return _dfs(0, effective_tokens, tree, [])[1:], effective_tokens


def get_effective_tokens(tokens):
    relative_ids = [None] + build_relative_ids(tokens)
    effective_tokens = apply_relative_ids(['ROOT'] + tokens, relative_ids)
    return effective_tokens


def build_relative_ids(tokens):
    relative_vocab = Counter()
    relative_ids = []
    for t in tokens:
        relative_vocab[t] += 1
        relative_ids.append(relative_vocab[t])
    for i, t in enumerate(tokens):
        if relative_vocab[t] == 1:
            relative_ids[i] = None
    return relative_ids


def apply_relative_ids(tokens, ids):
    effective_tokens = []
    for t, i in zip(tokens, ids):
        if i:
            t += f'_{i}'
        effective_tokens.append(t)
    return effective_tokens


def _dfs(root, tokens, tree, buffer: List):
    for child, rel in tree[root].items():
        buffer.append(f':{rel}')
        buffer.append(LB)
        buffer.append(tokens[child])
        _dfs(child, tokens, tree, buffer)
        buffer.append(RB)
    return buffer


def parse_relation(token: str):
    if token.startswith(':') and len(token) >= 2:
        return token[1:]


def decode_to_dep(bracketed, tokens):
    tree = defaultdict(dict)
    current = 'ROOT'
    relation = ':root'
    stack = []
    expect_token = False
    for c in bracketed:
        if c == LB:
            stack.append((current, relation))
            expect_token = True
        elif c == RB:
            parent, parent_relation = stack.pop()
            tree[parent][current] = parent_relation
            current = parent
        else:
            if expect_token:
                current = c
                expect_token = False
            else:
                relation = c
    deprel = [None] * len(tokens)
    for head, children in tree.items():
        for dep, rel in children.items():
            deprel[tokens.index(dep)] = (tokens.index(head), parse_relation(rel))
    return deprel[1:]


def main():
    tokens = ['A', 'record', 'date', 'has', "n't", 'been', 'set', '.']
    dep = [(3, 'det'), (3, 'nn'), (7, 'nsubjpass'), (7, 'aux'), (7, 'neg'), (7, 'auxpass'), (0, 'root'), (7, 'punct')]
    heads = [x[0] for x in dep]
    rels = [x[1] for x in dep]
    b, effective_tokens = bracketed(tokens, heads, rels)
    print(b)
    dep_decoded = decode_to_dep(b, effective_tokens)
    print(dep_decoded)
    assert dep == dep_decoded


if __name__ == '__main__':
    main()
