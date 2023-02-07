# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-10-21 11:19
from collections import defaultdict

import penman

from elit.components.amr.seq2seq.dataset.penman import pm_load
from tests import cdroot

trans_relation = {
    'ARG1': 'patient',
    'mod': 'modifier',
    'op1': 'first operator',
    'ARG0': 'agent',
    "op2": 'second operator',
    "ARG2": 'instrument',
    "quant": 'quantity',
    "li": 'list item',
    "poss": 'possession',
    "ARG3": 'starting point',
    "ARG4": 'ending point',
    "op3": 'third operator',
    "snt1": 'first sentence',
    "snt2": 'second sentence',
    "op4": 'fourth operator',
    "snt3": 'third sentence',
    "ord": 'ordinal',
    "consist-of": 'consists of',
    "dayperiod": 'day period',
    "prep-in": 'in',
    "prep-on": 'on',
    "prep-as": 'as',
    "op5": 'fifth operator',
    "op6": 'sixth operator',
    "op7": 'seventh operator',
    "prep-with": 'with',
    "prep-in-addition-to": 'in addition to',
    "prep-against": 'against',
    "snt4": 'forth sentence',
    "snt5": 'fifth sentence',
    "snt6": 'sixth sentence',
    "snt7": 'seventh sentence',
    "snt8": 'eighth sentence',
    "snt9": 'ninth sentence',
    "snt10": 'tenth sentence',
    "snt11": 'eleventh sentence',
    "op8": 'eighth operator',
    "op9": 'ninth sentence',
    "prep-to": 'to',
    "prep-on-behalf-of": 'on behalf of',
    "prep-from": 'from',
    "prep-for": 'for',
    "op10": 'tenth operator',
    "prep-along-with": 'along with',
    "prep-without": 'without',
    "prep-by": 'by',
    "prep-among": 'among',
    "conj-as-if": 'as if',
    "op11": 'eleventh operator',
    "op12": 'twelfth',
    "op13": 'thirteenth',
    "op14": 'fourteenth',
    "op15": 'fifteenth',
    "prep-under": 'under',
    "prep-amid": 'amid',
    "prep-toward": 'toward',
    "prep-out-of": 'out of',
    "prep-into": 'into',
    "quarter": 'quarter',
    "prep-at": 'at',
    "op16": 'sixteenth operator',
    "op17": 'seventeenth operator',
    "op18": 'eighteenth operator',
    "op19": 'nineteenth operator',
    "op20": 'twentieth operator',
    "op21": 'twenty first operator',
    "op22": 'twenty second operator',
    "op23": 'twenty third operator',
    "op24": 'twenty fourth operator',
}


def dfs(graph: penman.Graph, root):
    visited = {root}
    stack = [(None, root)]

    v2n = dict((s, t) for s, r, t in graph.instances())
    adj = defaultdict(dict)
    for s, r, t in graph.triples:
        if r == ':instance':
            continue
        adj[s][t] = r

    triples = []
    while stack:
        p, s = stack.pop()
        # print(s)
        if p:
            source = v2n[p]
            target = v2n[s]
            relation = adj[p][s]
            triples.append((source, relation, target))

        for neighbour in adj[s]:
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append((s, neighbour))
            else:
                source = v2n[s]
                target = v2n[neighbour]
                relation = adj[s][neighbour]
                triples.append((source, relation, target))
    assert sum(len(x) for x in adj.values()) == len(triples), 'Some triples missing'
    return triples


def to_prompt(graph: penman.Graph):
    triples = dfs(graph, graph.top)
    print(triples)


def main():
    cdroot()
    graph = pm_load('data/amr/amr_3.0/debug.txt', dereify=False)
    for each in graph:
        text = to_prompt(each)


if __name__ == '__main__':
    main()
