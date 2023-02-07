from typing import List, Tuple

SH = 0
RE = 1
RA = 2
LA = 3


def transition(trans, stack: List[int], buffer: List[int], arcs: List[Tuple[int, int, str]]):
    if trans == SH:
        stack.insert(0, buffer.pop(0))
    elif trans == RE:
        stack.pop(0)
    elif trans[0] == RA:
        top_w = stack[0]
        next_w = buffer[0]
        arcs.append((top_w, next_w, trans[1]))
        stack.insert(0, buffer.pop(0))
    elif trans[0] == LA:
        top_w = stack.pop(0)
        next_w = buffer[0]
        arcs.append((next_w, top_w, trans[1]))


def oracle(stack: List[int], buffer: List[int], heads: List[int], labels: List[str]):
    '''In accordance with algorithm 1 (Goldberg & Nivre, 2012)'''
    if heads[stack[0]] == buffer[0]:
        trans = (LA, labels[stack[0]])
    elif stack[0] == heads[buffer[0]]:
        trans = (RA, labels[buffer[0]])
    else:
        for i in range(stack[0]):
            if heads[i] == buffer[0] or heads[buffer[0]] == i:
                trans = RE
                return trans
        trans = SH
    return trans


def encode(trans) -> str:
    if trans == SH:
        return 'SH'
    elif trans == RE:
        return 'RE'
    elif trans[0] == RA:
        return 'RA-' + trans[1]
    elif trans[0] == LA:
        return 'LA-' + trans[1]


def decode(trans: str):
    if trans == 'SH':
        return SH
    elif trans == 'RE':
        return RE
    else:
        a, l = trans.split('-', 1)
        if a == 'RA':
            return RA, l
        else:
            return LA, l


def restore_from_arcs(arcs: List[Tuple[int, int, str]]):
    heads, labels = [None] * len(arcs), [None] * len(arcs)
    for h, d, l in arcs:
        d -= 1
        if d < len(arcs):
            heads[d] = h
            labels[d] = l
    return heads, labels
