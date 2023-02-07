# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-09-28 23:13
from typing import List

SH = 'SH'
RA = 'RA'
LA = 'LA'


class DependencyTree(object):
    def __init__(self, heads: List[int] = None, labels: List[str] = None, length: int = None) -> None:
        if not heads:
            heads = [-1] * length
        if not labels:
            labels = [None] * length
        self.heads = [-1] + heads
        self.labels = [None] + labels


def apply_transition(transition: str, stack: List[int], buffer: List[int], pred: DependencyTree):
    """

    Args:
        transition:
        stack:
        buffer:
        pred:

    Returns:
        (dependent, relation, head)
    """
    if transition == SH:
        stack.insert(0, buffer.pop(0))
    elif transition.startswith(LA):
        # Te lef-arc transition (LA) creates a dependency from the
        # topmost word on the stack to the second-topmost word, and
        # pops the second-topmost word.
        top, sec = stack[0], stack[1]
        pred.heads[sec] = top
        pred.labels[sec] = transition.split('-', 1)[-1]
        stack.pop(1)
        return sec, pred.labels[sec], top
    elif transition.startswith(RA):
        # Te right-arc transition (RA) creates a dependency from the
        # second-topmost word on the stack to the topmost word, and
        # pops the topmost word.
        top, sec = stack[0], stack[1]
        pred.heads[top] = sec
        pred.labels[top] = transition.split('-', 1)[-1]
        stack.pop(0)
        return top, pred.labels[top], sec


def oracle(stack: List[int], gold: DependencyTree, pred: DependencyTree):
    top, sec = stack[0], stack[1] if len(stack) > 1 else -1
    if sec > 0 and gold.heads[sec] == top:
        return LA + '-' + gold.labels[sec]
    elif 0 <= sec == gold.heads[top] and not has_other_child(top, gold.heads, pred.heads):
        return RA + '-' + gold.labels[top]
    else:
        return SH


def has_other_child(k, gold_heads: List[int], pred_heads: List[int]):
    for i in range(1, len(gold_heads)):
        if gold_heads[i] == k and pred_heads[i] != k:
            return True
    return False


class State(object):
    def __init__(self, length: int, root_label='root', single_root=True) -> None:
        self.single_root = single_root
        self.root_label = root_label
        self.pred = DependencyTree(length=length)
        self.buffer = [i + 1 for i in range(length)]
        self.stack = [0]
        self.transitions = []

    def can_apply(self, transition: str):
        if transition.startswith("L") or transition.startswith("R"):
            label = transition[2:-1]
            if transition.startswith("L"):
                h = self.stack[0] if self.stack else -1
            else:
                h = self.stack[1] if len(self.stack) > 1 else -1
            if h < 0:
                return False
            if h == 0 and label != self.root_label:
                return False

        n_stack = len(self.stack)
        n_buffer = len(self.buffer)

        if transition.startswith("L"):
            return n_stack > 2
        elif transition.startswith("R"):
            if self.single_root:
                return (n_stack > 2) or (n_stack == 2 and n_buffer == 0)
            else:
                return n_stack >= 2
        return n_buffer > 0

    def is_terminal(self) -> bool:
        return len(self.stack) == 1 and len(self.buffer) == 0

    def apply(self, transition: str):
        self.transitions.append(transition)
        return apply_transition(transition, self.stack, self.buffer, self.pred)


def main():
    heads = [2, 7, 4, 2, 6, 4, 0, 7, 10, 7, 13, 13, 7]
    labels = ['poss', 'nsubj', 'nsubj', 'relcl', 'case', 'obl', 'root', 'iobj', 'det', 'obj', 'case', 'compound', 'obl']
    stack = [0]
    buffer = [i + 1 for i in range(len(heads))]
    gold = DependencyTree(heads, labels)
    pred = DependencyTree(length=len(heads))
    actions = []
    while buffer or len(stack) != 1:
        action = oracle(stack, gold, pred)
        actions.append(action)
        apply_transition(action, stack, buffer, pred)
    assert pred.heads == gold.heads and pred.labels == gold.labels
    print(actions)


if __name__ == '__main__':
    main()
