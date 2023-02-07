# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-29 22:00
import copy
from abc import ABC, abstractmethod
from typing import List

from phrasetree.tree import Tree
from transformers import BartTokenizer
from elit.common.transform import VocabDict
from elit.common.vocab import Vocab
from elit.components.seq2seq.con.utility import bracket_linearize, flatten_terminals, unflatten_terminals, \
    find_first
from elit.components.seq2seq.dep.dep_utility import LB, RB
from elit.components.seq2seq.ner.seq2seq_ner import tokenize
from hanlp_common.configurable import AutoConfigurable
from hanlp_trie.dictionary import TupleTrieDict
from hanlp_common.document import Document


class Verbalizer(ABC, AutoConfigurable):
    def __call__(self, sample: dict):
        if 'constituency' in sample:
            tree: Tree = sample['constituency']
            sample['raw_gold'] = copy.deepcopy(tree)
            for subtree in tree.subtrees(lambda x: x.height() == 2):
                subtree.set_label('XX')
            sample['token'] = tokens = tree.leaves()
            sample['prompt'] = self.to_prompt(tokens, tree)
        return sample

    def tokenize_prompt(self, prompt, tokenizer: BartTokenizer):
        return tokenize(prompt, tokenizer, '')[-1]

    @abstractmethod
    def to_prompt(self, tokens, tree: Tree):
        pass

    def get_special_tokens(self):
        return []

    def get_tokens(self, sample):
        return sample['token']


class BracketedVerbalizer(Verbalizer):
    def __init__(self, vocabs=None, space='Ġ', flatten_pos=False, anonymize_token=False) -> None:
        super().__init__()
        self.space = space
        self.anonymize_token = anonymize_token
        self.flatten_pos = flatten_pos
        self.vocabs = vocabs or VocabDict(
            labels=Vocab(pad_token=None, unk_token=None)
        )

    def to_prompt(self, tokens, tree: Tree):
        brackets = []
        if self.flatten_pos:
            tree = copy.deepcopy(tree)
            flatten_terminals(tree, self.anonymize_token)
        bracket_linearize(tree, brackets)
        if isinstance(self.vocabs, VocabDict) and self.vocabs.mutable:
            self.vocabs['labels']([x.label() for x in tree.subtrees()])
        return brackets

    def get_special_tokens(self):
        if self.vocabs.mutable:
            self.vocabs.lock()
        tokens = set()
        tokens.update([f'{self.space}{x}' for x in sorted(self.vocabs['labels'].idx_to_token)])
        return sorted(tokens)

    def decode(self, batch, sample_index, prompt):
        prompt = prompt.replace(LB, '(').replace(RB, ')')
        nl = prompt.count('(')
        nr = prompt.count(')')
        if nl > nr:
            prompt += ')' * (nl - nr)
        elif nl < nr:
            prompt = '(' * (nr - nl) + prompt
        tree = Tree.fromstring(prompt)
        if self.flatten_pos:
            unflatten_terminals(tree)
        return tree

    def tokenize_prompt(self, prompt, tokenizer: BartTokenizer):
        ids = [tokenizer.bos_token_id]
        labels = self.vocabs['labels'].token_to_idx
        for t in prompt:
            if t in labels:
                ids.append(tokenizer.convert_tokens_to_ids(self.space + t))
            else:
                ids.extend(tokenizer(' ' + t, add_special_tokens=False).input_ids)
        ids.append(tokenizer.eos_token_id)
        return ids


def describe(tree: Tree, prompt: List, label_to_phrase, determiner=False, which=False):
    if which:
        prompt.append('which')
    else:
        label = label_to_phrase[tree.label()]
        if determiner:
            label = ' '.join(['the'] + label.split()[1:])
        prompt.append(label)
    if not all(isinstance(x, str) for x in tree):
        prompt.append('has')
    expanded = [False] * len(tree)
    if any(isinstance(son, Tree) for son in tree):
        i = 0
        while i < len(tree):
            son = tree[i]
            if isinstance(son, Tree):
                prompt.append(label_to_phrase[son.label()])
                if all(isinstance(x, str) for x in son):
                    prompt.append('" ')
                    prompt.append(' '.join(son))
                    prompt.append(' "')
                    expanded[i] = True
                i += 1
            else:
                j = i + 1
                while j < len(tree) and isinstance(tree[j], str):
                    j += 1
                prompt.append('" ' + ' '.join(tree[i:j]) + ' "')
                expanded[i:j] = [True] * (j - i)
                i = j
            if i != len(tree):
                prompt.append('and')
        n_to_expand = len(expanded) - sum(expanded)
        prompt.append(';' if n_to_expand > 1 else ',')
    else:
        n_to_expand = len(expanded)
    i = 0
    need_and = False
    while i < len(tree):
        if expanded[i]:
            i += 1
            continue
        son = tree[i]
        if isinstance(son, Tree):
            describe(son, prompt, label_to_phrase, determiner=True, which=(n_to_expand == 1 and i == len(tree) - 1))
            i += 1
        else:
            j = i + 1
            while j < len(tree) and isinstance(tree[j], str):
                j += 1
            prompt.append('" ' + ' '.join(tree[i:j]) + ' "')
            i = j
        if need_and:
            prompt.append('and')
        need_and = True
    if prompt[-1] in ('and',):
        del prompt[-1]


def describe_verbose(tree: Tree, prompt: List, label_to_phrase, determiner=False, which=False):
    which = False
    if which:
        prompt.append('which')
    else:
        label = label_to_phrase[tree.label()]
        if determiner:
            label = ' '.join(['the'] + label.split()[1:])
        prompt.append(label)
    if not all(isinstance(x, str) for x in tree):
        prompt.append('has')
    expanded = [False] * len(tree)
    if any(isinstance(son, Tree) for son in tree):
        i = 0
        while i < len(tree):
            son = tree[i]
            if isinstance(son, Tree):
                prompt.append(label_to_phrase[son.label()])
                if all(isinstance(x, str) for x in son):
                    prompt.append('" ')
                    prompt.append(' '.join(son))
                    prompt.append(' "')
                    expanded[i] = True
                i += 1
            else:
                j = i + 1
                while j < len(tree) and isinstance(tree[j], str):
                    j += 1
                prompt.append('" ' + ' '.join(tree[i:j]) + ' "')
                expanded[i:j] = [True] * (j - i)
                i = j
            if i != len(tree):
                prompt.append('and')
        n_to_expand = len(expanded) - sum(expanded)
        prompt.append(';' if n_to_expand > 1 else ',')
    else:
        n_to_expand = len(expanded)
    i = 0
    need_and = False
    while i < len(tree):
        if expanded[i]:
            i += 1
            continue
        son = tree[i]
        if isinstance(son, Tree):
            describe_verbose(son, prompt, label_to_phrase, determiner=True,
                             which=(n_to_expand == 1 and i == len(tree) - 1))
            i += 1
        else:
            j = i + 1
            while j < len(tree) and isinstance(tree[j], str):
                j += 1
            prompt.append('" ' + ' '.join(tree[i:j]) + ' "')
            i = j
        if need_and:
            prompt.append('and')
        need_and = True
    if prompt[-1] in ('and',):
        del prompt[-1]


def find_last_by_label(stack: List[Tree], label: str):
    for i in range(len(stack) - 1, -1, -1):
        if stack[i].label() == label:
            return i
    return -1


def find_first_by_label(stack: List[Tree], label: str):
    for i in range(len(stack)):
        if stack[i].label() == label:
            return i
    return -1


def find_last(stack, label: str):
    for i in range(len(stack) - 1, -1, -1):
        if stack[i] == label:
            return i
    return -1


def pop_last_by_label(stack: List[Tree], label: str):
    idx = find_last_by_label(stack, label)
    if idx == -1:
        raise IndexError(f'Cannot find {label} in {stack}')
    return stack.pop(idx)


def pop_first_by_label(stack: List[Tree], label: str):
    idx = find_first_by_label(stack, label)
    if idx == -1:
        raise IndexError(f'Cannot find {label} in {stack}')
    return stack.pop(idx)


def starts_with(seq: List, prefixes: List[List]):
    for p in prefixes:
        if seq[:len(p)] == p:
            return p
    return False


def find_closet(current: Tree, label: str):
    # noinspection PyUnresolvedReferences
    parent: Tree = current
    while parent:
        for sibling in parent:
            if isinstance(sibling, Tree) and sibling.label() == label and not len(sibling):
                return sibling
        # noinspection PyUnresolvedReferences
        parent = parent.parent
    return None


def un_describe(prompt: str, trie: TupleTrieDict, top=False):
    prompt = prompt.split()
    root = Tree('TOP', [])
    root.parent = None
    parent = root
    latest = None
    for step, chunk in enumerate(trie.split(prompt)):
        if isinstance(chunk, tuple):
            b, e, l = chunk
            if prompt[b] == 'the':  # reference
                closet = find_closet(parent, l)
                if closet is None:
                    continue
                parent = latest = closet
            else:
                latest = child = Tree(l, [])
                parent.append(child)
                child.parent = parent
        else:
            prefix = starts_with(chunk, [['has'], [',', 'which', 'has']])
            if prefix:
                if (not (chunk and chunk[0] == '"') and prefix[0] != 'has') or parent.label() == 'TOP':
                    parent = latest
                chunk = chunk[len(prefix):]
            prev_is_and = False
            if chunk and chunk[0] == 'and':
                chunk = chunk[1:]
                prev_is_and = True
            while chunk and chunk[0] == '"':
                idx = find_first(chunk, '"', 1)
                if idx < 0:
                    break
                form, chunk = chunk[1:idx], chunk[idx + 1:]
                if prev_is_and:
                    parent.extend(form)
                else:
                    latest.extend(form)
                if chunk and chunk[0] == 'and':
                    chunk = chunk[1:]
                    prev_is_and = True
    if top and root[0].label() == 'TOP':
        root = root[0]
    return root


def print_tree(tree: Tree):
    doc = Document(tok=tree.leaves(), con=tree.to_list())
    doc.pretty_print()


def save_tree(tree: Tree, path):
    with open(path, 'w') as out:
        out.write(str(tree))
        unflatten_terminals(tree)
        doc = Document(tok=tree.leaves(), con=tree.to_list())
        out.write(doc.to_pretty())


class IsAPhraseVerbalizer(Verbalizer):
    def __init__(self, label_to_phrase: dict, top=False, **kwargs) -> None:
        super().__init__()
        self.top = top
        self.label_to_phrase = label_to_phrase
        trie = dict((tuple(v.split()), k) for k, v in label_to_phrase.items())
        trie.update(dict((('the',) + tuple(v.split())[1:], k) for k, v in label_to_phrase.items()))
        self._trie = TupleTrieDict(trie)

    def to_prompt(self, tokens, tree: Tree):
        tree = copy.deepcopy(tree)
        # print_tree(tree)
        prompt = []
        flatten_terminals(tree, False)
        describe(tree if self.top else tree[0], prompt, self.label_to_phrase, determiner=False)
        prompt_text = ' '.join(prompt)
        # print(prompt_text)
        _de_tree = un_describe(prompt_text, self._trie, self.top)
        if _de_tree.to_list() != tree.to_list():
            # return None
            return ''
        #     print(prompt_text)
        #
        #     save_tree(tree, 'data/parsing/before.txt')
        #     save_tree(_de_tree, 'data/parsing/after.txt')
        #
        #     un_describe(prompt_text, self._trie, self.top)
        #     raise RuntimeError('Unable to recover this')
        return prompt_text

    def tokenize_prompt(self, prompt, tokenizer: BartTokenizer):
        return tokenizer(prompt).input_ids

    def decode(self, batch, sample_index, prompt):
        tree = un_describe(prompt, self._trie, self.top)
        unflatten_terminals(tree)
        if self.top:
            for pos_token in tree.subtrees(lambda x: x.height() == 2):
                if len(pos_token) > 1:
                    del pos_token[1:]
        return tree


class IsAPhraseVerbalizerVerbose(IsAPhraseVerbalizer):
    def to_prompt(self, tokens, tree: Tree):
        tree = copy.deepcopy(tree)
        # print_tree(tree)
        prompt = []
        flatten_terminals(tree, False)
        describe_verbose(tree if self.top else tree[0], prompt, self.label_to_phrase, determiner=False)
        prompt_text = ' '.join(prompt)
        return prompt_text


def shift_reduce_linearize(tree: Tree, buffer: List, lb=LB, rb=RB):
    buffer.append(lb + tree.label())
    for t in tree:
        if isinstance(t, Tree):
            shift_reduce_linearize(t, buffer, lb, rb)
        else:
            buffer.append(t)
    buffer.append(rb)


class ShiftReduceVerbalizer(BracketedVerbalizer):
    def to_prompt(self, tokens, tree: Tree):
        brackets = []
        tree = copy.deepcopy(tree)
        flatten_terminals(tree, True)
        shift_reduce_linearize(tree, brackets)
        if isinstance(self.vocabs, VocabDict) and self.vocabs.mutable:
            self.vocabs['labels'](brackets)
        return brackets

    def decode(self, batch, sample_index, prompt):
        prompt = prompt.replace('N-', '(').replace('RE', ')').replace(LB, '(').replace(RB, ')')
        nl = prompt.count('(')
        nr = prompt.count(')')
        if nl > nr:
            prompt += ')' * (nl - nr)
        elif nl < nr:
            prompt = '(' * (nr - nl) + prompt
        try:
            tree = Tree.fromstring(prompt)
        except ValueError as e:
            raise e
        unflatten_terminals(tree)
        return tree


def main():
    tree = Tree.fromstring(
        '(S (NP (NP My friend) (SBAR (WHNP who) (S (VP lives (PP in (NP Orlando)))))) (VP bought (NP me) (NP a gift (PP from (NP Disney World)))))')
    print(tree)
    # unflatten_terminals(tree)
    # print(tree.pformat(1000))
    # pass


if __name__ == '__main__':
    main()
