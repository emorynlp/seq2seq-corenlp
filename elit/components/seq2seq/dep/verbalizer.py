# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-29 22:00
from abc import ABC, abstractmethod
from typing import List, Dict

from transformers import BartTokenizer

from elit.common.transform import VocabDict
from elit.common.vocab import Vocab
from elit.components.seq2seq.dep import arc_eager, arc_standard
from elit.components.seq2seq.dep.arc_standard import DependencyTree
from elit.components.seq2seq.dep.dep_utility import bracketed, decode_to_dep, LB, RB, get_effective_tokens
from elit.components.seq2seq.ner.seq2seq_ner import tokenize
from hanlp_common.configurable import AutoConfigurable
from hanlp_common.conll import CoNLLSentence, CoNLLWord
from hanlp_common.constant import ROOT


class Verbalizer(ABC, AutoConfigurable):
    def __call__(self, sample: dict):
        if 'HEAD' in sample:
            sample['prompt'] = self.to_prompt(sample['FORM'], sample['HEAD'], sample['DEPREL'], sample=sample)
        return sample

    def tokenize_prompt(self, prompt, tokenizer: BartTokenizer):
        return tokenize(prompt, tokenizer, '')[-1]

    @abstractmethod
    def to_prompt(self, tokens, heads, relations, sample: dict):
        pass

    def get_special_tokens(self):
        return []

    def get_tokens(self, sample):
        return sample['FORM']

    def decode_head_rel(self, batch, index, pred_prompt: str, prompt_tokens: List[str]):
        return batch['_predictions'][index]


class HeadRelationVerbalizer(Verbalizer):
    def __init__(self, max_sent_len=0, relations=None) -> None:
        super().__init__()
        self.max_sent_len = max_sent_len
        self.relations = relations or Vocab(unk_token=None, pad_token=None)

    def to_prompt(self, tokens, heads, relations, sample: dict):
        return sum([(self.head_token(dep), self.relation_token(rel)) for dep, rel in zip(heads, relations)], ())

    def head_token(self, head):
        self.max_sent_len = max(head, self.max_sent_len)
        return f'<dep:{head}>'

    def relation_token(self, relation):
        token = f'<rel:{relation}>'
        if isinstance(self.relations, Vocab):
            self.relations.get_idx(token)
        return token

    def tokenize_prompt(self, prompt, tokenizer: BartTokenizer):
        return tokenizer.convert_tokens_to_ids(prompt)

    def get_special_tokens(self):
        special_tokens = set()
        for i in range(self.max_sent_len + 1):
            special_tokens.add(self.head_token(i))
        if isinstance(self.relations, Vocab):
            self.relations.lock()
            self.relations = self.relations.idx_to_token
        for relation in self.relations:
            special_tokens.add(relation)
        return sorted(special_tokens)

    def get_head_token_ids(self, tokenizer: BartTokenizer):
        ids = []
        for i in range(self.max_sent_len + 1):
            ids.append(tokenizer.convert_tokens_to_ids(self.head_token(i)))
        return ids

    def get_relation_token_ids(self, tokenizer: BartTokenizer):
        ids = []
        for i in self.relations:
            ids.append(tokenizer.convert_tokens_to_ids(i))
        return ids


class BracketedVerbalizer(Verbalizer):
    def __init__(self, vocabs=None, space='Ġ') -> None:
        super().__init__()
        self.space = space
        self.vocabs = vocabs or VocabDict(
            subscript=Vocab(pad_token=None, unk_token=None),
            deprel=Vocab(pad_token=None, unk_token=None)
        )

    def __call__(self, sample: dict):
        if 'HEAD' in sample:
            sample['prompt'], sample['effective_tokens'] = self.to_prompt(sample['FORM'], sample['HEAD'],
                                                                          sample['DEPREL'])
            # decoded = decode_to_dep(sample['prompt'], sample['effective_tokens'])
            # assert decoded == list(zip(sample['HEAD'], sample['DEPREL']))
            if isinstance(self.vocabs, VocabDict) and self.vocabs.mutable:
                self.vocabs['deprel']([f':{x}' for x in sample['DEPREL']])
                self.vocabs['subscript'](
                    ['_' + a.rsplit('_')[-1] for a, b in zip(sample['effective_tokens'][1:], sample['FORM']) if a != b])
        else:
            sample['effective_tokens'] = get_effective_tokens(sample['FORM'])
        return sample

    def to_prompt(self, tokens, heads, relations, sample: dict):
        return bracketed(tokens, heads, relations)

    def get_special_tokens(self):
        if self.vocabs.mutable:
            self.vocabs.lock()
        tokens = set()
        tokens.update(sorted(self.vocabs['subscript'].idx_to_token))
        tokens.update([f'{self.space}{x}' for x in sorted(self.vocabs['deprel'].idx_to_token)])
        return sorted(tokens)

    def get_tokens(self, sample):
        return sample['effective_tokens'][1:]

    def tokenize_prompt(self, prompt, tokenizer: BartTokenizer):
        ids = [tokenizer.bos_token_id]
        rel_vocab = self.vocabs['deprel'].token_to_idx
        for t in prompt:
            if t in rel_vocab:
                ids.append(tokenizer.convert_tokens_to_ids(self.space + t))
            else:
                ids.extend(tokenizer(' ' + t, add_special_tokens=False).input_ids)
        ids.append(tokenizer.eos_token_id)
        return ids

    def get_relation_token_ids(self, tokenizer: BartTokenizer):
        return dict(
            (tokenizer.convert_tokens_to_ids(self.space + x), x[1:]) for x in self.vocabs['deprel'].token_to_idx)

    def get_lb_id(self, tokenizer: BartTokenizer):
        return tokenizer(' ' + LB, add_special_tokens=False).input_ids[0]

    def get_rb_id(self, tokenizer: BartTokenizer):
        return tokenizer(' ' + RB, add_special_tokens=False).input_ids[0]

    def decode_head_rel(self, batch, index, pred_prompt: str, prompt_tokens: List[str]):
        dep_rel = decode_to_dep(pred_prompt.split(), batch['effective_tokens'][index])
        sanitized = []
        for each in dep_rel:
            if not each:
                each = (0, 'root')  # Just a random guess
            sanitized.append(each)
        return sanitized


class ArcEagerVerbalizer(Verbalizer):
    def __init__(self, vocabs=None, space='Ġ', lexical=False) -> None:
        super().__init__()
        self.lexical = lexical
        self.space = space
        self.vocabs = vocabs or VocabDict(
            action=Vocab(pad_token=None, unk_token=None)
        )

    def to_prompt(self, tokens, heads, relations, sample: dict):
        tokens_, heads_, relations_ = [ROOT] + tokens, [0] + heads, [ROOT] + relations
        stack = [0]
        buffer = [x for x in range(1, len(tokens_))]
        arcs = []
        actions = []
        while buffer:
            action = arc_eager.oracle(stack, buffer, heads_, relations_)
            actions.append(arc_eager.encode(action))
            arc_eager.transition(action, stack, buffer, arcs)
        if self.lexical:
            print()
        else:
            if self.vocabs.mutable:
                self.vocabs['action'].update(actions)
        return actions
        heads_, relations_ = arc_eager.restore_from_arcs(arcs)
        if heads_ != heads:
            pass
            sent = CoNLLSentence()
            for i, (t, h, r) in enumerate(zip(tokens, heads, relations)):
                sent.append(CoNLLWord(form=t, head=h, deprel=r, id=i + 1))
            print('Arc Eager failed')
            if not sent.projective:
                print('Tree is not projective')
            print(sent)
            raise RuntimeError('Arc Eager failed')
            # assert labels_ == labels[1:]

    def get_special_tokens(self):
        if self.vocabs.mutable:
            self.vocabs.lock()
        return sorted(self.vocabs['action'].idx_to_token)

    def get_action_token_ids(self, tokenizer):
        return tokenizer.convert_tokens_to_ids(self.get_special_tokens())

    def decode_head_rel(self, batch, index, pred_prompt: str, prompt_tokens: List[str]):
        tokens_ = [ROOT] + batch['FORM'][index]
        stack = [0]
        buffer = [x for x in range(1, len(tokens_))]
        arcs = []
        actions = prompt_tokens
        while buffer and actions:
            action = arc_eager.decode(actions.pop(0))
            try:
                arc_eager.transition(action, stack, buffer, arcs)
            except:
                continue
        heads_, relations_ = arc_eager.restore_from_arcs(arcs)
        return list(zip(heads_, relations_))


class ArcStandardVerbalizer(Verbalizer):
    def __init__(self, vocabs=None, lexical=False) -> None:
        super().__init__()
        self.lexical = lexical
        self.vocabs = vocabs or VocabDict(
            action=Vocab(pad_token=None, unk_token=None)
        )

    def to_prompt(self, tokens, heads, relations, sample: dict):
        stack = [0]
        buffer = [i + 1 for i in range(len(heads))]
        gold = DependencyTree(heads, relations)
        pred = DependencyTree(length=len(heads))
        actions = []
        try:
            while not (len(stack) == 1 and not buffer):
                action = arc_standard.oracle(stack, gold, pred)
                actions.append(action)
                arc_standard.apply_transition(action, stack, buffer, pred)
        except:
            # not projective
            # print('Arc Standard failed')
            pass

        if self.lexical:
            if self.vocabs.mutable:
                self.vocabs['action'].update(actions)
            offset = 0
            for i, a in enumerate(actions):
                if a == arc_standard.SH:
                    if offset < len(tokens):
                        actions[i] = tokens[offset]
                    offset += 1
        else:
            if self.vocabs.mutable:
                self.vocabs['action'].update(actions)

        # if not (pred.heads == gold.heads and pred.labels == gold.labels):
        #     return None
        # sent = CoNLLSentence()
        # for i, (t, h, r) in enumerate(zip(tokens, heads, relations)):
        #     sent.append(CoNLLWord(form=t, head=h, deprel=r, id=i + 1))
        # print('Arc Standard failed')
        # if not sent.projective:
        #     print('Tree is not projective')
        # print(sent)
        # raise RuntimeError('Arc Standard failed')
        return actions

    def get_special_tokens(self):
        if self.vocabs.mutable:
            self.vocabs.lock()
        return sorted(self.vocabs['action'].idx_to_token)

    def get_action_token_ids(self, tokenizer):
        return tokenizer.convert_tokens_to_ids(self.get_special_tokens())

    def decode_head_rel(self, batch, index, pred_prompt: str, prompt_tokens: List[str]):
        tokens = batch['FORM'][index]
        stack = [0]
        buffer = [i + 1 for i in range(len(tokens))]
        pred = DependencyTree(length=len(tokens))
        actions = self.de_lexical(prompt_tokens) if self.lexical else prompt_tokens
        while (buffer or len(stack) != 1) and actions:
            action = actions.pop(0)
            try:
                arc_standard.apply_transition(action, stack, buffer, pred)
            except:
                continue
        return list(zip(pred.heads[1:], pred.labels[1:]))

    def tokenize_prompt(self, prompt, tokenizer: BartTokenizer):
        if self.lexical:
            # [0, 440, 2156, 24, 21, 295, 75, 1378, 302, 479, 2]
            ids = [tokenizer.bos_token_id]
            for token in prompt:
                if token in self.vocabs['action']:
                    ids.append(tokenizer.convert_tokens_to_ids(token))
                else:
                    ids.extend(tokenize([token], tokenizer, ' ')[-1][1:-1])
            ids.append(tokenizer.eos_token_id)
            return ids
        else:
            return super().tokenize_prompt(prompt, tokenizer)

    def de_lexical(self, prompt_tokens: List[str]):
        de_lexical = []
        for t in prompt_tokens:
            if t in self.vocabs['action']:
                de_lexical.append(t)
            else:
                if t.startswith('Ġ'):
                    de_lexical.append(t[len('Ġ'):])
                else:
                    de_lexical[-1] += t
        return [x if x in self.vocabs['action'] else 'SH' for x in de_lexical]


def find_indices(haystack: List, needle):
    indices = []
    for i, each in enumerate(haystack):
        if each == needle:
            indices.append(i)
    return indices


class PromptVerbalizer(Verbalizer):
    def __init__(self, relations: Dict[str, str], vocabs: VocabDict = None, is_a_tag=False) -> None:
        super().__init__()
        self.is_a_tag = is_a_tag
        self.relations = relations
        self.vocabs = vocabs or VocabDict(
            action=Vocab(pad_token=None, unk_token=None)
        )

    def to_prompt(self, tokens, heads, relations, sample: dict):
        stack = [0]
        buffer = [i + 1 for i in range(len(heads))]
        gold = DependencyTree(heads, relations)
        pred = DependencyTree(length=len(heads))
        transitions = []
        prompt_tokens = []
        tokens = ['sentence'] + tokens
        debug_prompt = []
        try:
            while not (len(stack) == 1 and not buffer):
                transition = arc_standard.oracle(stack, gold, pred)
                transitions.append(transition)
                arc = arc_standard.apply_transition(transition, stack, buffer, pred)
                if arc:
                    d, r, h = arc
                    r = self.relations[r]
                    if transition.startswith('LA'):
                        phrase = ['is', r, 'of']
                        prompt_tokens.extend([tokens[d], *phrase, tokens[h], ';'])
                        debug_prompt.append((tokens[d], transition, tokens[h]))
                    elif transition.startswith('RA'):
                        phrase = ['has', r]
                        prompt_tokens.extend([tokens[h], *phrase, tokens[d], ';'])
                        debug_prompt.append((tokens[h], transition, tokens[d]))
                    else:
                        phrase = None
                    if self.vocabs.mutable and phrase:
                        self.vocabs['action'].add(' '.join(phrase))
        except Exception as e:
            # not projective
            # print(f'Arc Standard failed {e}')
            pass

        return prompt_tokens
        system = arc_standard.State(length=len(tokens) - 1)
        try:
            for second, transition, first in debug_prompt:
                make_token_on_stack(0, first, tokens, system)
                make_token_on_stack(1, second, tokens, system)
                system.apply(transition)

        except Exception as e:
            # traceback.print_exc()
            pass

        recovered_transitions = system.transitions
        sample['recoverable'] = recovered_transitions == transitions
        sample['prompt'] = ' '.join(sum([x + (';',) for x in debug_prompt], ()))
        sample['gold'] = ' '.join(transitions)
        sample['pred'] = ' '.join(recovered_transitions)
        if not sample['recoverable']:
            pass
            # print('Unable to recover transitions from the prompt')
        return prompt_tokens

    def tokenize_prompt(self, prompt, tokenizer: BartTokenizer):
        return tokenize(sum([x.split() for x in prompt], []), tokenizer, ' ')[-1]

    def get_special_tokens(self):
        if self.vocabs.mutable:
            self.vocabs.lock()
        if self.is_a_tag:
            return sorted(self.relations)
        return super().get_special_tokens()

    def decode_head_rel(self, batch, index, pred_prompt: str, prompt_tokens: List[str]):
        arc_standard_failed = batch['arc_standard_failed']
        system = batch['arc_standard'][index]
        if arc_standard_failed[index]:
            pred = batch['string_matching_results'][index]
            # sent = CoNLLSentence()
            # for i, (t, h, r) in enumerate(
            #         zip(batch['FORM'][index], batch['HEAD'][index], batch['DEPREL'][index])):
            #     sent.append(CoNLLWord(form=t, head=h, deprel=r, id=i + 1))
            # print('\n' + str(sent) + '\n\n')
            # exit(1)
        else:
            pred = system.pred
        return list(zip(pred.heads[1:], pred.labels[1:]))


def make_token_on_stack(index, target, tokens, system):
    while target != tokens[system.stack[index]]:
        system.apply('SH')


def make_index_on_stack(index, target, system):
    while target != system.stack[index]:
        system.apply('SH')
