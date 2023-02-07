# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-11-10 16:05
import math
import torch
from transformers import BartTokenizer
from transformers.generation_logits_process import LogitsProcessor

from elit.utils.log_util import cprint


class ShiftReduceProcessor(LogitsProcessor):
    def __init__(self, batch, ls, sh, rs, tokenizer: BartTokenizer):
        self.rs = rs
        self.sh = sh
        self.ls = ls
        self.tokenizer = tokenizer
        self.batch = batch
        self.eos = tokenizer.eos_token_id
        self.bos = tokenizer.bos_token_id
        tokens = batch['token']
        self.offsets = [0] * len(tokens)
        self.depth = [0] * len(tokens)
        self.batch['_predictions'] = [[] for _ in tokens]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        batch = self.batch
        for batch_id, beam_sent in enumerate(input_ids.view(-1, 1, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed_tokens = set()
                index = batch_id * 1 + beam_id
                prefix_ids: list = input_ids[index][1:].tolist()
                if self.eos in prefix_ids:
                    prefix_ids = prefix_ids[:prefix_ids.index(self.eos)]
                    allowed_tokens.add(self.eos)
                prefix_str = self.tokenizer.convert_ids_to_tokens(prefix_ids)
                tokens = self.batch['token'][index]
                if prefix_ids:
                    if prefix_ids[-1] == self.rs:
                        self.depth[index] -= 1
                    elif prefix_ids[-1] == self.sh:
                        self.offsets[index] += 1
                    elif prefix_ids[-1] in self.ls:
                        self.depth[index] += 1
                if self.depth[index]:
                    allowed_tokens.add(self.sh)
                allowed_tokens.update(self.ls)
                if self.depth[index]:
                    allowed_tokens.add(self.rs)
                elif prefix_ids:
                    allowed_tokens = {self.eos}
                allowed_tokens = sorted(list(allowed_tokens))
                mask[index, allowed_tokens] = 0
                # cprint(f'{len(prefix_ids)} {prefix_str} [yellow]{self.tokenizer.convert_ids_to_tokens(allowed_tokens)}[/yellow]')

        return scores + mask
