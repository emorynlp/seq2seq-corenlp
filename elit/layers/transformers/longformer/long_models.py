# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-10-13 19:52
import os
from typing import Optional, Union

import torch
from transformers.models.longformer.modeling_longformer import LongformerModel


def convert_model_to_long(model, convert_to_length, reserved_positions):
    """

    Args:
        model:
        convert_to_length:
        reserved_positions: RoBERTa has positions 0,1 reserved, so reserved_positions = 2

    Returns:

    """
    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    if convert_to_length:
        convert_to_length += reserved_positions
        if convert_to_length != current_max_pos:
            config = model.config
            config.max_position_embeddings = convert_to_length
            assert convert_to_length > current_max_pos
            # allocate a larger position embedding matrix
            new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(convert_to_length, embed_size)
            # copy position embeddings over and over to initialize the new position embeddings
            k = reserved_positions
            step = current_max_pos - reserved_positions
            while k < convert_to_length - 1:
                new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[reserved_positions:]
                k += step
            model.embeddings.position_embeddings.weight.data = new_pos_embed
            model.embeddings.position_embeddings.num_embeddings = convert_to_length
            model.embeddings.position_ids.data = torch.tensor(
                [i for i in range(convert_to_length)]).reshape(1, convert_to_length)
    return model


class LongBertModel(LongformerModel):
    base_model_prefix = 'bert'

    @classmethod
    def from_config(cls, config, **kwargs):
        if not hasattr(config, 'attention_window') and 'attention_window' not in kwargs:
            config.attention_window = 512
        if hasattr(config, 'convert_to_length'):
            config.max_position_embeddings = config.convert_to_length
        return cls._from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
                        *model_args, convert_to_length=None, reserved_positions=0, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return convert_model_to_long(model, convert_to_length, reserved_positions)


class LongElectraModel(LongBertModel):
    base_model_prefix = "electra"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
                        *model_args, convert_to_length=None, reserved_positions=0, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return convert_model_to_long(model, convert_to_length, reserved_positions)
