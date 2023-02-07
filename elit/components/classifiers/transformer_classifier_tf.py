# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-11-10 13:19

import math
from typing import Union, Tuple, Any, Iterable
import tensorflow as tf
from elit.common.keras_component import KerasComponent
from hanlp_common.structure import SerializableDict
from elit.layers.transformers.loader_tf import build_transformer
from elit.optimizers.adamw import create_optimizer
from elit.transform.table_tf import TableTransform
from elit.utils.log_util import logger
from hanlp_common.util import merge_locals_kwargs
from transformers.tokenization_utils import PreTrainedTokenizer

class TransformerTextTransform(TableTransform):

    def __init__(self, config: SerializableDict = None, map_x=False, map_y=True, x_columns=None,
                 y_column=-1, skip_header=True, delimiter='auto', multi_label=False, **kwargs) -> None:
        super().__init__(config, map_x, map_y, x_columns, y_column, multi_label, skip_header, delimiter, **kwargs)
        self.tokenizer: PreTrainedTokenizer = None

    def inputs_to_samples(self, inputs, gold=False):
        tokenizer = self.tokenizer
        max_length = self.config.max_length
        num_features = None
        pad_token = None if self.label_vocab.mutable else tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        for (X, Y) in super().inputs_to_samples(inputs, gold):
            if self.label_vocab.mutable:
                yield None, Y
                continue
            if isinstance(X, str):
                X = (X,)
            if num_features is None:
                num_features = self.config.num_features
            assert num_features == len(X), f'Numbers of features {num_features} ' \
                                           f'inconsistent with current {len(X)}={X}'
            text_a = X[0]
            text_b = X[1] if len(X) > 1 else None
            tokens_a = self.tokenizer.tokenize(text_a)
            tokens_b = self.tokenizer.tokenize(text_b) if text_b else None
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            if tokens_b:
                tokens += tokens_b
                segment_ids += [1] * len(tokens_b)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(token_ids)
            diff = max_length - len(token_ids)
            if diff < 0:
                # logger.warning(
                #     f'Input tokens {tokens} exceed the max sequence length of {max_length - 2}. '
                #     f'The exceeded part will be truncated and ignored. '
                #     f'You are recommended to split your long text into several sentences within '
                #     f'{max_length - 2} tokens beforehand.')
                token_ids = token_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                segment_ids = segment_ids[:max_length]
            elif diff > 0:
                token_ids += [pad_token] * diff
                attention_mask += [0] * diff
                segment_ids += [0] * diff

            assert len(token_ids) == max_length, "Error with input length {} vs {}".format(len(token_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                max_length)
            assert len(segment_ids) == max_length, "Error with input length {} vs {}".format(len(segment_ids),
                                                                                             max_length)

            label = Y
            yield (token_ids, attention_mask, segment_ids), label

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        max_length = self.config.max_length
        types = (tf.int32, tf.int32, tf.int32), tf.string
        shapes = ([max_length], [max_length], [max_length]), [None, ] if self.config.get('multi_label', None) else []
        values = (0, 0, 0), self.label_vocab.safe_pad_token
        return types, shapes, values

    def x_to_idx(self, x) -> Union[tf.Tensor, Tuple]:
        logger.fatal('map_x should always be set to True')
        exit(1)

    def y_to_idx(self, y) -> tf.Tensor:
        if self.config.get('multi_label', None):
            # need to change index to binary vector
            mapped = tf.map_fn(fn=lambda x: tf.cast(self.label_vocab.lookup(x), tf.int32), elems=y,
                               fn_output_signature=tf.TensorSpec(dtype=tf.dtypes.int32, shape=[None, ]))
            one_hots = tf.one_hot(mapped, len(self.label_vocab))
            idx = tf.reduce_sum(one_hots, -2)
        else:
            idx = self.label_vocab.lookup(y)
        return idx

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False, inputs=None, X=None,
                     batch=None) -> Iterable:
        # Prediction to be Y > 0:
        if self.config.get('multi_label', None):
            preds = Y
        else:
            preds = tf.argmax(Y, axis=-1)
        for y in preds:
            yield self.label_vocab.idx_to_token[y]

    def input_is_single_sample(self, input: Any) -> bool:
        return isinstance(input, (str, tuple))


class TransformerClassifierTF(KerasComponent):

    def __init__(self, bert_text_transform=None) -> None:
        if not bert_text_transform:
            bert_text_transform = TransformerTextTransform()
        super().__init__(bert_text_transform)
        self.model: tf.keras.Model
        self.transform: TransformerTextTransform = bert_text_transform

    # noinspection PyMethodOverriding
    def fit(self, trn_data: Any, dev_data: Any, save_dir: str, transformer: str, max_length: int = 128,
            optimizer='adamw', warmup_steps_ratio=0.1, use_amp=False, batch_size=32,
            epochs=3, logger=None, verbose=1, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def evaluate_output(self, tst_data, out, num_batches, metric):
        out.write('sentence\tpred\tgold\n')
        total, correct, score = 0, 0, 0
        for idx, batch in enumerate(tst_data):
            outputs = self.model.predict_on_batch(batch[0])
            outputs = tf.argmax(outputs, axis=1)
            for X, Y_pred, Y_gold, in zip(batch[0][0], outputs, batch[1]):
                feature = ' '.join(self.transform.tokenizer.convert_ids_to_tokens(X.numpy()))
                feature = feature.replace(' ##', '')  # fix sub-word generated by BERT tagger
                out.write('{}\t{}\t{}\n'.format(feature,
                                                self._y_id_to_str(Y_pred),
                                                self._y_id_to_str(Y_gold)))
                total += 1
                correct += int(tf.equal(Y_pred, Y_gold).numpy())
            score = correct / total
            print('\r{}/{} {}: {:.2f}'.format(idx + 1, num_batches, metric, score * 100), end='')
        print()
        return score

    def _y_id_to_str(self, Y_pred) -> str:
        return self.transform.label_vocab.idx_to_token[Y_pred.numpy()]

    def build_loss(self, loss, **kwargs):
        if loss:
            assert isinstance(loss, tf.keras.losses.loss), 'Must specify loss as an instance in tf.keras.losses'
            return loss
        elif self.config.get('multi_label', None):
            # Loss to be BinaryCrossentropy for multi-label:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss

    # noinspection PyMethodOverriding
    def build_optimizer(self, optimizer, use_amp, train_steps, warmup_steps, **kwargs):
        if optimizer == 'adamw':
            opt = create_optimizer(init_lr=5e-5, num_train_steps=train_steps, num_warmup_steps=warmup_steps)
            # opt = tfa.optimizers.AdamW(learning_rate=3e-5, epsilon=1e-08, weight_decay=0.01)
            # opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
            self.config.optimizer = tf.keras.utils.serialize_keras_object(opt)
            lr_config = self.config.optimizer['config']['learning_rate']['config']
            if hasattr(lr_config['decay_schedule_fn'], 'get_config'):
                lr_config['decay_schedule_fn'] = dict(
                    (k, v) for k, v in lr_config['decay_schedule_fn'].config().items() if not k.startswith('_'))
        else:
            opt = super().build_optimizer(optimizer)
        if use_amp:
            # loss scaling is currently required when using mixed precision
            opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
        return opt

    # noinspection PyMethodOverriding
    def build_model(self, transformer, max_length, **kwargs):
        model, self.transform.tokenizer = build_transformer(transformer, max_length, len(self.transform.label_vocab),
                                                            tagging=False)
        return model

    def build_vocab(self, trn_data, logger):
        train_examples = super().build_vocab(trn_data, logger)
        warmup_steps_per_epoch = math.ceil(train_examples * self.config.warmup_steps_ratio / self.config.batch_size)
        self.config.warmup_steps = warmup_steps_per_epoch * self.config.epochs
        return train_examples

    def build_metrics(self, metrics, logger, **kwargs):
        if self.config.get('multi_label', None):
            metric = tf.keras.metrics.BinaryAccuracy('binary_accuracy')
        else:
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        return [metric]
