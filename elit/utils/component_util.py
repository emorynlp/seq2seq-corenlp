# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-31 19:24
import os
from hanlp_common.constant import HANLP_VERBOSE
from hanlp_common.io import load_json, eprint, save_json
from hanlp_common.reflection import object_from_classpath, str_to_type
from elit import pretrained
from elit import version
from elit.common.component import Component
from elit.utils.io_util import get_resource, get_latest_info_from_pypi, check_version_conflicts
from hanlp_common.util import isdebugging


def load_from_meta_file(save_dir: str, meta_filename='meta.json', transform_only=False, verbose=HANLP_VERBOSE,
                        **kwargs) -> Component:
    """
    Load a component from a ``meta.json`` (legacy TensorFlow component) or a ``config.json`` file.

    Args:
        save_dir: The identifier.
        meta_filename (str): The meta file of that saved component, which stores the classpath and version.
        transform_only: Load and return only the transform.
        **kwargs: Extra parameters passed to ``component.load()``.

    Returns:

        A component.
    """
    identifier = save_dir
    load_path = save_dir
    save_dir = get_resource(save_dir)
    if save_dir.endswith('.json'):
        meta_filename = os.path.basename(save_dir)
        save_dir = os.path.dirname(save_dir)
    metapath = os.path.join(save_dir, meta_filename)
    if not os.path.isfile(metapath):
        tf_model = False
        metapath = os.path.join(save_dir, 'config.json')
    else:
        tf_model = True
    cls = None
    if not os.path.isfile(metapath):
        tips = ''
        if save_dir.isupper():
            from difflib import SequenceMatcher
            similar_keys = sorted(pretrained.ALL.keys(),
                                  key=lambda k: SequenceMatcher(None, k, identifier).ratio(),
                                  reverse=True)[:5]
            tips = f'Check its spelling based on the available keys:\n' + \
                   f'{sorted(pretrained.ALL.keys())}\n' + \
                   f'Tips: it might be one of {similar_keys}'
        # These components are not intended to be loaded in this way, but I'm tired of explaining it again and again
        if identifier in pretrained.word2vec.ALL.values():
            save_dir = os.path.dirname(save_dir)
            metapath = os.path.join(save_dir, 'config.json')
            save_json({'classpath': 'elit.layers.embeddings.word2vec.Word2VecEmbeddingComponent',
                       'embed': {'classpath': 'elit.layers.embeddings.word2vec.Word2VecEmbedding',
                                 'embed': identifier, 'field': 'token', 'normalize': 'l2'},
                       'hanlp_version': version.__version__}, metapath)
        elif identifier in pretrained.fasttext.ALL.values():
            save_dir = os.path.dirname(save_dir)
            metapath = os.path.join(save_dir, 'config.json')
            save_json({'classpath': 'elit.layers.embeddings.fast_text.FastTextEmbeddingComponent',
                       'embed': {'classpath': 'elit.layers.embeddings.fast_text.FastTextEmbedding',
                                 'filepath': identifier, 'src': 'token'},
                       'hanlp_version': version.__version__}, metapath)
        elif identifier in {pretrained.classifiers.LID_176_FASTTEXT_SMALL,
                            pretrained.classifiers.LID_176_FASTTEXT_BASE}:
            save_dir = os.path.dirname(save_dir)
            metapath = os.path.join(save_dir, 'config.json')
            save_json({'classpath': 'elit.components.classifiers.fasttext_classifier.FastTextClassifier',
                       'model_path': identifier,
                       'hanlp_version': version.__version__}, metapath)
        else:
            raise FileNotFoundError(f'The identifier {save_dir} resolves to a nonexistent meta file {metapath}. {tips}')
    meta: dict = load_json(metapath)
    cls = meta.get('classpath', cls)
    if not cls:
        cls = meta.get('class_path', None)  # For older version
    if tf_model:
        # tf models are trained with version < 2.1. To migrate them to 2.1, map their classpath to new locations
        upgrade = {
            'elit.components.tok_tf.TransformerTokenizerTF': 'elit.components.tokenizers.tok_tf.TransformerTokenizerTF',
            'elit.components.pos.RNNPartOfSpeechTagger': 'elit.components.taggers.pos_tf.RNNPartOfSpeechTaggerTF',
            'elit.components.pos_tf.RNNPartOfSpeechTaggerTF': 'elit.components.taggers.pos_tf.RNNPartOfSpeechTaggerTF',
            'elit.components.pos_tf.CNNPartOfSpeechTaggerTF': 'elit.components.taggers.pos_tf.CNNPartOfSpeechTaggerTF',
            'elit.components.ner_tf.TransformerNamedEntityRecognizerTF': 'elit.components.ner.ner_tf.TransformerNamedEntityRecognizerTF',
            'elit.components.parsers.biaffine_parser.BiaffineDependencyParser': 'elit.components.parsers.biaffine_parser_tf.BiaffineDependencyParserTF',
            'elit.components.parsers.biaffine_parser.BiaffineSemanticDependencyParser': 'elit.components.parsers.biaffine_parser_tf.BiaffineSemanticDependencyParserTF',
            'elit.components.tok_tf.NgramConvTokenizerTF': 'elit.components.tokenizers.tok_tf.NgramConvTokenizerTF',
            'elit.components.classifiers.transformer_classifier.TransformerClassifier': 'elit.components.classifiers.transformer_classifier_tf.TransformerClassifierTF',
            'elit.components.taggers.transformers.transformer_tagger.TransformerTagger': 'elit.components.taggers.transformers.transformer_tagger_tf.TransformerTaggerTF',
            'elit.components.tok.NgramConvTokenizer': 'elit.components.tokenizers.tok_tf.NgramConvTokenizerTF',
        }
        cls = upgrade.get(cls, cls)
    assert cls, f'{meta_filename} doesn\'t contain classpath field'
    try:
        obj: Component = object_from_classpath(cls)
        if hasattr(obj, 'load'):
            if transform_only:
                # noinspection PyUnresolvedReferences
                obj.load_transform(save_dir)
            else:
                if os.path.isfile(os.path.join(save_dir, 'config.json')):
                    obj.load(save_dir, verbose=verbose, **kwargs)
                else:
                    obj.load(metapath, **kwargs)
            obj.config['load_path'] = load_path
        return obj
    except ModuleNotFoundError as e:
        if isdebugging():
            raise e from None
        else:
            raise ModuleNotFoundError(
                f'Some modules ({e.name} etc.) required by this model are missing. Please install the full version:'
                '\n\n\tpip install elit[full] -U') from None
    except ValueError as e:
        if e.args and isinstance(e.args[0], str) and 'Internet connection' in e.args[0]:
            raise ConnectionError(
                'Hugging Face 🤗 Transformers failed to download because your Internet connection is either off or bad.\n'
                'See https://hanlp.hankcs.com/docs/install.html#server-without-internet for solutions.') \
                from None
        raise e from None
    except Exception as e:
        # Some users often install an incompatible tf and put the blame on HanLP. Teach them the basics.
        try:
            you_installed_wrong_versions, extras = check_version_conflicts(extras=('full',) if tf_model else None)
        except:
            you_installed_wrong_versions, extras = None, None
        if you_installed_wrong_versions:
            raise version.NotCompatible(you_installed_wrong_versions + '\nPlease reinstall HanLP in the right way:' +
                                        '\n\n\tpip install --upgrade elit' + (
                                            f'[{",".join(extras)}]' if extras else '')) from None
        eprint(f'Failed to load {identifier}')
        from pkg_resources import parse_version
        model_version = meta.get("hanlp_version", '2.0.0-alpha.0')
        if model_version == '2.0.0':  # Quick fix: the first version used a wrong string
            model_version = '2.0.0-alpha.0'
        model_version = parse_version(model_version)
        installed_version = parse_version(version.__version__)
        try:
            latest_version = get_latest_info_from_pypi()
        except:
            latest_version = None
        if model_version > installed_version:
            eprint(f'{identifier} was created with elit-{model_version}, '
                   f'while you are running a lower version: {installed_version}. ')
        if installed_version != latest_version:
            eprint(
                f'Please upgrade HanLP with:\n'
                f'\n\tpip install --upgrade elit\n')
        eprint(
            'If the problem still persists, please submit an issue to https://github.com/hankcs/HanLP/issues\n'
            'When reporting an issue, make sure to paste the FULL ERROR LOG below.')

        eprint(f'{"ERROR LOG BEGINS":=^80}')
        import platform
        eprint(f'OS: {platform.platform()}')
        eprint(f'Python: {platform.python_version()}')
        import torch
        eprint(f'PyTorch: {torch.__version__}')
        if tf_model:
            try:
                import tensorflow
                tf_version = tensorflow.__version__
            except ModuleNotFoundError:
                tf_version = 'not installed'
            eprint(f'TensorFlow: {tf_version}')
        eprint(f'HanLP: {version.__version__}')
        import sys
        sys.stderr.flush()
        try:
            if e.args and isinstance(e.args, tuple):
                for i in range(len(e.args)):
                    if isinstance(e.args[i], str):
                        from hanlp_common.util import set_tuple_with
                        e.args = set_tuple_with(e.args, e.args[i] + f'\n{"ERROR LOG ENDS":=^80}', i)
                        break
        except:
            pass
        raise e from None


def load_from_meta(meta: dict) -> Component:
    if 'load_path' in meta:
        return load_from_meta_file(meta['load_path'])
    cls = meta.get('class_path', None) or meta.get('classpath', None)
    assert cls, f'{meta} doesn\'t contain classpath field'
    cls = str_to_type(cls)
    return cls.from_config(meta)
