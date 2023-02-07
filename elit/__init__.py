# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-13 18:05
import elit.common
import elit.components
import elit.pretrained
import elit.utils
from elit.version import __version__

elit.utils.ls_resource_in_module(elit.pretrained)


def load(save_dir: str, verbose=None, **kwargs) -> elit.common.component.Component:
    """Load a pretrained component from an identifier.

    Args:
      save_dir (str): The identifier to the saved component. It could be a remote URL or a local path.
      verbose: ``True`` to print loading progress.
      **kwargs: Arguments passed to :func:`elit.common.torch_component.TorchComponent.load`, e.g.,
        ``devices`` is a useful argument to specify which GPU devices a PyTorch component will use.

    Examples::

        import elit
        # Load component onto the 0-th GPU.
        elit.load(..., devices=0)
        # Load component onto the 0-th and 1-st GPUs using data parallelization.
        elit.load(..., devices=[0, 1])

    .. Note::
        A component can have dependencies on other components or resources, which will be recursively loaded. So it's
        common to see multiple downloading messages per single load.

    Returns:
      elit.common.component.Component: A pretrained component.

    """
    save_dir = elit.pretrained.ALL.get(save_dir, save_dir)
    from elit.utils.component_util import load_from_meta_file
    if verbose is None:
        from hanlp_common.constant import HANLP_VERBOSE
        verbose = HANLP_VERBOSE
    return load_from_meta_file(save_dir, 'meta.json', verbose=verbose, **kwargs)


def pipeline(*pipes) -> elit.components.pipeline.Pipeline:
    """Creates a pipeline of components. It's made for bundling `KerasComponents`. For `TorchComponent`, use
    :class:`~elit.components.mtl.multi_task_learning.MultiTaskLearning` instead.

    Args:
      *pipes: Components if pre-defined any.

    Returns:
      hanlp.components.pipeline.Pipeline: A pipeline, which is a list of components in order.

    """
    return elit.components.pipeline.Pipeline(*pipes)
