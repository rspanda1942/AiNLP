# -*- coding: utf-8 -*-
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Model(object):
    """Base class for models."""

    def __init__(self,
                 name):

        self.name = name

    def __call__(self, inputs, vocab, FLAGS):
        """Calls the model function.
        Returns:
          outputs: The model outputs (usually unscaled probabilities).
            Optional if :obj:`mode` is ``tf.estimator.ModeKeys.PREDICT``.
          predictions: The model predictions.
            Optional if :obj:`mode` is ``tf.estimator.ModeKeys.TRAIN``.
        See Also:
          ``tf.estimator.Estimator`` 's ``model_fn`` argument for more details about
          the arguments of this function.
        """
        return self._build(inputs, vocab, FLAGS)

    @abc.abstractmethod
    def _build(self, inputs, vocab, FLAGS):
        """Creates the graph.
        Returns:
          outputs: The model outputs (usually unscaled probabilities).
            Optional if :obj:`mode` is ``tf.estimator.ModeKeys.PREDICT``.
          predictions: The model predictions.
            Optional if :obj:`mode` is ``tf.estimator.ModeKeys.TRAIN``.
        """
        raise NotImplementedError()
