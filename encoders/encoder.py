# -*- coding: utf-8 -*-
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Encoder(object):
    """Base class for encoders."""

    @abc.abstractmethod
    def encode(self, inputs, sequence_length=None, mode=None):
        """Encodes an input sequence.
        Args:
          inputs: The inputs to encode of shape :math:`[B, T, ...]`.
          sequence_length: The length of each input with shape :math:`[B]`.
          mode: A ``tf.estimator.ModeKeys`` mode.
        Returns:
          A tuple ``(outputs, state, sequence_length)``.
        """
        raise NotImplementedError()
