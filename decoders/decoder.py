# -*- coding: utf-8 -*-
import abc
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class Decoder(object):
    """Base class for decoders."""

    @abc.abstractmethod
    def decode(self,
               inputs,
               sequence_length,
               vocab_size=None,
               initial_state=None,
               sampling_probability=None,
               embedding=None,
               output_layer=None,
               mode=tf.estimator.ModeKeys.TRAIN,
               memory=None,
               memory_sequence_length=None):
        """Decodes a full input sequence.
        Usually used for training and evaluation where target sequences are known.
        Args:
          inputs: The input to decode of shape :math:`[B, T, ...]`.
          sequence_length: The length of each input with shape :math:`[B]`.
          vocab_size: The output vocabulary size. Must be set if :obj:`output_layer`
            is not set.
          initial_state: The initial state as a (possibly nested tuple of...) tensors.
          sampling_probability: The probability of sampling categorically from
            the output ids instead of reading directly from the inputs.
          embedding: The embedding tensor or a callable that takes word ids.
            Must be set when :obj:`sampling_probability` is set.
          output_layer: Optional layer to apply to the output prior sampling.
            Must be set if :obj:`vocab_size` is not set.
          mode: A ``tf.estimator.ModeKeys`` mode.
          memory: (optional) Memory values to query.
          memory_sequence_length: (optional) Memory values length.
        Returns:
          A tuple ``(outputs, state, sequence_length)``.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def dynamic_decode(self,
                       embedding,
                       start_tokens,
                       end_token,
                       vocab_size=None,
                       initial_state=None,
                       output_layer=None,
                       maximum_iterations=250,
                       mode=tf.estimator.ModeKeys.PREDICT,
                       memory=None,
                       memory_sequence_length=None,
                       dtype=None,
                       return_alignment_history=False):
        """Decodes dynamically from :obj:`start_tokens` with greedy search.
        Usually used for inference.
        Args:
          embedding: The embedding tensor or a callable that takes word ids.
          start_tokens: The start token ids with shape :math:`[B]`.
          end_token: The end token id.
          vocab_size: The output vocabulary size. Must be set if :obj:`output_layer`
            is not set.
          initial_state: The initial state as a (possibly nested tuple of...) tensors.
          output_layer: Optional layer to apply to the output prior sampling.
            Must be set if :obj:`vocab_size` is not set.
          maximum_iterations: The maximum number of decoding iterations.
          mode: A ``tf.estimator.ModeKeys`` mode.
          memory: (optional) Memory values to query.
          memory_sequence_length: (optional) Memory values length.
          dtype: The data type. Required if :obj:`memory` is ``None``.
          return_alignment_history: If ``True``, also returns the alignment
            history from the attention layer (``None`` will be returned if
            unsupported by the decoder).
        Returns:
          A tuple ``(predicted_ids, state, sequence_length, log_probs)`` or
          ``(predicted_ids, state, sequence_length, log_probs, alignment_history)``
          if :obj:`return_alignment_history` is ``True``.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def dynamic_decode_and_search(self,
                                  embedding,
                                  start_tokens,
                                  end_token,
                                  vocab_size=None,
                                  initial_state=None,
                                  output_layer=None,
                                  beam_width=5,
                                  length_penalty=0.0,
                                  maximum_iterations=250,
                                  mode=tf.estimator.ModeKeys.PREDICT,
                                  memory=None,
                                  memory_sequence_length=None,
                                  dtype=None,
                                  return_alignment_history=False):
        """Decodes dynamically from :obj:`start_tokens` with beam search.
        Usually used for inference.
        Args:
          embedding: The embedding tensor or a callable that takes word ids.
          start_tokens: The start token ids with shape :math:`[B]`.
          end_token: The end token id.
          vocab_size: The output vocabulary size. Must be set if :obj:`output_layer`
            is not set.
          initial_state: The initial state as a (possibly nested tuple of...) tensors.
          output_layer: Optional layer to apply to the output prior sampling.
            Must be set if :obj:`vocab_size` is not set.
          beam_width: The width of the beam.
          length_penalty: The length penalty weight during beam search.
          maximum_iterations: The maximum number of decoding iterations.
          mode: A ``tf.estimator.ModeKeys`` mode.
          memory: (optional) Memory values to query.
          memory_sequence_length: (optional) Memory values length.
          dtype: The data type. Required if :obj:`memory` is ``None``.
          return_alignment_history: If ``True``, also returns the alignment
            history from the attention layer (``None`` will be returned if
            unsupported by the decoder).
        Returns:
          A tuple ``(predicted_ids, state, sequence_length, log_probs)`` or
          ``(predicted_ids, state, sequence_length, log_probs, alignment_history)``
          if :obj:`return_alignment_history` is ``True``.
        """
        raise NotImplementedError()
