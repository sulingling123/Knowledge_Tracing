# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:model.py
@time:2019/8/95:18 PM
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from modules import *
import six


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


class DKT(object):
    def __init__(self, params, train):
        """
        Initialize layers to build DKT model
        Args:
            params: hyperparameter object defining layer size, dropout value etc
            train: boolean indicating whether the model is in training mode
        """
        self.train = train
        self.params = params



        self.inputs_embedding = EmbeddingShareWeights(vocab_size= 2*params['vocab_size'],
                                                      hidden_size=params['hidden_size'])
        self.target_idx_embedding = EmbeddingShareWeights(vocab_size=params['vocab_size'],
                                                          hidden_size=params['hidden_size'])
        self.encoder = EncoderStack(self.params, self.train)

    def __call__(self, inputs, target_ids):
        """
        Calculate logits or inferred target sequence
        Args:
            self:
            inputs: int tensor with shape [batch, input_length]  question & reaction encoding
            target_ids: int tensor with shape [batch, input_length] question encoding

            input_padding:

        Returns:

        """
        input_shape = get_shape_list(inputs, expected_rank=2)
        batch_size = input_shape[0]
        length = input_shape[1]

        target_ids = tf.reshape(target_ids, [batch_size, length])
        initializer = tf.variance_scaling_initializer(
            self.params['initializer_gain'],
            mode="fan_avg",
            distribution="uniform")
        with tf.variable_scope('DKT', initializer=initializer):
            inputs_embeddings = self.inputs_embedding(inputs)   # shape = [batch, length, hidden_size-feature_size]

            target_ids_embeddings = self.target_idx_embedding(target_ids)  # shape=[batch, length, hidden_size]

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(inputs_embeddings)[1]
                pos_encoding = position_encoding(length, self.params['hidden_size'])
                encoder_key = inputs_embeddings + pos_encoding
                encoder_query = target_ids_embeddings + pos_encoding

            if self.train:
                encoder_key = tf.nn.dropout(encoder_key, keep_prob=1-self.params['layer_postprocess_dropout'])
                encoder_query = tf.nn.dropout(encoder_query, keep_prob=1-self.params['layer_postprocess_dropout'])
            attention_bias = get_padding_bias(encoder_key)
            inputs_padding = get_padding(encoder_key)
            # shape=[batch, length, hidden_size]
            transformer_output = self.encoder(encoder_key, encoder_query=encoder_query, encoder_key=encoder_key,
                                                  attention_bias=attention_bias, inputs_padding=inputs_padding)

            with tf.name_scope("Outout_layer"):
                # shape=[batch, length, vocab_size]
                logits = tf.layers.dense(transformer_output, units=self.params['vocab_size'], activation=None)  # linear layer

        return logits


class PrepostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing"""

    def __init__(self, layer, params, train):
        self.layer = layer
        self.postprocess_dropout = params['layer_postprocess_dropout']
        self.train = train

        self.layer_norm = LayerNormalization(params['hidden_size'])

    def __call__(self, x, *args, **kwargs):
        y = self.layer_norm(x)

        y = self.layer(y, *args, **kwargs)
        if self.train:
            y = tf.nn.dropout(y, keep_prob=1 - self.postprocess_dropout)
        return x + y


class EncoderStack(tf.layers.Layer):
    """Transfomer encoder stack"""
    def __init__(self, params, train):
        super(EncoderStack, self).__init__()
        self.layers = []

        for i in range(params['num_hidden_layers']):
            # create sublayers for each layer
            self_attention_layer = Attention(hidden_size=params['hidden_size'],
                                         num_head=params['num_heads'],
                                         attention_dropout=params['attention_dropout'],
                                         train=train)
            feed_forward_netword = FeedForwardNetwork(hidden_size=params['hidden_size'],
                                      filter_size=params['filter_size'],
                                      relu_dropout=params['relu_dropout'],
                                      train=train,
                                      allow_pad=params['allow_ffn_pad'])
            self.layers.append([
                PrepostProcessingWrapper(self_attention_layer, params, train),
                PrepostProcessingWrapper(feed_forward_netword, params, train)
            ])

        self.output_normalization = LayerNormalization(params['hidden_size'])

    def call(self, inputs, encoder_query, encoder_key, attention_bias, inputs_padding):
        """
        Return the output of the encoder of layer stacks
        Args:
            encoder_query: tensor with shape [batch_size, input_length, hidden_size]  query
            encoder_key: tensor with shape [batch_size, input_length, hidden_size]  key & value
            attention_bias: bias for encoder self-attention layer [batch, 1, 1, input_length]
            input_padding: Padding

        Returns:
            output of encoder layer stack.
            float 32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.variable_scope("layer_%d" %n):
                with tf.variable_scope("self_attention"):
                    if n == 0:
                        encoder_inputs = self_attention_layer(x=encoder_query, y=encoder_key, bias=attention_bias)
                    else:
                        encoder_inputs = self_attention_layer(x=encoder_inputs, y=encoder_inputs, bias=attention_bias)
                with tf.variable_scope("ffn"):
                    encoder_inputs = feed_forward_network(encoder_inputs, padding=None)
        return self.output_normalization(encoder_inputs)
