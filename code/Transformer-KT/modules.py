# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:modules.py
@time:2019/8/911:18 AM
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import six


class EmbeddingShareWeights(tf.layers.Layer):
    """
    Calculates input embeddings
    """
    def __init__(self, vocab_size, hidden_size):
        """
        Specify characteristic parameters of embedding layer
        Args:
            vocab_size: Number of tokens in the embedding
            hidden_size: Dimensionality of the embedding.
        """
        super(EmbeddingShareWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, _):
        with tf.variable_scope("embedding_n_softmax", reuse=tf.AUTO_REUSE):
            """
            Create the initialize weights
            """
            self.shared_weight = tf.get_variable(
                name="weights", shape=[self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=self.hidden_size ** -0.5))
        self.built = True

    def call(self, x):
        """
        Get token embeddings of x
        Args:
            x:  An int64 tensor with shape [batch, length]
        Returns:
            embeddings: float32. Tensor with shape [batch, length, embedding_szie]
            padding: float32. Tensor with shape [batch, length] indicating the locations of the padding tokens in x.
        """
        with tf.name_scope("embeddings"):
            mask = tf.to_float(tf.not_equal(x, 0))
            embeddings = tf.gather(self.shared_weight, tf.cast(x, tf.int64))
            embeddings *= tf.expand_dims(mask, -1)

        embeddings *= self.hidden_size ** 0.5   # scale embedding by the sqrt of the hidden size
        return embeddings


def position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """
    Calculate the position encoding as a mix of sine and cosine function with geometrically
    increasing wavelengths.
    Defined and formulized in Attention is all your need.
    Args:
        length:  sequence lenght
        hidden_size: size of the embedding
        min_timescale: Minimum scale that will be applied at each position
        max_timescale: Maximum scale that will be applied at each position

    Returns:
        Tensor with shape [length, hidden_size]

    """
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) /
                               (tf.cast(num_timescales, tf.float32) -1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)  # shape= [hidden_size/2]
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)  # shape= [length, hidden_size/2]
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)  # shape= [length, hidden_size]

    return signal


class Attention(tf.layers.Layer):
    """Multi-headed attention layer"""
    def __init__(self, hidden_size, num_head, attention_dropout, train):
        if hidden_size % num_head != 0:
            raise ValueError("Hidden_size must be evenly divisible by the number of heads")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.attention_dropout = attention_dropout
        self.train = train

        # Layer for linear projecting query, key, values:
        self.q_dense_layer = tf.layers.Dense(units=hidden_size, use_bias=False, name='q')
        self.k_dense_layer = tf.layers.Dense(units=hidden_size, use_bias=False, name='k')
        self.v_dense_layer = tf.layers.Dense(units=hidden_size, use_bias=False, name='v')

        self.output_dense_layer = tf.layers.Dense(units=hidden_size, use_bias=False, name='output_transformer')

    def split_heads(self, x):
        """
        Split x into different heads, and transpose the resulting value
        Args:
            x: A tensor with shape [batch, length, hidden_size]

        Returns:
            A tensor with shape[batch, num_head, length, hidden_size]
        """
        with tf.name_scope("split_heads"):
            batch_size, length = tf.shape(x)[0], tf.shape(x)[1]

            depth = (self.hidden_size // self.num_head)

            x = tf.reshape(x, [batch_size, length, self.num_head, depth])

            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """
        combine tensor that has been split
        Args:
            x:  A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]

        Returns:
            A tensor with shape [batch, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size, length = tf.shape(x)[0], tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, hidden_size/num_heads]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, cache=None):
        """
        Apply attention mechanism to x and y
        Args:
            x: A tensor with shape [batch, length_x, hidden_size]
            y: A tensor with shape [batch, length_y, hidden_size]
            bias: attention bias that will be add to the result of the dot product.
            cache: (Used during prediction) dictionary with tensor containing results of previus attentions.
                The dictionary must have the items:
                {'k': tensor with shape [batch, i, key_channels],
                 'v': tensor with shape [batch, i, value_channels]}
                 where i is the current decoded length

        Returns:
            Attention layer output with shape [batch, length_x, hidden_size]

        """
        length = tf.shape(x)[1]
        q = self.q_dense_layer(x)  # [batch, length, hidden_size]
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # combine cached keys and values with new keys and values
            k = tf.concat([cache['k'], k], axis=1)
            v = tf.concat([cache['v'], v], axis=1)

            # update cache:
            cache['k'] = k
            cache['v'] = v

        # split q,k,v into heads:
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)  # [batch_size, length, num_head, hidden_size//num_head]

        # scale q to prevent dot product between q and k from growing too large
        depth = self.hidden_size // self.num_head
        q *= depth ** -0.5

        # calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)

        # add mask to prevent future words
        mask = create_look_ahead_mask(length)
        logits += mask
        weight = tf.nn.softmax(logits, name='attention_weigths')
        if self.train:
            weight = tf.nn.dropout(weight, keep_prob=1-self.attention_dropout)
        attention_output = tf.matmul(weight, v)

        # Recombine heads --> [batch, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined output through another linear projection layers
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """
    multihead self-attention layer.
    """
    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)


class FeedForwardNetwork(tf.layers.Layer):
    """Fully connected feedforward network"""
    def __init__(self, hidden_size, filter_size, relu_dropout, train, allow_pad):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train
        self.allow_pad = allow_pad

        self.filter_dense_layer = tf.layers.Dense(filter_size, use_bias=True, activation=tf.nn.relu,
                                                  name='filter_layer')
        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=True, name='output_layer')

    def call(self, x, padding=None):
        """
        Return outputs of the feedforward network"
        Args:
            x: Tensor with shape [batch_size, length, hidden_size]
            padding: Optional, if set, the padding values are temporatily removed from x.
                The padding values are placed back in the output tensor in the same locations.
                shape [batch, length]
        Returns:
            Output of the feed forward network
            shape [batch, length, hidden_size]
        """
        padding = None if not self.allow_pad else padding
        batch_size, length = tf.shape(x)[0], tf.shape(x)[1]

        if padding is not None:
            with tf.name_scope('remove_padding'):
                pad_mask = tf.reshape(padding, [-1, self.hidden_size])

                nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

                # reshape x to [batch*length, hidden_size] to remove padding
                x = tf.reshape(x, shape=[-1, self.hidden_size])
                x = tf.gather_nd(x, indices=nonpad_ids)

                # Reshape x from 2 dimensions to 3 dimensions
                x.set_shape([None, self.hidden_size])
                x = tf.expand_dims(x, axis=0)

        output = self.filter_dense_layer(x)
        if self.train:
            output = tf.nn.dropout(output, keep_prob=1-self.relu_dropout)
        output = self.output_dense_layer(output)

        if padding is not None:
            with tf.name_scope('re_add_padding'):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(indices=nonpad_ids, updates=output,
                                       shape=[batch_size*length, self.hidden_size])
                output = tf.reshape(output, [batch_size, length, self.hidden_size])
        return output


class LayerNormalization(tf.layers.Layer):
    """
    Apply layer normalization
    """
    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable(name='layer_norm_scale',
                                     shape=[self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable(name='layer_norm_bias',
                                    shape=[self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.build = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.sqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


def get_padding(x, padding_value=0, dtype=tf.float32):
    """

    Args:
        x: int tensor with any shape
        padding_value: int value which padding value set
        dtype: The dtype of the return value

    Returns:
    float tensor with same shape as x containing value 0,1
    0 means non-padding, 1 means padding

    """
    with tf.name_scope('padding'):
        return tf.cast(tf.equal(x, padding_value), dtype)


def get_padding_bias(x):
    """
    calculate bias tensot from padding values in tensor

    bias tensor that is added to the pre-softmax multi-head attention logits,
    which has shape [batch_size, num_heads, length, length]
    The tensor is zero at non-padding locations, and -1e9(negtive infinity) at padding locations
    Args:
        x:int tensor with shape [batch_size, length]
    Returns:
        Attention bias tensor of shape [batch_size, 1, 1, length]

    """
    with tf.name_scope('attention_bias'):
        padding = get_padding(x)
        attention_bias = padding * -1e9
        attention_bias = tf.expand_dims(attention_bias, axis=1)
    return attention_bias


def create_look_ahead_mask(length, dtype=tf.float32):

    """Calculate bias for decoder that maintains model's autoregressive property.
    Creates a tensor that masks out locations that correspond to illegal
    connections, so prediction at position i cannot draw information from future
    positions.
    Args:
      length: int length of sequences in batch.
      dtype: The dtype of the return value.
    Returns:
      float tensor of shape [1, 1, length, length]
    """
    neg_inf = -1e9
    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=dtype),
                                         -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = neg_inf * (1.0 - valid_locs)
    return decoder_bias


