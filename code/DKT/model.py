# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:model.py
@time:2019/10/83:13 PM
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DKT(object):
    def __init__(self, params, train):
        """
        Initialize layers to build DKT model
        Args:
            params: hyperparameter object defining layer size, dropout value etc
            train: boolean indicating whether the model is in training mode
        """
        self.params = params
        self.train = train

    def __call__(self, batch_size, inputs, seq_len):
        initial_state = []
        if self.params['use_one_hot']:
            self.embedding_output = tf.one_hot(inputs, depth=2*self.params['num_skills'] + 1)
        else:  # Use Embedding
            embedding_table = tf.get_variable(
                name="word_embedding",
                shape=[2*self.params['num_skills']+1, self.params['hidden_size']],
                initializer=tf.truncated_normal_initializer(self.params['initializer_range']))
            self.embedding_output = tf.nn.embedding_lookup(embedding_table, inputs)
            self.embedding_output = tf.contrib.layers.layer_norm(self.embedding_output,
                                                                 begin_norm_axis=-1,
                                                                 begin_params_axis=-1, scope='embedding_ln')
            self.embedding_output = tf.nn.dropout(self.embedding_output, keep_prob=self.params['keep_prob'])

        hidden_layers = []
        final_hidden_size = self.params['hidden_size']
        with tf.variable_scope('lstm_cell', initializer=tf.orthogonal_initializer()):
            for i in range(self.params['hidden_layer_num']):
                if self.params['use_LN']:
                    hidden_layer = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=final_hidden_size,
                                                                         forget_bias=0.0,
                                                                         layer_norm=True,
                                                                         dropout_keep_prob=self.params['keep_prob'])
                else:
                    hidden_layer = tf.contrib.rnn.BasicLSTMCell(num_units=final_hidden_size, forget_bias=0,
                                                                state_is_tuple=True)
                    # lstm_layer = tf.contrib.rnn.GRUCell(num_units=final_hidden_size, state_is_tuple=True)

                    hidden_layer = tf.contrib.rnn.DropoutWrapper(cell=hidden_layer,
                                                                 output_keep_prob=self.params['keep_prob'])
                hidden_layers.append(hidden_layer)
                initial_state.append(hidden_layer.zero_state(batch_size, dtype=tf.float32))
            self.hidden_cell = tf.contrib.rnn.MultiRNNCell(cells=hidden_layers, state_is_tuple=True)

            # dynamic rnn
            state_series, self.current_state = tf.nn.dynamic_rnn(cell=self.hidden_cell,
                                                                 inputs=self.embedding_output,
                                                                 sequence_length=seq_len,
                                                                 initial_state= tuple(initial_state))

        output_w = tf.get_variable("weights", [final_hidden_size, self.params['num_skills']],
                                   initializer=tf.truncated_normal_initializer(self.params['initializer_range']),
                                   # regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization_lambda)
                                   regularizer=None
                                   )

        output_b = tf.get_variable("biases", [self.params['num_skills']],
                                   initializer=tf.zeros_initializer(),
                                   # regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization_lambda)
                                   regularizer=None
                                   )

        batch_size, max_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        self.state_series = tf.reshape(state_series, [batch_size * max_len, final_hidden_size])

        logits = tf.matmul(self.state_series, output_w) + output_b
        logits = tf.reshape(logits, [batch_size, max_len, self.params['num_skills']])
        # self.pred_all = tf.sigmoid(self.mat_logits)

        return logits