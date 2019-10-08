# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:model.py
@time:2019/10/86:24 PM
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from modules import TemporalConvNet


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
        num_channels = [self.params['hidden_size'] * self.params['hidden_layer_num']]
        self.tcn = TemporalConvNet(num_channels,
                                   stride=1,
                                   kernel_size=self.params['kernel_size'],
                                   dropout=1-self.params['keep_prob'])

    def __call__(self, inputs):

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
        tcn_output = self.tcn(self.embedding_output)
        logits = tf.layers.dense(tcn_output, self.params['num_skills'], activation=None, use_bias=True)
        return logits