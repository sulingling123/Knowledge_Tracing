# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:utils.py
@time:2019/8/125:04 PM
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import six
import re

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


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """calculate learning rate with linear warmup and rsqrt decay"""
    with tf.name_scope('learning_rate'):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size ** -0.5)
        # Apply linear warmup
        learning_rate *= tf.minimum(1.0, step/warmup_steps)
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

        tf.identity(learning_rate, "learning_rate")
        return learning_rate


def get_train_op_and_metrics(loss, params):
    """Generate training op and metrics to save in tensorboard"""
    with tf.variable_scope('get_train_op'):
        learning_rate = get_learning_rate(
            learning_rate=params['learning_rate'],
            hidden_size=params['hidden_size'],
            learning_rate_warmup_steps=params['learning_rate_warmup_steps']
        )
        # create optimizer , Use lazyAdamOptimizer from TF contrib,which is faster than the TF core Adam operation
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate=learning_rate,
            beta1=params['optimizer_adam_beta1'],
            beta2=params['optimizer_adam_beta2'],
            epsilon=params['optimizer_adam_epsilon']
        )
        # calculate and apply graph gradient using LazyAdamOptimizer
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        gradients = optimizer.compute_gradients(loss, tvars, colocate_gradients_with_ops=True)
        minimize_op = optimizer.apply_gradients(gradients,
                                                global_step=global_step,
                                                name='train')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

        train_metrics = {'learning_rate': learning_rate,
                         'global_step': global_step}
        return train_op, train_metrics


def record_scalars(metric_dict):
    for key, value in metric_dict.items():
        print('records_scalars', key)
        if key == 'accuracy':
            tf.summary.scalar(name=key, tensor=value[1])
        else:
            tf.summary.scalar(name=key, tensor=value)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def parse_exmp(serial_exmp):
    feats = tf.parse_single_example(serial_exmp, features={'inputs': tf.VarLenFeature(tf.float32),
                                                           'target_correct': tf.VarLenFeature(tf.int64),
                                                           'target_id': tf.VarLenFeature(tf.float32),

                                                           'correct': tf.VarLenFeature(tf.int64),
                                                           'id': tf.VarLenFeature(tf.float32),

                                                           'seq_len': tf.FixedLenFeature([], tf.int64)})
    inputs = tf.sparse_tensor_to_dense(feats['inputs'])  # 使用VarLenFeature读入的是一个sparse_tensor，用该函数进行转换
    target_correct = tf.sparse_tensor_to_dense(feats['target_correct'])
    target_id = tf.sparse_tensor_to_dense(feats['target_id'])

    correct = tf.sparse_tensor_to_dense(feats['correct'])
    id = tf.sparse_tensor_to_dense(feats['id'])

    inputs = tf.cast(inputs, tf.int32)
    target_correct = tf.cast(target_correct, tf.float32)
    target_id = tf.cast(target_id, tf.int32)

    correct = tf.cast(correct, tf.float32)
    id = tf.cast(id, tf.int32)

    seq_len = tf.cast(feats['seq_len'], tf.int32)

    return {'inputs': inputs,
            'target_correct': target_correct,
            'target_id': target_id,
            'ids': id,
            'correct': correct,
            'seq_len': seq_len}


def get_dataset(fname):
    dataset = tf.data.TFRecordDataset(fname)
    return dataset.map(parse_exmp)  # use padded_batch method if padding needed
