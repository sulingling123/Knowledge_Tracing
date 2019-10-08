# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:metrics.py
@time:2019/8/122:57 PM
"""
import tensorflow as tf
from model_params import  *
from model import DKT


def dkt_loss(logits, target_correct, target_ids, correct, ids,  seq_steps):
    """
    calculate cross-entropy loss for dkt
    Args:
        logits: logits(no sigmoid func) with shape [batch_size, length, vocab_size]
        target_correct:  targets with shape [batch_size, length]
        target_ids:
        seq_steps:

    Returns:
        cross-entropy loss

    """
    with tf.name_scope('loss'):
        batch_size, length, vocab_size = tf.shape(logits)[0], tf.shape(logits)[1], tf.shape(logits)[2]
        flat_logits = tf.reshape(logits, [-1])
        flat_target_correct = tf.reshape(target_correct, [-1])

        flat_correctness = tf.reshape(correct, [-1])

        flat_base_target_index = tf.range(batch_size * length) * vocab_size
        flat_bias_target_id = tf.reshape(target_ids, [-1])
        flat_target_id = flat_base_target_index + flat_bias_target_id

        flat_target_logits = tf.gather(flat_logits, flat_target_id)
        mask = tf.reshape(tf.sequence_mask(seq_steps, maxlen=length), [-1])
        # drop predict which user not react to
        flat_target_correct_mask = tf.boolean_mask(flat_target_correct, mask)
        flat_target_logits_mask = tf.boolean_mask(flat_target_logits, mask)

        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=flat_target_correct_mask,
            logits=flat_target_logits_mask,
            pos_weight=1
        ))

        r = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=flat_correctness,
                                                                    logits=flat_target_logits,
                                                                    pos_weight=1))
        pred_all = tf.sigmoid(logits)
        w1 = tf.reduce_mean(tf.abs(pred_all[:, :-1, :] - pred_all[:, 1:, :]))
        w2 = tf.reduce_mean(tf.square(pred_all[:, :-1, :] - pred_all[:, 1:, :]))
        loss += 0.5 * r + 0.1 * w1 + 0.1 * w2
        return loss


def padded_accuracy(logits, target, target_ids, seq_len):
    """
    Percentage of times that predictions matches albels on non-0s
    Args:
        logits: Tensor with shape [batch_size, length, vocab_size]
        target: Tesnor wtth shape [batch_size, length]

    Returns:

    """
    with tf.variable_scope('padded_accuracy', values=[logits, target, target_ids, seq_len]):
        batch_size, length, vocab_size = tf.shape(logits)[0], tf.shape(logits)[1], tf.shape(logits)[2]
        flat_logits = tf.reshape(logits, [-1])
        flat_target_correct = tf.reshape(target, [-1])

        flat_base_target_index = tf.range(batch_size * length) * vocab_size
        flat_bias_target_id = tf.reshape(target_ids, [-1])
        flat_target_id = flat_base_target_index + flat_bias_target_id

        flat_target_logits = tf.gather(flat_logits, flat_target_id)
        pred = tf.sigmoid(tf.reshape(flat_target_logits, [batch_size, length]))
        # self.binary_pred = tf.cast(tf.greater_equal(self.pred, 0.5), tf.int32)
        binary_pred = tf.cast(tf.greater(pred, 0.5), tf.float32)

        predict = tf.reshape(binary_pred, [-1])
        mask = tf.reshape(tf.sequence_mask(seq_len, maxlen=tf.shape(logits)[1]), [-1])
        flat_predict_mask = tf.boolean_mask(predict, mask)
        flat_target_mask = tf.boolean_mask(flat_target_correct, mask)
    return tf.equal(flat_target_mask, flat_predict_mask)


def _convert_to_eval_metric(metric_fn):
    """
    warper a metric_fn that returns scores and weights as an eval metric fn
    The input metric_fn returns values for the current batch.
    The wrapper aggregates the return values collected over all of the batches evaluated
    Args:
        metric_fn: function that return socres  for current batch's logits and targets

    Returns:
        function that aggregates the score and weights from metric_fn

    """
    def problem_metric_fn(*args):
        """
        return an aggregation of the metric_fn's returned values
        Args:
            *args:

        Returns:

        """
        score = metric_fn(*args)
        return tf.metrics.mean(score)
    return problem_metric_fn


def get_eval_metrics(logits, labels, target_ids, seq_len):
    """
    return dictionary of model evaluation metrics
    Args:
        logits:
        labels:
        params:

    Returns:

    """
    metrics = {
        'accuracy': _convert_to_eval_metric(padded_accuracy)(logits, labels, target_ids, seq_len),
    }
    return metrics