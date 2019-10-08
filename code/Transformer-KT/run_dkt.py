# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:run_dkt.py
@time:2019/8/1212:39 PM
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from model_params import BASE_PARAMS
from model import *
from metrics import *
from utils import *



flags = tf.flags
logging = tf.logging

flags.DEFINE_string(name='train_data', default='../data/train.tfrecord',
                    help='Path of training data ')
flags.DEFINE_string(name='valid_data', default='../data/eval.tfrecord',
                    help='Path of valid data')
flags.DEFINE_string(name='predict_data', default='',
                    help='Path of predict data')
flags.DEFINE_string(name='saved_model', default='./saved_model', help='Path to save model')
flags.DEFINE_string(name='output_dir', default='./output_dir', help='Path to save model')
flags.DEFINE_integer(name='epoch', default=100, help='Num of epoch')
flags.DEFINE_integer(name='batch_size', default=1, help='Num of batch')
flags.DEFINE_bool(name="do_train", default=True, help="Whether to run training.")
flags.DEFINE_bool(name="do_eval", default=True, help="Whether to run eval on the dev set.")
flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")
flags.DEFINE_bool(name='pb', default=True, help='')
FLAGS = flags.FLAGS


def model_fn_builder():
    def model_fn(features, labels, mode, params):
        """
        define how to train, evaluate and predict from the transfomer model.
        Args:

            mode:
            params:

        Returns:

        """
        inputs = features['inputs']
        target_ids = features['target_id']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        try:
            batch_size, length = get_shape_list(inputs, expected_rank=2)
        except ValueError:
            batch_size = 1
            length = get_shape_list(inputs, expected_rank=1)[0]
            inputs = tf.reshape(inputs, [batch_size, length])

        with tf.variable_scope('model'):
            # Build model
            model = DKT(params, is_training)
            logits = model(inputs, target_ids)  # [batch, length, vocab_size]

            # when in prediction mode, the label/target is Bone, the model output is the prediction
            if mode == tf.estimator.ModeKeys.PREDICT:
                export_outputs = {'predict_output': tf.estimator.export.PredictOutput({"predict": tf.sigmoid(logits)})}
                output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                         predictions={'predict': tf.sigmoid(logits)},
                                                         export_outputs=export_outputs
                                                        )
            else:
                # Calculate model loss
                seq_steps = features['seq_len']
                target_correct = features['target_correct']

                loss = dkt_loss(logits, target_correct, target_ids, seq_steps)
                record_dict = {}
                record_dict['minibatch_loss'] = loss
                 # Save loss as named tensor will be logged with the logging hook
                tf.identity(loss, 'cross_entropy')

                if mode == tf.estimator.ModeKeys.EVAL:
                    metric_dict = get_eval_metrics(logits, target_correct, target_ids, seq_steps)
                    record_dict['accuracy'] = metric_dict['accuracy']
                    record_scalars(record_dict)
                    output_spec = tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                                             loss=loss,
                                                             predictions={'predict': tf.sigmoid(logits)},
                                                             eval_metric_ops=metric_dict )
                else:  # train
                    # check whether restore from checkpoint
                    tvars = tf.trainable_variables()
                    initialized_variable_names = {}

                    tf.logging.info("**** Trainable Variables ****")
                    for var in tvars:
                        init_string = ""
                        if var.name in initialized_variable_names:
                            init_string = ", *INIT_FROM_CKPT*"
                        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                        init_string)

                    train_op, metric_dict = get_train_op_and_metrics(loss, params)
                    acc_metric = get_eval_metrics(logits, target_correct, target_ids, seq_steps)
                    record_dict['accuracy'] = acc_metric['accuracy']
                    record_dict['learning_rate'] = metric_dict['learning_rate']
                    record_scalars(record_dict)
                    output_spec = tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                                             loss=loss,
                                                             train_op=train_op
                                                            )
        return output_spec
    return model_fn


def input_fn_builder(datadir, epoch):
    def input_fn():
        padded_shapes = {'inputs': [None],
                         'target_correct': [None],
                         'target_id': [None],
                         'ids': [None],
                         'correct': [None],

                         'seq_len': [],
                       }

        dataset_train = get_dataset(datadir)
        dataset_train = dataset_train.repeat(epoch).shuffle(1000).padded_batch(FLAGS.batch_size,
                                                                                     padded_shapes=padded_shapes)

        return dataset_train

    return input_fn


def main():
    # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    #     raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    params = BASE_PARAMS  # param dict from model_params.py
    tf.gfile.MakeDirs(FLAGS.saved_model)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.saved_model,
        tf_random_seed=tf.set_random_seed([42]),
        # save_summary_step=100,
        save_checkpoints_steps=1000,
        # save_checkpoints_secs=600,
        session_config=None,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=100,
        train_distribute=None
    )
    model_fn = model_fn_builder()
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.saved_model,
        config=run_config,
        params=params,
        warm_start_from=None
    )
    # train_input_fn = input_fn_builder(FLAGS.train_data, FLAGS.epoch)
    # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
    #                                     max_steps=None)
    # eval_input_fn = input_fn_builder(FLAGS.valid_data, 1)
    # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
    #                                   steps=None,
    #                                   throttle_secs=60)
    # tf.estimator.train_and_evaluate(estimator,
    #                                 train_spec=train_spec,
    #                                 eval_spec=eval_spec)
    for epoch in range(FLAGS.epoch):
        if FLAGS.do_train:
            train_input_fn = input_fn_builder(FLAGS.train_data, 1)

            estimator.train(input_fn=train_input_fn)
        if FLAGS.do_eval:
            eval_input_fn = input_fn_builder(FLAGS.valid_data, 1)
            result = estimator.evaluate(input_fn=eval_input_fn)

            output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w+") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # flags.mark_flag_as_required("train_data")
    # flags.mark_flag_as_required("valid_data")
    # flags.mark_flag_as_required("saved_model")

    main()
