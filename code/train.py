# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:train.py
@time:2019/2/19下午4:24
"""
import tensorflow as tf
import numpy as np
import datetime
import time
import os

from model.rnn_dkt import DKT_LSTM
from utils.config import get_config
from utils.data_generator import *


flags = tf.flags
flags.DEFINE_string('train_dir', './data/2016-EDM/0910_b_train.csv',
                    'The directory of train data')
flags.DEFINE_string('eval_dir', './data/2016-EDM/0910_b_test.csv',
                    'The directory of eval data')
flags.DEFINE_integer('epoch', 20, 'number of epoch')

FLAGS = flags.FLAGS


def train(config):
    g = tf.Graph()
    with g.as_default() as g:
        with tf.device('/CPU:0'):
            processor = EDM_Processor()
            inputs, target_id, target_correct, seq_len = processor.get_train_examples(FLAGS.train_dir)
            config.num_skills = processor.num_skill  # 调用完get_train_examples后num_skill才有值
            inputs_dev, target_id_dev, target_correct_dev, seq_len_dev = processor.get_train_examples(FLAGS.eval_dir)
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num examples = %d", len(inputs))
            tf.logging.info("  Batch size = %d", config.batch_size)
            tf.logging.info("  Saved path: %s" % config.save_path)
            tf.logging.info('  Num_layers = %s ' % config.hidden_layer_num)
            tf.logging.info('  Hidden size = %s' % config.hidden_size)
            tf.logging.info('  Num_skill = %s' % config.num_skills)
            tf.logging.info('  Learning_rate = %s' % config.learning_rate)
            tf.logging.info('  Keep_prob = %s' % config.keep_prob)

            tf.logging.info('train data_dir is %s' % FLAGS.train_dir)

            # train_data
            def train_generator():
                for sub_inputs, sub_target_id, sub_target_correct, sub_seq_len in zip(inputs, target_id, target_correct,
                                                                                      seq_len):
                    yield {'inputs': sub_inputs,
                           'target_id': sub_target_id,
                           'target_correct': sub_target_correct,
                           'seq_len': sub_seq_len}

            train_dataset = tf.data.Dataset.from_generator(generator=train_generator,
                                                           output_types={'inputs': tf.int32,
                                                                         'target_id': tf.int32,
                                                                         'target_correct': tf.float32,
                                                                         'seq_len': tf.int32})

            padded_shapes = {'inputs': [None],
                             'target_id': [None],
                             'target_correct': [None],
                             'seq_len': []}
            train_dataset = train_dataset.repeat(1).shuffle(1000).padded_batch(config.batch_size,
                                                                               padded_shapes=padded_shapes)

            train_iterator = train_dataset.make_initializable_iterator()
            train_data = train_iterator.get_next()

            # ---------------------------------- eval_data -----------------------------
            def eval_generator():
                for sub_inputs_dev, sub_target_id_dev, sub_target_correct_dev, sub_seq_len_dev in zip(inputs_dev,
                                                                                                      target_id_dev,
                                                                                                      target_correct_dev,
                                                                                                      seq_len_dev):
                    yield {'inputs': sub_inputs_dev,
                           'target_id': sub_target_id_dev,
                           'target_correct': sub_target_correct_dev,
                           'seq_len': sub_seq_len_dev}

            eval_dataset = tf.data.Dataset.from_generator(generator=eval_generator,
                                                          output_types={'inputs': tf.int32,
                                                                        'target_id': tf.int32,
                                                                        'target_correct': tf.float32,
                                                                        'seq_len': tf.int32})
            eval_dataset = eval_dataset.repeat(1).shuffle(1000).padded_batch(1000, padded_shapes=padded_shapes)

            eval_iterator = eval_dataset.make_initializable_iterator()
            eval_data = eval_iterator.get_next()
        #  ======Define model ======
        model = DKT_LSTM(config, is_training=True)
        train_op, loss, global_step = model.optimizer()
        accuracy = model.accuracy()
        soft_placement = False
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement,
                                      device_count={"CPU": config.CPU_NUMS},
                                      intra_op_parallelism_threads=config.CPU_NUMS,
                                      inter_op_parallelism_threads=config.CPU_NUMS)
        config_proto.gpu_options.allow_growth = True

        saver = tf.train.Saver(max_to_keep=4)
        with tf.Session(graph=g, config=config_proto) as sess:
            sess.run(tf.global_variables_initializer())
            summaries = tf.summary.merge_all()
            file_path = os.path.join('./log_tensorboard/dkt', time.strftime("%Y-%m-%d-%H-%M-%S"))
            writer = tf.summary.FileWriter(file_path)
            writer.add_graph(sess.graph)
            for epoch in range(FLAGS.epoch):
                fetches = {
                    "loss": loss,
                    "eval_op": train_op,
                    "global_step": global_step,
                    "accuracy": accuracy}
                sess.run(train_iterator.initializer)
                while True:
                    try:
                        starttime = datetime.datetime.now()
                        feed_data = sess.run(train_data)

                        batch_size, max_len = feed_data['inputs'].shape
                        initial_state = np.zeros([config.hidden_layer_num, 2, batch_size, config.hidden_size],
                                                 np.float32)
                        feed_dict = {
                            model.inputs: feed_data['inputs'],
                            model.target_id: feed_data['target_id'],
                            model.target_correct: feed_data['target_correct'],
                            model.seq_len: feed_data['seq_len'],
                            model.init_state: initial_state
                        }

                        summ, val = sess.run([summaries, fetches], feed_dict=feed_dict)

                        writer.add_summary(summ, val["global_step"])
                        if val["global_step"] % 1 == 0:
                            print("Epoch %d: %d training steps  | loss %.4g  |  "
                                  "training accuracy %.4g  |  epoch lr: %.4g | costs: %s"
                                  % (epoch, val["global_step"], val["loss"], val["accuracy"],
                                     model.get_lr(sess), str(datetime.datetime.now() - starttime)))
                        # saved model
                        if val["global_step"] % 50 == 0:
                            saver.save(sess, os.path.join(config.save_path, 'model.ckpt'),
                                       global_step=val["global_step"])
                    except tf.errors.OutOfRangeError:  # 一个epoch结束
                        break

                sess.run(eval_iterator.initializer)
                feed_data = sess.run(eval_data)

                batch_size, max_len = feed_data['inputs'].shape
                initial_state = np.zeros([config.hidden_layer_num, 2, batch_size, config.hidden_size], np.float32)
                feed_dict = {
                    model.inputs: feed_data['inputs'],
                    model.target_id: feed_data['target_id'],
                    model.target_correct: feed_data['target_correct'],
                    model.seq_len: feed_data['seq_len'],
                    model.init_state: initial_state
                }
                fetches = {
                    "loss": loss,
                    "accuracy": accuracy,

                }
                summ, val = sess.run([summaries, fetches], feed_dict=feed_dict)
                print("After epoch %d:  test loss %g, test accuracy %g, " % (epoch, val["loss"], val["accuracy"]))


if __name__ == '__main__':
    if not FLAGS.train_dir:
        raise ValueError("Must set --data_path to data directory")
    tf.logging.set_verbosity(tf.logging.INFO)
    starttime = datetime.datetime.now()
    config = get_config('RNN')
    config.save_path = os.path.join(config.save_path, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.mkdir(config.save_path)
    print("Data path: %s  |  num skill: %s  |  save path : %s"
          % (FLAGS.train_dir, config.num_skills, config.save_path))

    train(config)
    print("train costs %s" % str(datetime.datetime.now() - starttime))
