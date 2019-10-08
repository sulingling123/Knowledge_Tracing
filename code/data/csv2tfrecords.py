# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:csv2tfrecords.py
@time:2019/10/85:43 PM
"""
import numpy as np
from collections import defaultdict
import tensorflow as tf
import random
import csv


class EDM_Processor(object):
    """Processor for the MultiNLI data set (GLUE version)."""
    def __init__(self):
        self.num_skill = None
        self.max_len = None

    def _read_tsv(self, dataset_path):
        """Reads a tab separated value file."""
        rows = []
        max_skill_num = 0
        max_num_problems = 0
        with open(dataset_path, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                rows.append(row)
        index = 0
        i = 0
        print("the number of rows is " + str(len(rows)))
        tuple_rows = []
        # turn list to tuple
        while (index < len(rows) - 1):
            problems_num = int(rows[index][0])
            tmp_max_skill = max(map(int, rows[index + 1]))
            if (tmp_max_skill > max_skill_num):
                max_skill_num = tmp_max_skill
            if (problems_num <= 2):
                index += 3
            else:
                if problems_num > max_num_problems:
                    max_num_problems = problems_num
                tup = (rows[index], rows[index + 1], rows[index + 2])
                tuple_rows.append(tup)
                index += 3
        # shuffle the tuple

        random.shuffle(tuple_rows)
        print("The number of num_skill is ", max_skill_num+1)
        print("Finish reading data...")
        self.max_len, self.num_skill = max_num_problems, max_skill_num + 1
        return tuple_rows

    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_examples(
            self._read_tsv(data_dir))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir))

    # def get_test_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def _create_examples(self, tuple_rows):
        """Creates examples for the training and dev sets."""
        seq_len = []
        inputs = []
        target_correct = []
        target_id = []
        ids = []
        correct = []
        for i in range(len(tuple_rows)):
            # inputs
            inputs.append([int(tuple_rows[i][1][j]) + int(tuple_rows[i][2][j]) * self.num_skill for j in range(len(tuple_rows[i][1]) - 1)])
            seq_len.append(int(tuple_rows[i][0][0]) - 1)  # sequence
            target_id.append(list(map(lambda k: int(k), tuple_rows[i][1][1:])))
            target_correct.append(list(map(lambda k: int(k), tuple_rows[i][2][1:])))

            ids.append(list(map(lambda k: int(k), tuple_rows[i][1][:-1])))
            correct.append(list(map(lambda k: int(k), tuple_rows[i][2][:-1])))

        return np.array(inputs), np.array(target_id), np.array(target_correct), np.array(ids), np.array(correct), np.array(seq_len)


def encode_tfrecord(tfrecords_filename, inputs, target_id, target_correct, ids, correct, seq_len):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    nums = inputs.shape[0]
    print('%s records' % nums)
    for i in range(nums):
        tfrecords_features = dict()

        tfrecords_features['inputs'] = tf.train.Feature(float_list=tf.train.FloatList(value=inputs[i]))
        tfrecords_features['target_id'] = tf.train.Feature(float_list=tf.train.FloatList(value=target_id[i]))
        tfrecords_features['target_correct'] = tf.train.Feature(int64_list=tf.train.Int64List(value=target_correct[i]))
        tfrecords_features['ids'] = tf.train.Feature(float_list=tf.train.FloatList(value=ids[i]))
        tfrecords_features['correct'] = tf.train.Feature(int64_list=tf.train.Int64List(value=target_id[i]))
        tfrecords_features['seq_len'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len[i]]))
        example = tf.train.Example(features=tf.train.Features(feature=tfrecords_features))
        exmp_serial = example.SerializeToString()
        writer.write(exmp_serial)
    writer.close()


if __name__ == '__main__':

    train_dir = '../data/2016-EDM/0910_b_train.csv'
    processor = EDM_Processor()
    inputs, target_id, target_correct, ids, correct, seq_len = processor.get_train_examples(train_dir)
    print('data shape:', inputs.shape, target_id.shape, target_correct.shape, seq_len.shape)

    encode_tfrecord('train.tfrecord', inputs, target_id, target_correct, ids, correct, seq_len)

    train_dir = '../data/2016-EDM/0910_b_test.csv'
    processor = EDM_Processor()
    inputs, target_id, target_correct, ids, correct, seq_len = processor.get_train_examples(train_dir)
    print('data shape:', inputs.shape, target_id.shape, target_correct.shape, seq_len.shape)
    encode_tfrecord('eval.tfrecord', inputs, target_id, target_correct, ids, correct, seq_len)
