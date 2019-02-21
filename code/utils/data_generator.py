# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:data_generator.py
@time:2019/2/19下午12:41
"""
import numpy as np
from collections import defaultdict
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
        for i in range(len(tuple_rows)):
            # inputs
            inputs.append([int(tuple_rows[i][1][j]) + int(tuple_rows[i][2][j]) * self.num_skill for j in range(len(tuple_rows[i][1]) - 1)])
            seq_len.append(int(tuple_rows[i][0][0]) - 1)  # sequence
            target_id.append(list(map(lambda k: int(k), tuple_rows[i][1][1:])))
            target_correct.append(list(map(lambda k: int(k), tuple_rows[i][2][1:])))

        return np.array(inputs), np.array(target_id), np.array(target_correct), np.array(seq_len)


if __name__ == '__main__':

    train_dir = '../data/2016-EDM/0910_b_train.csv'
    processor = EDM_Processor()
    inputs, target_id, target_correct, seq_len = processor.get_train_examples(train_dir)
    print('data shape:', inputs.shape, target_id.shape, target_correct.shape, seq_len.shape)