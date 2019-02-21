# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:config.py
@time:2019/2/19下午4:20
"""
class RNN_config(object):
    # model config
    hidden_size = 256
    hidden_layer_num = 4

    initializer_range = 0.02
    max_position_embeddings = 216

    # training params
    CPU_NUMS = 3
    num_skills = None
    learning_rate = 0.1
    decay_step = 100
    lr_decay = 0.94
    keep_prob = 0.5
    pos_weight = 1
    batch_size = 64
    save_path = './saved_models'


def get_config(model_type):
    if model_type == 'RNN':
        config = RNN_config()

    return config
