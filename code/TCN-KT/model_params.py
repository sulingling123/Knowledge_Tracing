# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:model_params.py
@time:2019/10/83:31 PM
"""
from collections import defaultdict


BASE_PARAMS = defaultdict(
    # Input params
    num_skills=125,

    # Model params
    initializer_range=0.02,  # embedding table initializer
    keep_prob=0.5,
    use_LN=True,
    hidden_layer_num=4,
    hidden_size=128,
    kernel_size = 3,  # 卷积核大小
    use_one_hot=False,

    # Training params
    learning_rate=0.1,
    learning_rate_decay_rate=1.0,
    learning_rate_warmup_steps=16000,

    # Optimizer params
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,

)