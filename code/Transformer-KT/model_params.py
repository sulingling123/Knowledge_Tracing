# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:model_params.py
@time:2019/8/1212:42 PM
"""
from collections import defaultdict


BASE_PARAMS = defaultdict(
    # Input params
    default_batch_size=256,  # Maximum number of tokens per batch of examples.
    max_length=512,  # Maximum number of tokens per example.


    # Model Params
    initializer_gain=1.0,  # Used in trainable variable initialization
    vocab_size=124,  # unique tokes defined in the vocabulary   (num_skills)
    hidden_size=256,  # hidden size
    num_hidden_layers=2,  # number of layers in the encoder
    num_heads=8,  # number of heads to use in the multi-head attention

    filter_size=512,  # Inner layer dimension in the feedward netword
    allow_ffn_pad=True,

    # Dropout values(Only used in training)
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,


    # Training params
    learning_rate=0.1,
    learning_rate_decay_rate=1.0,
    learning_rate_warmup_steps=16000,

    # Optimizer params
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,

)