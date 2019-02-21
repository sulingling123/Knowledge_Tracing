# -*- coding:utf-8 -*-
"""
@author:Zoe
@file:rnn_dkt.py
@time:2019/2/12下午2:37
"""
import tensorflow as tf


class DKT_LSTM(object):
    def __init__(self, config, is_training):
        # hyper params
        self.config = config
        self.hidden_size = config.hidden_size
        self.hidden_layer_num = config.hidden_layer_num
        self.num_skills = config.num_skills
        self.keep_prob = config.keep_prob
        self.learning_rate = config.learning_rate
        self.pos_weight = config.pos_weight

        if not is_training:  # predict
            self.keep_prob = 1

        # inputs placeholder
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.target_id = tf.placeholder(tf.int32, [None, None], name='target_id')
        self.target_correct = tf.placeholder(tf.float32, [None, None], name='target_correct')
        self.seq_len = tf.placeholder(tf.int32, [None], name='actual_steps')

        # initial state
        self.init_state = tf.placeholder(tf.float32, [self.hidden_layer_num, 2, None, self.hidden_size])
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        initial_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(self.config.hidden_layer_num)]
        )

        self.embedding_output = tf.one_hot(self.inputs, depth=2*self.num_skills + 1)

        hidden_layers = []
        final_hidden_size = self.hidden_size
        with tf.VariableScope('lstm_cell', initializer=tf.orthogonal_initializer()):
            for i in range(self.hidden_layer_num):
                # final_hidden_size = int(self.hidden_size/(i+1))

                hidden_layer = tf.contrib.rnn.BasicLSTMCell(num_units=final_hidden_size, forget_bias=0, state_is_tuple=True)
                # lstm_layer = tf.contrib.rnn.GRUCell(num_units=final_hidden_size, state_is_tuple=True)

                hidden_layer = tf.contrib.rnn.DropoutWrapper(cell=hidden_layer,
                                                             output_keep_prob=self.keep_prob)
                hidden_layers.append(hidden_layer)
            self.hidden_cell = tf.contrib.rnn.MultiRNNCell(cells=hidden_layers, state_is_tuple=True)

            # dynamic rnn
            state_series, self.current_state = tf.nn.dynamic_rnn(cell=self.hidden_cell,
                                                                 inputs=self.embedding_output,
                                                                 sequence_length=self.seq_len,
                                                                 initial_state=initial_state)

        output_w = tf.get_variable("W", [final_hidden_size, self.num_skills], initializer=tf.contrib.layers.xavier_initializer())

        output_b = tf.get_variable("b", [self.num_skills])

        self.batch_size, self.max_len = tf.shape(self.inputs)[0], tf.shape(self.inputs)[1]
        self.state_series = tf.reshape(state_series, [self.batch_size * self.max_len, final_hidden_size])

        self.logits = tf.matmul(self.state_series, output_w) + output_b
        self.mat_logits = tf.reshape(self.logits, [self.batch_size, self.max_len, self.num_skills])
        self.pred_all = tf.sigmoid(self.mat_logits)
        tf.summary.histogram("weights", output_w)
        tf.summary.histogram("biases", output_b)

    def optimizer(self):
        # compute loss
        flat_logits = tf.reshape(self.logits, [-1])
        flat_target_correctness = tf.reshape(self.target_correct, [-1])
        flat_base_target_index = tf.range(self.batch_size * self.max_len) * self.num_skills
        flat_bias_target_id = tf.reshape(self.target_id, [-1])
        flat_target_id = flat_bias_target_id + flat_base_target_index
        flat_target_logits = tf.gather(flat_logits, flat_target_id)
        self.pred = tf.sigmoid(tf.reshape(flat_target_logits, [self.batch_size, self.max_len]))
        self.binary_pred = tf.cast(tf.greater_equal(self.pred, 0.5), tf.int32)
        self.loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(targets=flat_target_correctness,
                                                                           logits=flat_target_logits,
                                                                           pos_weight=self.pos_weight))

        trainable_vars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), 3)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # decay learning rate
        starter_learning_rate = self.learning_rate
        self.lr = tf.train.exponential_decay(starter_learning_rate, self.global_step, self.config.decay_step,
                                             self.config.lr_decay, staircase=True)

        optimizer = tf.train.AdamOptimizer(self.lr, epsilon=0.1)
        # optimizer = tf.train.GradientDescentOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(self.grads, trainable_vars), global_step=self.global_step,
                                                  name="train_op")
        # tf.summary.
        tf.summary.scalar('cross_entropy', self.loss)
        tf.summary.scalar('learning_rate', self.lr)

        return self.train_op, self.loss, self.global_step

    def get_lr(self, sess):
        return sess.run(self.lr)

    def accuracy(self):
        predict = tf.reshape(self.binary_pred, [-1])
        target = tf.reshape(self.target_correct, [-1])

        mask = tf.reshape(tf.sequence_mask(self.seq_len, maxlen=tf.shape(self.inputs)[1]), [-1])
        flat_predict_mask = tf.boolean_mask(predict, mask)
        flat_target_mask = tf.boolean_mask(target, mask)

        correct_prediction = tf.equal(tf.cast(flat_target_mask, tf.int32), tf.cast(flat_predict_mask, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # accuracy_session = tf.metrics.accuracy(self.target_correctness, self.predict[1])[1]
        # auc_session = tf.metrics.auc(self.target_correctness, self.predict[1])[1]
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

