# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet with Keras
Tested under Keras 2.0.5 with tensorflow-gpu 1.2.1 as backend

@author: Shuai He
""" 

from __future__ import print_function
from model import OSELM
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras.utils import np_utils

import numpy as np
import pickle
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)


# # 训练OS-ELM不要放到代码中间，否则多次运行会报错,报参数改变的错误，读入OS-ELM模型取消注释
# elm_batch_size = 512
# elm_hidden_num = 100
# # path = "./elmmodel/OSELM_2019_05_21_07_01.h5"
# path = "./elmmodel/OSELM_1.h5"
# elm = OSELM(sess, elm_batch_size, 64, elm_hidden_num, 1)
# elm.load(sess, path)

class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, model_file):
        self.board_width = board_width
        self.board_height = board_height 
        self.l2_const = 1e-4  # coef of l2 penalty 
        self.create_policy_value_net()   
        self._loss_train_op()

        if model_file:
            self.model.load_weights(model_file)
        
    def create_policy_value_net(self):
        """create the policy value network """   
        in_x = network = Input((4, self.board_width, self.board_height))

        # conv layers
        network = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        # action policy layers
        policy_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        policy_net = Flatten()(policy_net)
        self.policy_net = Dense(self.board_width*self.board_height, activation="softmax", kernel_regularizer=l2(self.l2_const))(policy_net)
        # state value layers
        value_net = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        value_net = Flatten()(value_net)
        value_net = Dense(64, kernel_regularizer=l2(self.l2_const))(value_net)
        self.value_net = Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(value_net)

        self.model = Model(in_x, [self.policy_net, self.value_net])
        self.model.summary()
        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results
        self.policy_value = policy_value
        
    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """

        legal_positions = board.availables
        current_state = board.current_state()
        act_probs, value = self.policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])

        # 构建中间层输出，使用fnnm-OSELM预测时取消注释
        # layer_name = 'dense_2'
        # hidden_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        # hidden_layer_model_output = hidden_layer_model.predict(current_state.reshape(-1, 4, self.board_width, self.board_height))
        #
        # oselm_predict_value = elm.predict_value(hidden_layer_model_output)
        # print("oselm_predict_value:{}", oselm_predict_value[0][0])
        # print("value[0][0]:{}", value[0][0])

        #使用fnnm-OSELM进行预测
        # return act_probs, value[0][0]
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        # get the train op   
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(state_input, mcts_probs, winner, learning_rate):
        	
            layer_name = 'flatten_2'

            # layer_name = 'dense_2'
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            winner_union = np.array(winner)

            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            # action_probs, _ = self.model.predict_on_batch(state_input_union)
            action_probs, predict_value = self.model.predict_on_batch(state_input_union)

            entropy = self_entropy(action_probs)

            K.set_value(self.model.optimizer.lr, learning_rate)
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)

            hidden_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
            hidden_layer_model_output = hidden_layer_model.predict(state_input_union)

            return loss[0], entropy, hidden_layer_model_output, predict_value
        
        self.train_step = train_step

    # def get_policy_param(self):
    #     self.model.save(model_file)
    #     net_params = self.model.get_weights()
    #     return net_params

    # def elm_offline(self, batch_size, input_len, hidden_num, output_len):
    #     W = tf.Variable(tf.float32, [input])

    def save_model(self, model_file):
        """ save model params to file """
        self.model.save(model_file)
        # net_params = self.get_policy_param()
        # pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
