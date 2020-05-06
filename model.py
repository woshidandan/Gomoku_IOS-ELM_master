# -*- coding: utf-8 -*-
"""
The model of OS-ELM

@author: Shuai He
"""

import tensorflow as tf
import numpy as np

class OSELM(object):
  def __init__(self, sess, batch_size, input_len, hidden_num, output_len):
    '''
    Args:
      sess : TensorFlow session.
      batch_size : The batch size (N)   1100
      input_len : The length of input. (L)  784
      hidden_num : The number of hidden node. (K)   1000
      output_len : The length of output. (O)   10
    '''

    self._sess = sess
    self._batch_size = batch_size
    self._input_len = input_len
    self._hidden_num = hidden_num
    self._output_len = output_len
    # Variables
    self._W = tf.Variable(
      tf.random_normal([self._input_len, self._hidden_num]),
      trainable=False, dtype=tf.float32)
    self._b = tf.Variable(
      tf.random_normal([self._hidden_num]),
      trainable=False, dtype=tf.float32)
    self._beta = tf.Variable(
      tf.zeros([self._hidden_num, self._output_len]),
      trainable=False, dtype=tf.float32)

    self._var_list = [self._W, self._b, self._beta]

    self._X = tf.placeholder(
      tf.float32, [self._batch_size, self._input_len])
    self._T = tf.placeholder(
      tf.float32, [self._batch_size, self._output_len])


    # for train
    self._P = tf.Variable(
      tf.zeros([self._hidden_num, self._hidden_num]),
      trainable=False, dtype=tf.float32)
    self._H = tf.Variable(
      tf.zeros([self._batch_size, self._hidden_num]),
      trainable=False, dtype=tf.float32)
    # print(self._X)  1100 784
    # print(self._W)  784  1100
    self._set_H = self._H.assign(
      tf.nn.sigmoid(tf.matmul(self._X, self._W) + self._b))
    self._H_T = tf.transpose(self._H)

    # train : init phase
    t_P0 = tf.matrix_inverse(tf.cast(tf.matmul(self._H_T, self._H), tf.float32))
    self._init_P0 = self._P.assign(t_P0)

    t_beta0 = tf.matmul(tf.matmul(
      self._P, self._H_T), self._T)
    self._init_beta = self._beta.assign(t_beta0)

    self._init_flag = False

    # train : sequential learning phase
    self._t_beta = tf.Variable(
      tf.zeros([self._hidden_num, self._output_len]),
      trainable=False, dtype=tf.float32)
    self._t_P = tf.Variable(
      tf.zeros([self._hidden_num, self._hidden_num]),
      trainable=False, dtype=tf.float32)

    self._swap_P = self._t_P.assign(self._P)
    self._swap_beta = self._t_beta.assign(self._beta)

    eye = tf.constant(np.identity(self._batch_size), dtype=tf.float32)
    t_P1 = self._t_P - tf.matmul(tf.matmul(tf.matmul(tf.matmul(self._t_P, self._H_T), 
      tf.matrix_inverse(eye + tf.matmul(tf.matmul(self._H, self._t_P), self._H_T))), self._H), self._t_P)
    t_beta1 = self._t_beta + tf.matmul(tf.matmul(t_P1, self._H_T), (tf.tanh(self._T) - tf.tanh(tf.matmul(self._H, self._t_beta))))

    self._update_P = self._P.assign(t_P1)
    self._update_beta = self._beta.assign(t_beta1)
    ###以上完成了OS-ELM参数的更新

    # for test
    self._X1 = tf.placeholder(
      tf.float32, [None, self._input_len])
    self._T1 = tf.placeholder(
      tf.float32, [None, self._output_len])
    self._H1 = tf.nn.sigmoid(tf.matmul(self._X1, self._W) + self._b)
    # print(self._beta)  1000 10
    # self._fx = tf.tanh(tf.matmul(self._H1, self._beta))
    self._fx = tf.tanh(tf.matmul(self._H1, self._beta))


    self._loss = tf.losses.mean_squared_error(self._fx, self._T1)

    self._saver = tf.train.Saver(self._var_list, max_to_keep = 1000)
    self._sess.run(tf.variables_initializer(self._var_list))

  def train(self, x, t):
    self._sess.run(self._set_H, {self._X:x})

    if not self._init_flag :
      self._sess.run(tf.cast(self._init_P0, tf.float32))
      self._sess.run(self._init_beta, {self._T:t})
      self._init_flag = True
    else :
      self._sess.run(self._swap_P)
      self._sess.run(self._swap_beta)

      self._sess.run(self._update_P)
      self._sess.run(self._update_beta, {self._T:t})

  def save_first(self, sess, ckpt_path):
    self._saver.save(sess, ckpt_path)
    # tf.reset_default_graph()
  def save_nofirst(self, sess, ckpt_path):
    self._saver.save(sess, ckpt_path, write_meta_graph=False)

  def load(self, sess, ckpt_path):
    self._saver.restore(sess, ckpt_path)
    # tf.reset_default_graph()

  def predict_value(self, x):
    # print("ELM Predict:{}".format(self._sess.run(self._fx , {self._X1:x})))
    return self._sess.run(self._fx , {self._X1:x})

  def test_print(self, x):
    print("SUCCESS")


  def test(self, x, t):
    if not self._init_flag :
      print("Former_Accuracy: {:.9f}".format(
        self._sess.run(self._accuracy, {self._X1:x, self._T1:t})))
      print("Predict is : {}".format(self._sess.run(tf.argmax(self._fx, 1), {self._X1:x})))
      print("table is : {}".format(self._sess.run(tf.argmax(self._T1, 1), {self._T1:t})))


    if t is not None :
      return self._sess.run(self._loss, {self._X1:x, self._T1:t})
    else : return self._sess.run(self._fx , {self._X1:x})
