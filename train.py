from __future__ import print_function
from model import OSELM
import tensorflow as tf
import time
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_keras import PolicyValueNet  # Keras
import loggers as lg

# 构建OS-ELM
elm_batch_size = 1024
elm_hidden_num = 330
elm = OSELM(sess, elm_batch_size, 72, elm_hidden_num, 1)

# import tensorflow as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class TrainPipeline():
    def __init__(self, init_model = './best_policy.model'):
    # def __init__(self, init_model = None):

        # params of the board and the game
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 1024  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        # self.epochs = 1
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 3500
        # self.game_batch_num = 150

        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        # 版本的编号
        self.version_number = 0
        # 是否是第一次保存版本
        self.first_flag = True
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self, train_batch):
        """update the policy-value net"""

        # # 构建OS-ELM
        # elm_batch_size = 512
        # elm_hidden_num = 100

        # 防止多次打印提示的日志
        nn_logger_flag = True
        elm_logger_flag = True
        '''
        512    30
        在50左右时效果比value_network要好

        elm的batch可能会影响结果
        如0.53
          0.48
        '''
        # tf.set_random_seed(2016)  # 随机序列可重复
        # sess = tf.Session()
        # elm = OSELM(sess, elm_batch_size, 64, elm_hidden_num, 1)

        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy, hidden_layer_model_output, predict_value = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            winner_union = np.array(winner_batch)

            predict_value = predict_value.reshape([-1])
            # 还要返回一个中间的模型回去
            # winner_union = winner_union[:, np.newaxis]
            # print("predict_value_shape:", predict_value.shape)
            # print("winner_union_shape:", winner_union.shape)
            value_loss = tf.losses.mean_squared_error(predict_value, winner_union)
            value_loss = sess.run(value_loss)
            # 抽出中间模型在阶段层的数据
            # state_input_union = np.array(state_batch)
            # state_input_union = tf.convert_to_tensor(state_input_union)
            # # mcts_probs_union = np.array(mcts_probs)
            # print("Value_Network_Loss:{}".format(value_loss))

            # 输出日志
            # if (train_batch + 1) % self.check_freq == 0 or train_batch == 0:
            if nn_logger_flag and ((train_batch + 1) % self.check_freq == 0 or train_batch == 1):
                lg.logger_nn.info(str(train_batch) + '-----------------' + time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
                nn_logger_flag = False
            lg.logger_nn.info(value_loss)

            # 防止编译器输出过多日志报错
            # print("Value_Network_Loss")
            # print(value_loss)

            # hidden_layer_output = hidden_layer_model(state_input_union)
            elm_epoch = 0
            data_container = []
            target_container = []
            while (elm_epoch < 1):
                k = 0

                for (index, data) in enumerate(hidden_layer_model_output, start=0):
                    if (index >= (elm_epoch) * elm_batch_size and index < (elm_epoch + 1) * elm_batch_size):
                        # if (index >= (epoch ) * batch_size and index < (epoch +1) * batch_size):
                        data_container.append(data)
                        k += 1
                        if k == elm_batch_size:
                            break
                j = 0
                for (index1, target) in enumerate(winner_union, start=0):
                    if (index1 >= (elm_epoch) * elm_batch_size and index1 < (elm_epoch + 1) * elm_batch_size):
                        # if (index >= (epoch) * batch_size and index < (epoch + 1) * batch_size):
                        target_container.append(target)
                        j += 1
                        if j == elm_batch_size:
                            break
                data_container = np.array(data_container)
                target_container = np.array(target_container)
                target_container = target_container[:, np.newaxis]

                # # print(data_container.shape)
                elm.train(data_container, target_container)
                # # elm.save(sess, path)
                # # elm.test(X_test, Y_test)
                #
                ELM_Loss = elm.test(data_container, target_container)

                # print("ELM_LOSS:{}".format(ELM_Loss))

                # 输出日志
                # if (train_batch + 1) % self.check_freq == 0 or train_batch == 0:
                if elm_logger_flag and ((train_batch + 1) % self.check_freq == 0 or train_batch == 1):
                    lg.logger_elm.info(str(train_batch) + '-----------------' + time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
                    elm_logger_flag = False
                lg.logger_elm.info(ELM_Loss)




                # # 保存fnnmOS-ELM模型
                # # 反复保存.meta文件会变大导致出错
                # if self.first_flag == True:
                # # if self.first_flag == True:
                #     # elm.save(sess, 'elmmodel/OSELM_' + time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())) + '.h5')
                #     elm.save_first(sess, 'elmmodel/OSELM_' + str(self.version_number) + '.h5')
                #     self.version_number = self.version_number + 1
                #     self.first_flag = False
                #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                # if (train_batch + 1) % self.check_freq == 0 and self.first_flag == False and i==4:
                # # if self.first_flag == False:
                #     # elm.save(sess, 'elmmodel/OSELM_' + time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())) + '.h5')
                #     elm.save_nofirst(sess, 'elmmodel/OSELM_' + str(self.version_number) + '.h5')
                #     self.version_number = self.version_number + 1


                if (train_batch + 1) % self.check_freq == 0 and i==4:
                # if self.first_flag == False:
                    # elm.save(sess, 'elmmodel/OSELM_' + time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())) + '.h5')
                    elm.save_first(sess, 'elmmodel/OSELM_' + time.strftime('%d_%H_%M', time.localtime(time.time())) + '.h5')
                    # self.version_number = self.version_number + 1





                # if ELM_Loss < 0.34:
                #     elm.save(sess, "./OSELM.h5")
                #
                # elm.test(data_container, target_container)

                # print(epoch)
                elm_epoch += 1
                data_container = []
                target_container = []

            # print("new_v: {}",new_v)   # 介于-1到1之间的小数
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        # print(("kl:{:.5f},"
        #        "lr_multiplier:{:.3f},"
        #        "loss:{},"
        #        "entropy:{},"
        #        "explained_var_old:{:.3f},"
        #        "explained_var_new:{:.3f}"
        #        ).format(kl,
        #                 self.lr_multiplier,
        #                 loss,
        #                 entropy,
        #                 explained_var_old,
        #                 explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                    i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update(i)

                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate()

                    # 保存为当下所用
                    self.policy_value_net.save_model('./current_policy.model')

                    # 输出以系统时间作为版本号，存在另一文件夹中比较
                    self.policy_value_net.save_model('nnmodel/current_policy_' + time.strftime('%d_%H_%M', time.localtime(time.time())) + '.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        # 保存为当下所用
                        self.policy_value_net.save_model('./best_policy.model')

                        # 输出以系统时间作为版本号，存在另一文件夹中比较
                        self.policy_value_net.save_model('nnmodel/best_policy_' + time.strftime('%d_%H_%M', time.localtime(time.time())) + '.model')

                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
