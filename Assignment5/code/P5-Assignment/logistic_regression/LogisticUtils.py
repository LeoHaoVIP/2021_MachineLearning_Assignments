# * coding: utf8 *
import numpy as np
import pickle
import os
import math
import random
import cv2

CLASS_NUM = 10


class LogisticRegressionML:

    def __init__(self, train_set, test_set, train_label, test_label, epochs, batch_size, lr, _lambda=1e-5,
                 gd_type='normal'):
        # =========@BigData=========
        # 导入训练和测试数据
        self.train_set = np.reshape(train_set, (len(train_set), np.size(train_set[0])))
        self.test_set = np.reshape(test_set, (len(test_set), np.size(test_set[0])))
        self.train_label = train_label
        self.test_label = test_label
        self.gd_type = gd_type
        # =========@Model=========
        # 模型参数
        self.weight = np.zeros((CLASS_NUM, len(self.train_set[0])))
        self.bias = np.zeros(CLASS_NUM)
        # 训练参数
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._lambda = _lambda
        # 训练过程记录
        self.train_record = []
        # 模型保存相关参数
        if self.gd_type == 'regularization':
            self.save_path = './model/{0}_epochs={1}_batch_size={2}' \
                             '_lr={3}_lambda={4}.pkl'.format(self.gd_type, self.epochs, self.batch_size, self.lr,
                                                             self._lambda)
        else:
            self.save_path = './model/{0}_epochs={1}_batch_size={2}' \
                             '_lr={3}.pkl'.format(self.gd_type, self.epochs, self.batch_size, self.lr)

    # =========@Loss=========
    # 损失函数
    def loss(self):
        # 计算损失loss
        train_num = len(self.train_set)
        loss = 0
        for i in range(train_num):
            loss += -np.log(soft_max(np.dot(self.weight, self.train_set[i]) + self.bias)[self.train_label[i]])
        return loss

    # =========@Algorithm=========
    def train(self):
        # 循环所有样本为一个epoch
        for k in range(self.epochs):
            # 每次epoch执行一次shuffle
            print('---> shuffling train_set for epoch {}...'.format(k))
            shuffle_index = random.sample(range(0, len(self.train_set)), len(self.train_set))
            self.train_set = [self.train_set[i] for i in shuffle_index]
            self.train_label = [self.train_label[i] for i in shuffle_index]
            # 循环一个batch内的样本为一个iteration
            batch_num = int(len(self.train_set) / self.batch_size)
            print('batch_num: ', batch_num)
            x_batches = np.zeros((batch_num, self.batch_size, len(self.train_set[0])))
            y_batches = np.zeros((batch_num, self.batch_size))
            # 分批
            for i in range(0, len(self.train_set), self.batch_size):
                x_batches[int(i / self.batch_size)] = self.train_set[i:i + self.batch_size]
                y_batches[int(i / self.batch_size)] = self.train_label[i:i + self.batch_size]
            for i in range(batch_num):
                # 每次iter都要进行梯度初始化
                gd_w = np.zeros((CLASS_NUM, len(self.train_set[0])))
                gd_b = np.zeros(CLASS_NUM)
                # 从trainSet中随机选取batch_size个样本（随机小批量梯度下降MBGD）
                for j in range(self.batch_size):
                    gd_weight, gd_bias = gradient(self.weight, self.bias, x_batches[i][j], y_batches[i][j],
                                                  self._lambda,
                                                  self.gd_type)
                    gd_w += gd_weight
                    gd_b += gd_bias
                # 取整个batch的梯度平均为本次梯度更新值
                self.weight -= self.lr * (gd_w / self.batch_size)
                self.bias -= self.lr * (gd_b / self.batch_size)
                # 性能指标
                acc_train = self.accuracy_on_train()
                acc_test = self.accuracy_on_test()
                loss = self.loss()
                # 保存每一次训练迭代记录
                self.train_record.append([k, i, acc_train, acc_test, loss])
                if i % 1 == 0:
                    print('--->> iter= {0}, acc_train= {1}%, acc_test= {2}%, loss= {3}'.format(i, acc_train * 100,
                                                                                               acc_test * 100, loss))

    # =========@Application=========
    # 模型应用，输出预测结果
    def test_single(self, x):
        return np.argmax(soft_max(np.dot(self.weight, x) + self.bias))

    # 获取测试集得分
    def score_on_test(self):
        test_num = len(self.test_set)
        scores = np.zeros((test_num, CLASS_NUM))
        for i in range(test_num):
            scores[i] = soft_max(np.dot(self.weight, self.test_set[i]) + self.bias)
        return scores

    # 获取训练集准确率
    def accuracy_on_train(self):
        hit_count = 0
        train_num = len(self.train_set)
        for i in range(train_num):
            predict_y = np.argmax(soft_max(np.dot(self.weight, self.train_set[i]) + self.bias))
            if predict_y == self.train_label[i]:
                hit_count += 1
        return 1.0 * hit_count / train_num

    # 获取测试集准确率
    def accuracy_on_test(self):
        hit_count = 0
        test_num = len(self.test_set)
        for i in range(test_num):
            predict_y = np.argmax(soft_max(np.dot(self.weight, self.test_set[i]) + self.bias))
            if predict_y == self.test_label[i]:
                hit_count += 1
        return 1.0 * hit_count / test_num

    # 判断模型是否存在
    def model_exists(self):
        return os.path.exists(self.save_path)

    # 加载本地模型
    def load_model(self):
        with open(self.save_path, 'rb') as f:
            model = pickle.load(f)
        return model

    # 保存模型到本地
    def save_model(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self, f)


# soft_max function
def soft_max(x):
    return np.exp(x) / np.exp(x).sum()


# 计算梯度
def gradient(w, b, x, y, _lambda=0.0, gd_type='normal'):
    gd_weight = np.zeros(np.shape(w))
    gd_bias = np.zeros(np.shape(b))
    s = soft_max(np.dot(w, x) + b)
    r_w = np.shape(w)[0]
    c_w = np.shape(w)[1]
    b_column = b.shape[0]
    # 对Wij和bi分别求梯度
    for i in range(r_w):
        for j in range(c_w):
            gd_weight[i][j] = (s[i] - 1) * x[j] if y == i else s[i] * x[j]
    for i in range(b_column):
        gd_bias[i] = s[i] - 1 if y == i else s[i]
    # 带有正则项的梯度下降
    if gd_type == 'regularization':
        gd_weight += _lambda * w
    # 带有随机噪音项的梯度下降
    elif gd_type == 'noise':
        gd_weight += np.random.normal(0, 0.1, np.shape(w))
    # 普通随机下降
    return gd_weight, gd_bias
