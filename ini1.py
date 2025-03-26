import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import random
import torch
from torch import nn, dtype
from AQI_DataSet import *

learning_rate = 0.01
betas = (0.9, 0.999)
alpha = 0.9

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    # with tqdm(total=len(self.train_data_loader), desc=f"training_epoch{num}") as pbar:


def zongchazhi(y_test, y_test_pred):
    '''计算实际结果与预测结果的总差值,偏差率'''
    aaa = y_test
    bbb = y_test_pred
    sum1 = sum(aaa)
    sum2 = sum(aaa) - sum(bbb)
    rate = sum2 / sum1
    print(f'总差值：{round(sum2, 3)}, 偏差率：{round(rate * 100, 2)}%', end='   ')
    return round(sum2, 3), round(rate * 100, 2)


def print_evaluate(true, predicted):
    '''预测结果评价'''
    mae = round(metrics.mean_absolute_error(true, predicted), 3)
    mse = round(metrics.mean_squared_error(true, predicted), 3)
    rmse = round(np.sqrt(metrics.mean_squared_error(true, predicted)), 3)
    r2_square = round(metrics.r2_score(true, predicted), 3)
    error, errate = zongchazhi(true, predicted)
    print('MAE:', mae, end='   ')
    print('MSE:', mse, end='   ')
    print('RMSE:', rmse, end='   ')
    print('R2 Square:', r2_square)
    # print('__________________________________')
    return [mae, mse, rmse, r2_square, error, errate]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_data():
    # 加载数据集
    HUM = np.genfromtxt("data/micro_HT3.csv", delimiter=",")

    features = HUM[:, :2]  # 选择除最后一列之外的所有列作为特征
    target = HUM[:, [2]]  # 选择最后一列作为标签

    # 将输入数据改为矩阵；
    samplein = np.mat(features).T

    # 将输出数据改为矩阵；
    sampleout = np.mat(target).T

    # 求得输入数据的最小值和最大值：[最小值 最大值]；

    sampleinminmax = np.array(
        [samplein.min(axis=1).T.tolist()[0],
         samplein.max(axis=1).T.tolist()[0]]
    ).transpose()

    # 求得 passengertraffic 和 freighttraffic 的最小值和最大值：[最小值 最大值]；
    sampleoutminmax = np.array(
        [sampleout.min(axis=1).T.tolist()[0],
         sampleout.max(axis=1).T.tolist()[0]]
    ).transpose()

    # 将数据进行标准化（归一化）；

    # 将数据映射到 -1 到 1 的范围内；
    sampleinnorm = (
            2 *
            (np.array(samplein.T) - sampleinminmax.transpose()[0]) /
            (sampleinminmax.transpose()[1] - sampleinminmax.transpose()[0])
            - 1
    ).transpose()

    # 将数据映射到 -1 到 1 的范围内；
    sampleoutnorm = (
            2 *
            (np.array(sampleout.T).astype(float) - sampleoutminmax.transpose()[0]) /
            (sampleoutminmax.transpose()[1] - sampleoutminmax.transpose()[0])
            - 1
    ).transpose()

    # 返回标准化后的输入、输出数据、原始数据的最值；
    return sampleinnorm, sampleoutnorm, sampleoutminmax, sampleout


def predict1(w1, b1, w11, b11, w2, b2, sampleinnorm):
    hiddenout = sigmoid((np.dot(w1, sampleinnorm).transpose() + b1.transpose())).transpose()
    hiddenout1 = sigmoid((np.dot(w11, hiddenout).transpose() + b11.transpose())).transpose()
    networkout = (np.dot(w2, hiddenout1).transpose() + b2.transpose()).transpose()
    return networkout


class BPNN:
    # 构造7层BP网络架构
    def __init__(self, hiddenunitnum1=4, hiddenunitnum2=6, hiddenunitnum3=8, hiddenunitnum4=10, hiddenunitnum5=8,
                 hiddenunitnum6=6, hiddenunitnum7=4, maxepochs=10000, learnrate=0.001, findBest='SSA'):
        # 训练的次数；
        self.maxepochs = maxepochs
        # 学习率；
        self.learnrate = learnrate
        # 认为可以停止训练的理想误差，达到该误差值时停止训练；
        self.errorfinal = 3 * (10 ** -2)
        # 搜索算法
        self.finfBest = findBest

        self.hiddenunitnum1 = hiddenunitnum1
        self.hiddenunitnum2 = hiddenunitnum2
        self.hiddenunitnum3 = hiddenunitnum3
        self.hiddenunitnum4 = hiddenunitnum4
        self.hiddenunitnum5 = hiddenunitnum5
        self.hiddenunitnum6 = hiddenunitnum6
        self.hiddenunitnum7 = hiddenunitnum7

        # 创建一个列表，用于存储每次训练产生的误差，训练结束后用于绘图，进行可视化，便于分析整个训练过程误差的变化情况；
        self.errhistory = []
        self.deltaStroy = np.array([])  # 存储梯度

        self.device = device

    def train(self, sampleinnorm, sampleoutnorm):
        self.sampleinnorm = torch.tensor(sampleinnorm, dtype=torch.float32)
        self.sampleoutnorm = torch.tensor(sampleoutnorm, dtype=torch.float32)
        # 样本数，下面反向求解计算时用于产生样本数个 1 参与运算，用于计数；
        self.samnum = sampleinnorm.shape[1]
        # 输入神经网络的数据维度；
        self.indim = sampleinnorm.shape[0]
        # 输出神经网络的数据维度；
        self.outdim = sampleoutnorm.shape[0]

        # 创建隐藏层的 w 权重和 b 偏置矩阵；
        self.w1 = 2 * np.random.rand(self.hiddenunitnum1, self.indim) - 1
        self.b1 = 2 * np.random.rand(self.hiddenunitnum1, 1) - 1

        # 创建隐藏层的 w 权重和 b 偏置矩阵；
        self.w2 = 2 * np.random.rand(self.hiddenunitnum2, self.hiddenunitnum1) - 1
        self.b2 = 2 * np.random.rand(self.hiddenunitnum2, 1) - 1

        self.w3 = 2 * np.random.rand(self.hiddenunitnum3, self.hiddenunitnum2) - 1
        self.b3 = 2 * np.random.rand(self.hiddenunitnum3, 1) - 1

        self.w4 = 2 * np.random.rand(self.hiddenunitnum4, self.hiddenunitnum3) - 1
        self.b4 = 2 * np.random.rand(self.hiddenunitnum4, 1) - 1

        self.w5 = 2 * np.random.rand(self.hiddenunitnum5, self.hiddenunitnum4) - 1
        self.b5 = 2 * np.random.rand(self.hiddenunitnum5, 1) - 1

        self.w6 = 2 * np.random.rand(self.hiddenunitnum6, self.hiddenunitnum5) - 1
        self.b6 = 2 * np.random.rand(self.hiddenunitnum6, 1) - 1

        self.w7 = 2 * np.random.rand(self.hiddenunitnum7, self.hiddenunitnum6) - 1
        self.b7 = 2 * np.random.rand(self.hiddenunitnum7, 1) - 1

        # 创建输出层的 w 权重和 b 偏置矩阵；
        self.w0 = 2 * np.random.rand(self.outdim, self.hiddenunitnum2) - 1
        self.b0 = 2 * np.random.rand(self.outdim, 1) - 1

        self.deltaStroy = np.array([[np.zeros([self.hiddenunitnum1, self.indim]), np.zeros([self.hiddenunitnum1, 1]),
                                     np.zeros([self.hiddenunitnum2, self.hiddenunitnum1]),
                                     np.zeros([self.hiddenunitnum2, 1]),
                                     np.zeros([self.hiddenunitnum3, self.hiddenunitnum2]),
                                     np.zeros([self.hiddenunitnum3, 1]),
                                     np.zeros([self.hiddenunitnum4, self.hiddenunitnum3]),
                                     np.zeros([self.hiddenunitnum4, 1]),
                                     np.zeros([self.hiddenunitnum5, self.hiddenunitnum4]),
                                     np.zeros([self.hiddenunitnum5, 1]),
                                     np.zeros([self.hiddenunitnum6, self.hiddenunitnum5]),
                                     np.zeros([self.hiddenunitnum6, 1]),
                                     np.zeros([self.hiddenunitnum7, self.hiddenunitnum6]),
                                     np.zeros([self.hiddenunitnum7, 1]),
                                     np.zeros([self.outdim, self.hiddenunitnum7]), np.zeros([self.outdim, 1])]],
                                   dtype=object)  # 存储梯度
        self.deltaWB = np.array([[np.zeros([self.hiddenunitnum1, self.indim]), np.zeros([self.hiddenunitnum1, 1]),
                                  np.zeros([self.hiddenunitnum2, self.hiddenunitnum1]),
                                  np.zeros([self.hiddenunitnum2, 1]),
                                  np.zeros([self.hiddenunitnum3, self.hiddenunitnum2]),
                                  np.zeros([self.hiddenunitnum3, 1]),
                                  np.zeros([self.hiddenunitnum4, self.hiddenunitnum3]),
                                  np.zeros([self.hiddenunitnum4, 1]),
                                  np.zeros([self.hiddenunitnum5, self.hiddenunitnum4]),
                                  np.zeros([self.hiddenunitnum5, 1]),
                                  np.zeros([self.hiddenunitnum6, self.hiddenunitnum5]),
                                  np.zeros([self.hiddenunitnum6, 1]),
                                  np.zeros([self.hiddenunitnum7, self.hiddenunitnum6]),
                                  np.zeros([self.hiddenunitnum7, 1]),
                                  np.zeros([self.outdim, self.hiddenunitnum7]), np.zeros([self.outdim, 1])]],
                                dtype=object)  # 存储累加量

        if self.finfBest == 'SSA':
            self.errorbackpropagateSSA()  # 麻雀搜索
            self.trial1()
            self.save_model()
            n_feature = 2
            n_hidden11 = 4
            n_hidden22 = 6
            n_hidden33 = 8
            n_hidden44 = 10
            n_hidden55 = 8
            n_hidden66 = 6
            n_hidden77 = 4
            n_output = 1
            # model = Net(n_feature, n_hidden11, n_hidden22, n_hidden33, n_hidden44, n_hidden55, n_hidden66, n_hidden77,
            #             n_output)
            # state = {'model': model.state_dict()}
            # torch.save(state, "model/BPNN_adam1.pt")

    def errorbackpropagateSSA(self, ):
        # 麻雀搜索权值和阈值
        import SSA
        # 设置参数
        pop = 80  # 种群数量
        Max_iter = 300  # 最大迭代次数
        dim = self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + \
              self.hiddenunitnum1 * self.hiddenunitnum2 + self.hiddenunitnum2 + \
              self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 + \
              self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 + \
              self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5 + \
              self.hiddenunitnum5 * self.hiddenunitnum6 + self.hiddenunitnum6 + \
              self.hiddenunitnum6 * self.hiddenunitnum7 + self.hiddenunitnum7 + \
              self.hiddenunitnum7 * self.outdim + self.outdim
        # 维度  dim=inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum
        lb = np.zeros((dim, 1)) - 3
        ub = np.zeros((dim, 1)) + 3

        fun = self.fun_cnn
        GbestScore, GbestPositon, Curve = SSA.Tent_SSA(pop, dim, lb, ub, Max_iter, fun)
        self.XtoWB(GbestPositon.T)
        # 使用一个列表将误差存储下来；
        self.errhistory = Curve.tolist()

    def fun_cnn(self, X):
        self.XtoWB(X)  # X还原阈值w1、w0;权值b1、b0
        networkout = self.predict2(self.sampleinnorm)  # 正向传播
        sse = metrics.mean_squared_error(self.sampleoutnorm, networkout)
        return sse

    def predict2(self, sampleinnorm):
        '''使用训练好的神经网络进行预测；'''
        # 使用新的权重计算隐藏层和输出层的输出；
        # h = f(w1 x + b1)
        self.hiddenout1 = sigmoid((np.dot(self.w1, sampleinnorm).transpose() + self.b1.transpose())).transpose()
        # h2 = f(w2 x + b2)
        self.hiddenout2 = sigmoid((np.dot(self.w2, self.hiddenout1).transpose() + self.b2.transpose())).transpose()
        self.hiddenout3 = sigmoid((np.dot(self.w3, self.hiddenout2).transpose() + self.b3.transpose())).transpose()
        self.hiddenout4 = sigmoid((np.dot(self.w4, self.hiddenout3).transpose() + self.b4.transpose())).transpose()
        self.hiddenout5 = sigmoid((np.dot(self.w5, self.hiddenout4).transpose() + self.b5.transpose())).transpose()
        self.hiddenout6 = sigmoid((np.dot(self.w6, self.hiddenout5).transpose() + self.b6.transpose())).transpose()
        self.hiddenout7 = sigmoid((np.dot(self.w7, self.hiddenout6).transpose() + self.b7.transpose())).transpose()
        # o = w0 h2 + b0
        self.networkout = (np.dot(self.w0, self.hiddenout7).transpose() + self.b0.transpose()).transpose()
        return self.networkout

    def XtoWB(self, X):
        '''将X拆解成权值w和阈值b'''
        X = X[:]
        # print(X)
        # print(X.shape)
        X = np.array(X)
        self.w1 = X[:self.indim * self.hiddenunitnum1].reshape(self.hiddenunitnum1, self.indim)
        self.b1 = X[self.indim * self.hiddenunitnum1:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1
                  ].reshape(self.hiddenunitnum1, 1)

        self.w2 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                  ].reshape(self.hiddenunitnum2, self.hiddenunitnum1)
        self.b2 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2
                  ].reshape(self.hiddenunitnum2, 1)

        self.w3 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3
                  ].reshape(self.hiddenunitnum3, self.hiddenunitnum2)
        self.b3 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3
                  ].reshape(self.hiddenunitnum3, 1)

        self.w4 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4
                  ].reshape(self.hiddenunitnum4, self.hiddenunitnum3)
        self.b4 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4
                  ].reshape(self.hiddenunitnum4, 1)

        self.w5 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5
                  ].reshape(self.hiddenunitnum5, self.hiddenunitnum4)
        self.b5 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5
                  ].reshape(self.hiddenunitnum5, 1)

        self.w6 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5 +
                    self.hiddenunitnum5 * self.hiddenunitnum6
                  ].reshape(self.hiddenunitnum6, self.hiddenunitnum5)
        self.b6 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5 +
                    self.hiddenunitnum5 * self.hiddenunitnum6:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5 +
                    self.hiddenunitnum5 * self.hiddenunitnum6 + self.hiddenunitnum6
                  ].reshape(self.hiddenunitnum6, 1)

        self.w7 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + self.hiddenunitnum1 * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 + self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5 +
                    self.hiddenunitnum5 * self.hiddenunitnum6 + self.hiddenunitnum6:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 +
                    self.hiddenunitnum1 * self.hiddenunitnum2 + self.hiddenunitnum2 +
                    self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5 +
                    self.hiddenunitnum5 * self.hiddenunitnum6 + self.hiddenunitnum6 +
                    self.hiddenunitnum6 * self.hiddenunitnum7
                  ].reshape(self.hiddenunitnum7, self.hiddenunitnum6)

        self.b7 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 +
                    self.hiddenunitnum1 * self.hiddenunitnum2 + self.hiddenunitnum2 +
                    self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5 +
                    self.hiddenunitnum5 * self.hiddenunitnum6 + self.hiddenunitnum6 +
                    self.hiddenunitnum6 * self.hiddenunitnum7:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 +
                    self.hiddenunitnum1 * self.hiddenunitnum2 + self.hiddenunitnum2 +
                    self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5 +
                    self.hiddenunitnum5 * self.hiddenunitnum6 + self.hiddenunitnum6 +
                    self.hiddenunitnum6 * self.hiddenunitnum7 + self.hiddenunitnum7
                  ].reshape(self.hiddenunitnum7, 1)

        self.w0 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 +
                    self.hiddenunitnum1 * self.hiddenunitnum2 + self.hiddenunitnum2 +
                    self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5 +
                    self.hiddenunitnum5 * self.hiddenunitnum6 + self.hiddenunitnum6 +
                    self.hiddenunitnum6 * self.hiddenunitnum7 + self.hiddenunitnum7:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 +
                    self.hiddenunitnum1 * self.hiddenunitnum2 + self.hiddenunitnum2 +
                    self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5 +
                    self.hiddenunitnum5 * self.hiddenunitnum6 + self.hiddenunitnum6 +
                    self.hiddenunitnum6 * self.hiddenunitnum7 + self.hiddenunitnum7 +
                    self.hiddenunitnum7 * self.outdim
                  ].reshape(self.outdim, self.hiddenunitnum7)
        self.b0 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 +
                    self.hiddenunitnum1 * self.hiddenunitnum2 + self.hiddenunitnum2 +
                    self.hiddenunitnum2 * self.hiddenunitnum3 + self.hiddenunitnum3 +
                    self.hiddenunitnum3 * self.hiddenunitnum4 + self.hiddenunitnum4 +
                    self.hiddenunitnum4 * self.hiddenunitnum5 + self.hiddenunitnum5 +
                    self.hiddenunitnum5 * self.hiddenunitnum6 + self.hiddenunitnum6 +
                    self.hiddenunitnum6 * self.hiddenunitnum7 + self.hiddenunitnum7 +
                    self.hiddenunitnum7 * self.outdim:].reshape(self.outdim, 1)

    def trial1(self, ):
        self.w11 = torch.tensor(self.w1)
        self.b11 = torch.tensor(self.b1.reshape(self.hiddenunitnum1))
        self.w22 = torch.tensor(self.w2)
        self.b22 = torch.tensor(self.b2.reshape(self.hiddenunitnum2))
        self.w33 = torch.tensor(self.w3)
        self.b33 = torch.tensor(self.b3.reshape(self.hiddenunitnum3))
        self.w44 = torch.tensor(self.w4)
        self.b44 = torch.tensor(self.b4.reshape(self.hiddenunitnum4))
        self.w55 = torch.tensor(self.w5)
        self.b55 = torch.tensor(self.b5.reshape(self.hiddenunitnum5))
        self.w66 = torch.tensor(self.w6)
        self.b66 = torch.tensor(self.b6.reshape(self.hiddenunitnum6))
        self.w77 = torch.tensor(self.w7)
        self.b77 = torch.tensor(self.b7.reshape(self.hiddenunitnum7))
        self.w00 = torch.tensor(self.w0)
        self.b00 = torch.tensor(self.b0.reshape(self.outdim))

    def save_model(self, ):
        self.state = {
            'hidden11.weight': self.w11,
            'hidden11.bias': self.b11,
            'hidden22.weight': self.w22,
            'hidden22.bias': self.b22,
            'hidden33.weight': self.w33,
            'hidden33.bias': self.b33,
            'hidden44.weight': self.w44,
            'hidden44.bias': self.b44,
            'hidden55.weight': self.w55,
            'hidden55.bias': self.b55,
            'hidden66.weight': self.w66,
            'hidden66.bias': self.b66,
            'hidden77.weight': self.w77,
            'hidden77.bias': self.b77,
            'predict.weight': self.w00,
            'predict.bias': self.b00
        }
        torch.save(self.state, "model/BPNN_adam1.pt")

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden11, n_hidden22, n_hidden33, n_hidden44, n_hidden55, n_hidden66, n_hidden77,
                 n_output):
        super(Net, self).__init__()
        # 定义每层用什么样的形式
        self.hidden11 = torch.nn.Linear(n_feature, n_hidden11)
        self.hidden22 = torch.nn.Linear(n_hidden11, n_hidden22)
        self.hidden33 = torch.nn.Linear(n_hidden22, n_hidden33)
        self.hidden44 = torch.nn.Linear(n_hidden33, n_hidden44)
        self.hidden55 = torch.nn.Linear(n_hidden44, n_hidden55)
        self.hidden66 = torch.nn.Linear(n_hidden55, n_hidden66)
        self.hidden77 = torch.nn.Linear(n_hidden66, n_hidden77)
        # 定义隐藏层，线性输出
        self.predict = torch.nn.Linear(n_hidden77, n_output)

class Net1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden11, n_hidden22, n_hidden33, n_hidden44, n_hidden55, n_hidden66, n_hidden77,
                 n_output):  # 定义层的信息，n_feature多少个输入, n_hidden每层神经元, n_output多少个输出
        super(Net1, self).__init__()  # 继承 __init__ 功能（固定）
        # 定义每层用什么样的形式
        self.hidden11 = torch.nn.Linear(n_feature, n_hidden11)
        self.hidden22 = torch.nn.Linear(n_hidden11, n_hidden22)
        self.hidden33 = torch.nn.Linear(n_hidden22, n_hidden33)
        self.hidden44 = torch.nn.Linear(n_hidden33, n_hidden44)
        self.hidden55 = torch.nn.Linear(n_hidden44, n_hidden55)
        self.hidden66 = torch.nn.Linear(n_hidden55, n_hidden66)
        self.hidden77 = torch.nn.Linear(n_hidden66, n_hidden77)
        # 定义隐藏层，线性输出
        self.predict = torch.nn.Linear(n_hidden77, n_output)  # 定义输出层线性输出

        pretrained_weights = torch.load("model/BPNN_adam1.pt")
        self.load_state_dict(pretrained_weights)

    def forward(self, sampleinnorm):
        hiddenout11 = torch.sigmoid(self.hidden11(sampleinnorm))
        hiddenout22 = torch.sigmoid(self.hidden22(hiddenout11))
        hiddenout33 = torch.sigmoid(self.hidden33(hiddenout22))
        hiddenout44 = torch.sigmoid(self.hidden44(hiddenout33))
        hiddenout55 = torch.sigmoid(self.hidden55(hiddenout44))
        hiddenout66 = torch.sigmoid(self.hidden66(hiddenout55))
        hiddenout77 = torch.sigmoid(self.hidden77(hiddenout66))

        networkout = self.predict(hiddenout77)
        return networkout


def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)


# 训练
def train(model, epochs, x_train, y_train):
    # optimizer 是训练的工具
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)
    # 训练与绘制训练图像
    plt.ion()  # 画图
    plt.show()
    for i in range(epochs):
        model.train()
        # net训练数据x, 输出预测值
        prediction = model(x_train)
        # 计算两者的均方误差
        loss = loss_func(prediction, y_train)
        # 上一步的更新梯度留在net.parameters()中，清空上一步的残余更新参数值
        optimizer.zero_grad()
        # 误差反向传播, 计算参数更新值
        loss.backward()
        # 更新参数
        optimizer.step()
        print(i)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epochs}
    torch.save(state, "model/BPNN_adam2.pt")


def test(model, x_test, y_test, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)
    optimizer.load_state_dict(checkpoint['optimizer'])
    output = model(x_test)
    return output


def scalableData1(normalizeData, minmax):
    # 由于进行了归一化，用最小值和最大值计算原始输出；
    # 计算样本原始数据的变化范围：最大值 - 最小值；
    diff = minmax[:, 1] - minmax[:, 0]

    # 将新权重计算的输出映射到 [0, 1] 范围；
    srcData = (normalizeData + 1) / 2
    # 将神经网络输出的第 1 个参数还原到原始的数据范围；
    for i in range(0, minmax.shape[0]):
        srcData[i] = srcData[i] * diff[i] + minmax[i][0]

    # srcData[0] = srcData[0] * diff[0] + minmax[0][0]
    # # 将神经网络输出的第 2 个参数还原到原始的数据范围；
    # srcData[1] = srcData[1] * diff[1] + minmax[1][0]
    return srcData


def plotGraph1(sampleout, errhistory, networkPredict):
    '''结果可视化；'''
    # global sampleout, errhistory, networkPredict
    # 输出转为 numpy；
    sampleout = np.array(sampleout)

    # 绘制第 1 张图；
    if errhistory != []:
        plt.figure(1)
        plt.plot(errhistory, label="err")
        plt.legend(loc='upper left')
        plt.show()

    # 绘制第 2 张图；
    plt.figure(2)
    # plt.subplot(2, 1, 1)
    plt.plot(sampleout[0],
             color="blue",
             linewidth=1.5,
             linestyle="-",
             label=u"Actual")

    plt.plot(networkPredict[0],
             color="red",
             linewidth=1.5,
             linestyle="--",
             label=u"Predict")
    plt.legend(loc='upper left')
    plt.show()
    # plt.draw()


def inverse_normalized(output, y_test):
    output = output * (np.max(y_test) - np.min(y_test)) + np.min(y_test)
    return output


def normalized(x_data, y_data):
    e = 1e-7  # 防止出现0
    for i in range(x_data.shape[1]):
        max_num = np.max(x_data[:, i])
        min_num = np.min(x_data[:, i])
        x_data[:, i] = (x_data[:, i] - min_num + e) / (max_num - min_num + e)
    y_data = (y_data - np.min(y_data) + e) / (np.max(y_data) - np.min(y_data) + e)
    return x_data, y_data


def haveTryBPNN():
    sampleinnorm, sampleoutnorm, sampleoutminmax, sampleout = make_data()  # normalizeData1(X_train, y_train) # 数据获取与归一化
    # 训练集，测试集划分
    # split_num = int(sampleinnorm.shape[1] * 0.98)
    # X_train = sampleinnorm[:, :split_num]
    # Y_train = sampleoutnorm[:, :split_num]
    # X_test = sampleinnorm[:, split_num:]
    # Y_test = sampleoutnorm[:, split_num:]

    X_train = sampleinnorm
    Y_train = sampleoutnorm
    X_test = sampleinnorm
    Y_test = sampleoutnorm

    bp = BPNN(findBest='SSA')
    bp.train(X_train, Y_train)
    networkPredict = bp.predict2(X_train)
    # 将预测好的结果缩放到正常的数据范围内；
    networkPredict = scalableData1(networkPredict, sampleoutminmax)
    # plotGraph1(sampleout[:, :split_num], bp.errhistory, networkPredict)
    # plotGraph1(sampleout, bp.errhistory, networkPredict)
    # print('Train set evaluation:', end=' ')
    # zhibiao = print_evaluate(sampleout[:, :split_num].tolist()[0], networkPredict.tolist()[0])
    zhibiao = print_evaluate(sampleout.tolist()[0], networkPredict.tolist()[0])

    # 测试集测试
    textPredict = bp.predict2(X_test)
    textPredict = scalableData1(textPredict, sampleoutminmax)
    # plotGraph1(sampleout[:, split_num:], [], textPredict)
    plotGraph1(sampleout, [], textPredict)
    # print('Test set evaluation:', end=' ')
    # zhibiao1 = print_evaluate(sampleout[:, split_num:].tolist()[0], textPredict.tolist()[0])
    zhibiao1 = print_evaluate(sampleout.tolist()[0], textPredict.tolist()[0])
    print()
    return zhibiao, zhibiao1


def main():
    x_train, y_train = DataSet_All()
    x_test, y_test = DataSet_All()
    # x_train, x_test, y_train, y_test = DataSet()
    # x_train, x_test, y_train, y_test = DataSet_Random(3)
    x_train_normalized, y_train_normalized = normalized(x_train, y_train)  # 归一化
    x_test_normalized, y_test_normalized = normalized(x_test, y_test)  # 归一化
    # 转换为tensor
    x_train_normalized = torch.from_numpy(x_train_normalized).clone().detach().float()
    y_train_normalized = torch.from_numpy(y_train_normalized).clone().detach().float()
    # y_train_normalized = torch.from_numpy(y_train_normalized).unsqueeze(1).clone().detach().float()
    x_test_normalized = torch.from_numpy(x_test_normalized).clone().detach().float()
    y_test_normalized = torch.from_numpy(y_test_normalized).clone().detach().float()

    # 初始化网络
    net = Net1(n_feature=2, n_hidden11=4, n_hidden22=6, n_hidden33=8, n_hidden44=10, n_hidden55=8, n_hidden66=6,
               n_hidden77=4, n_output=1)
    # 定义路径
    path_adam = "model/BPNN_adam2.pt"

    train(net, epochs=1000000, x_train=x_train_normalized, y_train=y_train_normalized)
    test(net, x_test_normalized, y_test_normalized, path_adam)

    output1 = test(net, x_train_normalized, y_train_normalized, path_adam)
    output1 = inverse_normalized(output1, y_train)
    output1 = output1.detach().numpy()

    output2 = test(net, x_test_normalized, y_test_normalized, path_adam)
    output2 = inverse_normalized(output2, y_test)
    output2 = output2.detach().numpy()

    # errors_std = np.std(np.array(output1) - np.array(y_train))
    # mse = mean_squared_error(output1, y_train)
    # plt.title("Adam test")
    # x = np.linspace(0, len(y_train), len(y_train))
    # plt.plot(x, y_train, color='blue', marker='.', label="train data")
    # plt.plot(x, output1, color='red', marker='.', label="predict data")
    # plt.xticks([])
    # plt.text(0, 97, 'error=%.4f' % errors_std, fontdict={'size': 15, 'color': 'red'})
    # plt.text(0, 92, 'mse=%.4f' % mse, fontdict={'size': 15, 'color': 'red'})
    # plt.legend(loc="upper right")
    # plt.savefig('model/Adam_train.png')
    # plt.show()

    errors_std = np.std(np.array(output2) - np.array(y_test))
    mse = mean_squared_error(output2, y_test)
    plt.title("Adam test")
    x = np.linspace(0, len(y_test), len(y_test))
    plt.plot(x, y_test, color='blue', marker='.', label="test data")
    plt.plot(x, output2, color='red', marker='.', label="predict data")
    plt.xticks([])
    plt.text(0, 97, 'error=%.4f' % errors_std, fontdict={'size': 15, 'color': 'red'})
    plt.text(0, 92, 'mse=%.4f' % mse, fontdict={'size': 15, 'color': 'red'})
    plt.legend(loc="upper right")
    plt.savefig('model/Adam_test.png')
    plt.show()


if __name__ == "__main__":
    # for i in range(0, 1):
    #     zhibiao, zhibiao1 = haveTryBPNN()
    #     zhibiao, zhibiao1 = np.array(zhibiao), np.array(zhibiao1)

    zhibiao, zhibiao1 = haveTryBPNN()
    main()
