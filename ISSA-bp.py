### 该文件针对简易结构的bp网络
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import random
import torch
from torch import nn, dtype
from AQI_DataSet import *
import copy
import csv

learning_rate = 0.01
betas = (0.9, 0.999)
alpha = 0.9

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def relative_error(true_values, predicted_values, n):
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    mae_metrix = np.abs((true_values - predicted_values) / true_values)
    mre_metrix = mae_metrix / true_values
    re_sum = np.sum(mre_metrix)
    mre = re_sum / n
    return mre

def zongchazhi(y_test, y_test_pred):
    '''计算实际结果与预测结果的总差值,偏差率'''
    aaa = y_test
    bbb = y_test_pred
    sum1 = sum(aaa)
    sum2 = sum(aaa) - sum(bbb)
    rate = sum2 / sum1
    # print(f'总差值：{round(sum2, 3)}, 偏差率：{round(rate * 100, 2)}%', end='   ')
    return round(sum2, 3), round(rate * 100, 2)

def print_evaluate1(true, predicted, n1):
    mae = round(metrics.mean_absolute_error(true, predicted), 3)
    mse = round(metrics.mean_squared_error(true, predicted), 3)
    rmse = round(np.sqrt(metrics.mean_squared_error(true, predicted)), 3)
    mape = round(metrics.mean_absolute_percentage_error(true, predicted), 3)
    r2_square = round(metrics.r2_score(true, predicted), 4)
    error, errate = zongchazhi(true, predicted)
    print('MAE:', mae, end='   ')
    print('MSE:', mse, end='   ')
    print('RMSE:', rmse, end='   ')
    print('MAPE:', mape, end='   ')
    print('R2 Square:', r2_square)
    # print('__________________________________')
    return [mae, mse, rmse, mape, r2_square, error, errate]

# def print_evaluate2(true, predicted, n2):
#     mae = round(metrics.mean_absolute_error(true, predicted), 3)
#     mse = round(metrics.mean_squared_error(true, predicted), 3)
#     rmse = round(np.sqrt(metrics.mean_squared_error(true, predicted)), 3)
#     mre = round(relative_error(true, predicted, n2), 3)
#     r2_square = round(metrics.r2_score(true, predicted), 3)
#     error, errate = zongchazhi(true, predicted)
#     print('MAE:', mae, end='   ')
#     print('MSE:', mse, end='   ')
#     print('RMSE:', rmse, end='   ')
#     print('MRE:', mre, end='   ')
#     print('R2 Square:', r2_square)
#     # print('__________________________________')
#     return [mae, mse, rmse, mre, r2_square, error, errate]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_data():
    HUM = np.genfromtxt("data/cross-validation1/10.csv", delimiter=",")
    features = HUM[:, :2]
    labels = HUM[:, [2]]
    samplein = np.mat(features).T
    sampleout = np.mat(labels).T
    sampleinminmax = np.array(
        [samplein.min(axis=1).T.tolist()[0],
         samplein.max(axis=1).T.tolist()[0]]
    ).transpose()
    sampleoutminmax = np.array(
        [sampleout.min(axis=1).T.tolist()[0],
         sampleout.max(axis=1).T.tolist()[0]]
    ).transpose()
    sampleinnorm = (
            2 *
            (np.array(samplein.T) - sampleinminmax.transpose()[0]) /
            (sampleinminmax.transpose()[1] - sampleinminmax.transpose()[0])
            - 1
    ).transpose()
    sampleoutnorm = (
            2 *
            (np.array(sampleout.T).astype(float) - sampleoutminmax.transpose()[0]) /
            (sampleoutminmax.transpose()[1] - sampleoutminmax.transpose()[0])
            - 1
    ).transpose()
    return sampleinnorm, sampleoutnorm, sampleoutminmax, sampleout

class BPNN:
    def __init__(self, hiddenunitnum1=8, maxepochs=10000, learnrate=0.01, findBest='SSA'):
        self.maxepochs = maxepochs
        self.learnrate = learnrate
        self.errorfinal = 10 ** -8
        self.finfBest = findBest
        self.hiddenunitnum1 = hiddenunitnum1
        self.errhistory = []
        self.deltaStroy = np.array([])
        self.device = device

    def train(self, sampleinnorm, sampleoutnorm):
        self.sampleinnorm = torch.tensor(sampleinnorm, dtype=torch.float32)
        self.sampleoutnorm = torch.tensor(sampleoutnorm, dtype=torch.float32)
        self.samnum = sampleinnorm.shape[1]
        self.indim = sampleinnorm.shape[0]
        self.outdim = sampleoutnorm.shape[0]

        self.w1 = 2 * np.random.rand(self.hiddenunitnum1, self.indim) - 1
        self.b1 = 2 * np.random.rand(self.hiddenunitnum1, 1) - 1

        self.w0 = 2 * np.random.rand(self.outdim, self.hiddenunitnum1) - 1
        self.b0 = 2 * np.random.rand(self.outdim, 1) - 1

        self.deltaStroy = np.array([[np.zeros([self.hiddenunitnum1, self.indim]), np.zeros([self.hiddenunitnum1, 1]),
                                     np.zeros([self.outdim, self.hiddenunitnum1]), np.zeros([self.outdim, 1])]], dtype=object)  # 存储梯度
        self.deltaWB = np.array([[np.zeros([self.hiddenunitnum1, self.indim]), np.zeros([self.hiddenunitnum1, 1]),
                                     np.zeros([self.outdim, self.hiddenunitnum1]), np.zeros([self.outdim, 1])]], dtype=object)  # 存储累加量

        if self.finfBest == 'SSA':
            self.errorbackpropagateSSA()
            self.trial1()
            self.save_model()

    def errorbackpropagateSSA(self,):
        import SSA
        pop = 25
        Max_iter = 100
        dim = self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 + \
              self.hiddenunitnum1 * self.outdim + self.outdim

        lb = np.zeros((dim, 1)) - 2
        ub = np.zeros((dim, 1)) + 2

        fun = self.fun_cnn
        GbestScore, GbestPositon, Curve = SSA.Tent_SSA2(pop, dim, lb, ub, Max_iter, fun)
        self.XtoWB(GbestPositon.T)
        # 使用一个列表将误差存储下来；
        self.errhistory = Curve.tolist()

    def fun_cnn(self, X):
        self.XtoWB(X)
        networkout = self.predict2(self.sampleinnorm)
        sse = metrics.mean_squared_error(self.sampleoutnorm, networkout)
        return sse


    def predict2(self, sampleinnorm):
        self.hiddenout1 = sigmoid((np.dot(self.w1, sampleinnorm).transpose() + self.b1.transpose())).transpose()
        self.networkout = (np.dot(self.w0, self.hiddenout1).transpose() + self.b0.transpose()).transpose()
        return self.networkout

    def XtoWB(self, X):
        X = X[:]
        X = np.array(X)
        self.w1 = X[:self.indim * self.hiddenunitnum1].reshape(self.hiddenunitnum1, self.indim)
        self.b1 = X[self.indim * self.hiddenunitnum1:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1
                  ].reshape(self.hiddenunitnum1, 1)

        self.w0 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1:
                    self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 +
                    self.hiddenunitnum1 * self.outdim].reshape(self.outdim, self.hiddenunitnum1)
        self.b0 = X[self.indim * self.hiddenunitnum1 + self.hiddenunitnum1 +
                    self.hiddenunitnum1 * self.outdim:].reshape(self.outdim, 1)

    def trial1(self, ):
        self.w11 = torch.tensor(self.w1)
        self.b11 = torch.tensor(self.b1.reshape(self.hiddenunitnum1))
        self.w00 = torch.tensor(self.w0)
        self.b00 = torch.tensor(self.b0.reshape(self.outdim))

    def save_model(self,):
        self.state = {
            'hidden11.weight': self.w11,
            'hidden11.bias': self.b11,
            'predict.weight': self.w00,
            'predict.bias':  self.b00
        }
        torch.save(self.state, "C-V-output/model/10.pt")

def scalableData1(normalizeData, minmax):
    diff = minmax[:, 1] - minmax[:, 0]
    srcData = (normalizeData + 1) / 2
    for i in range(0, minmax.shape[0]):
        srcData[i] = srcData[i] * diff[i] + minmax[i][0]
    return srcData

def plotGraph1(sampleout, errhistory, networkPredict):
    sampleout = np.array(sampleout)

    if errhistory != []:
        plt.figure(1)
        plt.plot(errhistory, label="err")
        plt.legend(loc='upper left')
        plt.show()

    plt.figure(2)
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

def plotGraph2(errhistory1, errhistory2):
    if errhistory1 != []:
        plt.figure(1)
        plt.plot(errhistory1,
                 color="red",
                 linewidth=1.5,
                 linestyle="-",
                 label=u"err train")

        plt.plot(errhistory2,
                 color="blue",
                 linewidth=1.5,
                 linestyle="-",
                 label=u"err test")

        plt.legend(loc='upper left')
        plt.show()

def inverse_normalized(output, y_test):
    output = output * (np.max(y_test) - np.min(y_test)) + np.min(y_test)
    return output

def normalized(x_data, y_data):
    e = 1e-7
    for i in range(x_data.shape[1]):
        max_num = np.max(x_data[:, i])
        min_num = np.min(x_data[:, i])
        x_data[:, i] = (x_data[:, i] - min_num + e) / (max_num - min_num + e)
    y_data = (y_data - np.min(y_data) + e) / (np.max(y_data) - np.min(y_data) + e)
    return x_data, y_data

def haveTryBPNN():
    sampleinnorm, sampleoutnorm, sampleoutminmax, sampleout = make_data()
    # split_num = int(sampleinnorm.shape[1] * 0.9)
    split_num = 138
    X_train = sampleinnorm[:, :split_num]
    Y_train = sampleoutnorm[:, :split_num]
    X_test = sampleinnorm[:, split_num:]
    Y_test = sampleoutnorm[:, split_num:]
    # n1 = split_num
    # n2 = len(sampleout.T) - split_num
    n1 = 138
    n2 = 15

    bp = BPNN(findBest='SSA')
    bp.train(X_train, Y_train)
    networkPredict = bp.predict2(X_train)
    networkpredict = scalableData1(networkPredict, sampleoutminmax)
    plotGraph1(sampleout[:, :split_num], bp.errhistory, networkpredict)
    err_data = [item[0] for item in bp.errhistory]
    df = pd.DataFrame({'Error': err_data})
    df.to_excel('C-V-output/loss/10.xlsx', index=False)
    print('Train set evaluation:', end=' ')
    zhibiao = print_evaluate1(sampleout[:, :split_num].tolist()[0], networkpredict.tolist()[0], n1)

    testPredict = bp.predict2(X_test)
    testpredict = scalableData1(testPredict, sampleoutminmax)
    plotGraph1(sampleout[:, split_num:], [], testpredict)
    print('Test set evaluation:', end=' ')
    zhibiao1 = print_evaluate1(sampleout[:, split_num:].tolist()[0], testpredict.tolist()[0], n2)
    print()

    train_t2 = networkpredict.T
    train = pd.DataFrame(columns=['predict'], data=train_t2)
    # train.to_csv('csv_to_xlsx/train.csv', index=False, sep=',')
    train.to_excel('C-V-output/predict/train10.xlsx', sheet_name='data')

    test_t2 = testpredict.T
    test = pd.DataFrame(columns=['predict'], data= test_t2)
    # test.to_csv('csv_to_xlsx/test.csv', index=False, sep=',')
    test.to_excel('C-V-output/predict/test10.xlsx', sheet_name='data')

    return zhibiao, zhibiao1

if __name__ == "__main__":

    zhibiao, zhibiao1 = haveTryBPNN()




