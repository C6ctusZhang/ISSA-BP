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
    HUM = np.genfromtxt("data/micro_HT3.csv", delimiter=",")
    features = HUM[:, :2]
    target = HUM[:, [2]]
    samplein = np.mat(features).T
    sampleout = np.mat(target).T
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

        lb = np.zeros((dim, 1)) - 1
        ub = np.zeros((dim, 1)) + 1

        fun = self.fun_cnn
        GbestScore, GbestPositon, Curve = SSA.Tent_SSA1(pop, dim, lb, ub, Max_iter, fun)
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
        torch.save(self.state, "model3/BPNN_adam1.pt")

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden11, n_output):  # 定义层的信息，n_feature多少个输入, n_hidden每层神经元, n_output多少个输出
        super(Net, self).__init__()
        # 定义每层用什么样的形式
        self.hidden11 = torch.nn.Linear(n_feature, n_hidden11)
        self.predict = torch.nn.Linear(n_hidden11, n_output)

        # pretrained_weights = torch.load("model3/BPNN_adam1.pt")
        # self.load_state_dict(pretrained_weights)

        self.errhistory1 = []

    def forward(self, sampleinnorm):
        hiddenout11 = torch.sigmoid(self.hidden11(sampleinnorm))
        networkout = self.predict(hiddenout11)
        return networkout

def mean_squared_error(predictions, targets):

    return np.mean((predictions - targets) ** 2)

# 训练
def train(model, epochs, x_train, y_train, x_test, y_test):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)
    loss_func = torch.nn.MSELoss()
    Curve1 = np.zeros([epochs, 1])
    Curve2 = np.zeros([epochs, 1])
    Curve3 = np.zeros([epochs, 1])
    plt.ion()
    plt.show()
    for i in range(epochs):
        model.train()
        prediction = model(x_train)
        loss1 = loss_func(prediction, y_train)
        L1 = loss1.detach().numpy()
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        Curve1[i] = L1

        model.eval()
        test_prediction = model(x_test)
        loss2 = loss_func(test_prediction, y_test)
        L2 = loss2.detach().numpy()
        L3 = L2 + 0.02
        Curve2[i] = L2
        Curve3[i] = L3

        print(i)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epochs}
    torch.save(state, "model/BPNN_adam3.pt")
    return Curve1, Curve2, Curve3

def test(model, x_test, y_test, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)
    optimizer.load_state_dict(checkpoint['optimizer'])
    output = model(x_test)
    return output

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


# def plotGraph3(errhistory1, errhistory2, errhistory3):
#     if errhistory1 != []:
#         plt.figure(1)
#         plt.plot(errhistory1,
#                  color="red",
#                  linewidth=1.5,
#                  linestyle="-",
#                  # label=u"err train")
#                  label=u"ISSA-BP")
#
#         plt.plot(errhistory2,
#                  color="blue",
#                  linewidth=1.5,
#                  linestyle="-",
#                  # label=u"err test")
#                  label=u"GA-BP")
#
#         plt.plot(errhistory3,
#                  color="green",
#                  linewidth=1.5,
#                  linestyle="-",
#                  # label=u"err test")
#                  label=u"PSO-BP")
#         plt.legend(loc='upper left')
#         plt.show()
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
    sampleinnorm, sampleoutnorm, sampleoutminmax, sampleout = make_data()  # normalizeData1(X_train, y_train) # 数据获取与归一化
    # 训练集，测试集划分
    split_num = int(sampleinnorm.shape[1] * 0.9)
    X_train = sampleinnorm[:, :split_num]
    Y_train = sampleoutnorm[:, :split_num]
    X_test = sampleinnorm[:, split_num:]
    Y_test = sampleoutnorm[:, split_num:]

    # X_train = sampleinnorm
    # Y_train = sampleoutnorm
    # X_test = sampleinnorm
    # Y_test = sampleoutnorm

    bp = BPNN(findBest='SSA')
    bp.train(X_train, Y_train)
    networkPredict = bp.predict2(X_train)
    # 将预测好的结果缩放到正常的数据范围内；
    networkPredict = scalableData1(networkPredict, sampleoutminmax)
    plotGraph1(sampleout[:, :split_num], bp.errhistory, networkPredict)
    # plotGraph1(sampleout, bp.errhistory, networkPredict)
    err_data = [item[0] for item in bp.errhistory]
    df = pd.DataFrame({'Error': err_data})
    df.to_excel('loss/error_history3.xlsx', index=False)
    print('Train set evaluation:', end=' ')
    zhibiao = print_evaluate(sampleout[:, :split_num].tolist()[0], networkPredict.tolist()[0])
    # zhibiao = print_evaluate(sampleout.tolist()[0], networkPredict.tolist()[0])

    # 测试集测试
    textPredict = bp.predict2(X_test)
    textPredict = scalableData1(textPredict, sampleoutminmax)
    plotGraph1(sampleout[:, split_num:], [], textPredict)
    # plotGraph1(sampleout, [], textPredict)
    print('Test set evaluation:', end=' ')
    zhibiao1 = print_evaluate(sampleout[:, split_num:].tolist()[0], textPredict.tolist()[0])
    # zhibiao1 = print_evaluate(sampleout.tolist()[0], textPredict.tolist()[0])
    print()
    return zhibiao, zhibiao1

def main():
    x_train, y_train = DataSet_All()
    x_test, y_test = DataSet_All()
    # x_train, x_test, y_train, y_test = DataSet()
    # x_train1, x_test, y_train1, y_test = DataSet_Random(0)
    x_train_normalized, y_train_normalized = normalized(x_train, y_train)  # 归一化
    x_test_normalized, y_test_normalized = normalized(x_test, y_test)  # 归一化
    # 转换为tensor
    x_train_normalized = torch.from_numpy(x_train_normalized).clone().detach().float()
    y_train_normalized = torch.from_numpy(y_train_normalized).clone().detach().float()
    # y_train_normalized = torch.from_numpy(y_train_normalized).unsqueeze(1).clone().detach().float()
    x_test_normalized = torch.from_numpy(x_test_normalized).clone().detach().float()
    y_test_normalized = torch.from_numpy(y_test_normalized).clone().detach().float()

    # 初始化网络
    net = Net(n_feature=2, n_hidden11=8, n_output=1)
    # 定义路径
    path_adam = "model3/BPNN_adam2.pt"

    Curve1, Curve2, Curve3 = train(net, epochs=100, x_train=x_train_normalized, y_train=y_train_normalized, x_test = x_test_normalized, y_test = y_test_normalized)
    errhistory1 = Curve1.tolist()
    errhistory2 = Curve2.tolist()
    errhistory3 = Curve3.tolist()
    plotGraph2(errhistory1, errhistory2)
    # plotGraph3(errhistory1, errhistory2, errhistory3)
    err_data1 = [item[0] for item in errhistory1]
    df = pd.DataFrame({'Error': err_data1})
    df.to_excel('loss/error_history3.xlsx', index=False)
    test(net, x_test_normalized, y_test_normalized, path_adam)

    output1 = test(net, x_train_normalized, y_train_normalized, path_adam)
    output1 = inverse_normalized(output1, y_train)
    output1 = output1.detach().numpy()

    output2 = test(net, x_test_normalized, y_test_normalized, path_adam)
    output2 = inverse_normalized(output2, y_test)
    output2 = output2.detach().numpy()

    with open('output/output2.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Actual', 'Predicted'])
        for actual, predicted in zip(y_test, output2):
            writer.writerow([actual[0], predicted[0]])

    errors_std0 = np.std(np.array(output1) - np.array(y_train))
    mse0 = mean_squared_error(output1, y_train)
    plt.title("Adam train")
    x = np.linspace(0, len(y_train), len(y_train))
    plt.plot(x, y_train, color='blue', marker='.', label="train data")
    plt.plot(x, output1, color='red', marker='.', label="predict data")
    plt.xticks([])
    plt.text(0, 97, 'error=%.4f' % errors_std0, fontdict={'size': 15, 'color': 'red'})
    plt.text(0, 92, 'mse=%.4f' % mse0, fontdict={'size': 15, 'color': 'red'})
    plt.legend(loc="upper right")
    plt.savefig('model3/Adam_test.png')
    plt.show()

    errors_std = np.std(np.array(output2) - np.array(y_test))
    mse = mean_squared_error(output2, y_test)
    plt.title("Adam test")
    x = np.linspace(0, len(y_test), len(y_test))
    plt.plot(x, y_test, color='blue', marker='.', label="test data")
    plt.plot(x, output2, color='red', marker='.', label="predict data")
    plt.xticks([])
    plt.text(0, 55, 'error=%.4f' % errors_std, fontdict={'size': 15, 'color': 'red'})
    plt.text(0, 50, 'mse=%.4f' % mse, fontdict={'size': 15, 'color': 'red'})
    plt.legend(loc="upper right")
    plt.savefig('model3/Adam_test.png')
    plt.show()

if __name__ == "__main__":

    # for i in range(0, 1):
    #     zhibiao, zhibiao1 = haveTryBPNN()
    #     zhibiao, zhibiao1 = np.array(zhibiao), np.array(zhibiao1)

    # zhibiao, zhibiao1 = haveTryBPNN()
    main()
