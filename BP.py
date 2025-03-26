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
import time

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

def print_evaluate(true, predicted):
    mae = round(metrics.mean_absolute_error(true, predicted), 3)
    mse = round(metrics.mean_squared_error(true, predicted), 3)
    rmse = round(np.sqrt(metrics.mean_squared_error(true, predicted)), 3)
    mape = round(metrics.mean_absolute_percentage_error(true, predicted), 3)
    r2_square = round(metrics.r2_score(true, predicted), 3)
    error, errate = zongchazhi(true, predicted)
    print('MAE:', mae, end='   ')
    print('MSE:', mse, end='   ')
    print('RMSE:', rmse, end='   ')
    print('MAPE:', mape, end='   ')
    print('R2 Square:', r2_square)
    # print('__________________________________')
    return [mae, mse, rmse, mape, r2_square]

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
    HUM = np.genfromtxt("data/micro_HT0.csv", delimiter=",")
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

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden11, n_output):  # 定义层的信息，n_feature多少个输入, n_hidden每层神经元, n_output多少个输出
        super(Net, self).__init__()
        # 定义每层用什么样的形式
        self.hidden11 = torch.nn.Linear(n_feature, n_hidden11)
        self.predict = torch.nn.Linear(n_hidden11, n_output)

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
    torch.save(state, "bp/BPNN_adambp.pt")
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

def main():
    x_train, x_test, y_train, y_test = DataSet()
    x_train_normalized, y_train_normalized = normalized(x_train, y_train)  # 归一化
    x_test_normalized, y_test_normalized = normalized(x_test, y_test)  # 归一化
    # 转换为tensor
    x_train_normalized = torch.from_numpy(x_train_normalized).clone().detach().float()
    y_train_normalized = torch.from_numpy(y_train_normalized).clone().detach().float()
    # y_train_normalized = torch.from_numpy(y_train_normalized).unsqueeze(1).clone().detach().float()
    x_test_normalized = torch.from_numpy(x_test_normalized).clone().detach().float()
    y_test_normalized = torch.from_numpy(y_test_normalized).clone().detach().float()

    # 初始化网络
    net = Net(n_feature=2, n_hidden11=35, n_output=1)
    # 定义路径
    path_adam = "bp/BPNN_adambp.pt"

    Curve1, Curve2, Curve3 = train(net, epochs=1000, x_train=x_train_normalized, y_train=y_train_normalized, x_test = x_test_normalized, y_test = y_test_normalized)
    errhistory1 = Curve1.tolist()
    errhistory2 = Curve2.tolist()
    errhistory3 = Curve3.tolist()
    plotGraph2(errhistory1, errhistory2)
    # plotGraph3(errhistory1, errhistory2, errhistory3)
    err_data1 = [item[0] for item in errhistory1]
    df = pd.DataFrame({'Error': err_data1})
    df.to_excel('bp/error_history.xlsx', index=False)
    test(net, x_test_normalized, y_test_normalized, path_adam)

    output1 = test(net, x_train_normalized, y_train_normalized, path_adam)
    output1 = inverse_normalized(output1, y_train)
    output1 = output1.detach().numpy()

    output2 = test(net, x_test_normalized, y_test_normalized, path_adam)
    output2 = inverse_normalized(output2, y_test)
    output2 = output2.detach().numpy()

    # with open('bp/output2.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Actual', 'Predicted'])
    #     for actual, predicted in zip(y_test, output2):
    #         writer.writerow([actual[0], predicted[0]])

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
    plt.savefig('bp/Adam_train.png')
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
    plt.savefig('bp/Adam_test.png')
    plt.show()

    n1 = 138
    n2 = 15

    networkpredict = output1
    testpredict = output2
    # print('Train set evaluation:', end=' ')
    zhibiao = print_evaluate(y_train.tolist()[0], networkpredict.tolist()[0])
    # print('Test set evaluation:', end=' ')
    zhibiao1 = print_evaluate(y_test.tolist()[0], testpredict.tolist()[0])
    # print()

    zhibiao2 = np.vstack([zhibiao, zhibiao1])
    zhibiaoexcel = pd.DataFrame(zhibiao2)
    zhibiaoexcel.to_excel('bp/zhibiao_bp.xlsx', index=False, header=False)

    train_t2 = networkpredict
    train1 = pd.DataFrame(columns=['predict'], data=train_t2)
    # train.to_csv('csv_to_xlsx/train.csv', index=False, sep=',')
    train1.to_excel('bp/train.xlsx', sheet_name='data')

    test_t2 = testpredict
    test1 = pd.DataFrame(columns=['predict'], data=test_t2)
    # test.to_csv('csv_to_xlsx/test.csv', index=False, sep=',')
    test1.to_excel('bp/test.xlsx', sheet_name='data')

    # return zhibiao, zhibiao1

if __name__ == "__main__":

    # zhibiao, zhibiao1 = main()
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
