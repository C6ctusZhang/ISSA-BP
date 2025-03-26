import pandas as pd
from sklearn import metrics
import numpy as np

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
    mre = round(relative_error(true, predicted, n1), 5)
    r2_square = round(metrics.r2_score(true, predicted), 3)
    error, errate = zongchazhi(true, predicted)
    print('MAE:', mae, end='   ')
    print('MSE:', mse, end='   ')
    print('RMSE:', rmse, end='   ')
    print('MRE:', mre, end='   ')
    print('R2 Square:', r2_square)
    # print('__________________________________')
    return [mae, mse, rmse, mre, r2_square, error, errate]
def print_evaluate2(true, predicted, n2):
    mae = round(metrics.mean_absolute_error(true, predicted), 3)
    mse = round(metrics.mean_squared_error(true, predicted), 3)
    rmse = round(np.sqrt(metrics.mean_squared_error(true, predicted)), 3)
    mre = round(relative_error(true, predicted, n2), 4)
    r2_square = round(metrics.r2_score(true, predicted), 3)
    error, errate = zongchazhi(true, predicted)
    print('MAE:', mae, end='   ')
    print('MSE:', mse, end='   ')
    print('RMSE:', rmse, end='   ')
    print('MRE:', mre, end='   ')
    print('R2 Square:', r2_square)
    # print('__________________________________')
    return [mae, mse, rmse, mre, r2_square, error, errate]

def make_data1():
    HUM = np.genfromtxt("train.csv", delimiter=",")
    actual = HUM[:, :1]
    predict = HUM[:, [1]]
    actual = np.mat(actual).T
    predict = np.mat(predict).T

    return actual, predict

def make_data2():
    HUM = np.genfromtxt("test.csv", delimiter=",")
    actual = HUM[:, :1]
    predict = HUM[:, [1]]
    actual = np.mat(actual).T
    predict = np.mat(predict).T

    return actual, predict

def haveTryBPNN():
    actual1, predict1 = make_data1()
    actual2, predict2 = make_data2()
    n1 = 138
    n2 = 15
    print('Train set evaluation:', end=' ')
    zhibiao = print_evaluate1(actual1.tolist()[0], predict1.tolist()[0], n1)
    print('Test set evaluation:', end=' ')
    zhibiao1 = print_evaluate2(actual2.tolist()[0], predict2.tolist()[0], n2)
    print()
    return zhibiao, zhibiao1

if __name__ == "__main__":

    zhibiao, zhibiao1 = haveTryBPNN()