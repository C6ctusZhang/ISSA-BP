from torch.utils.tensorboard import SummaryWriter
import copy
import random
import numpy as np
import os
import scipy.io
from matplotlib import pyplot as plt
import pandas as pd
import torch
''' Tent种群初始化函数 '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initial0(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            round = random.uniform(-2, 2)  # Logistic混沌映射
            X[i, j] = round
    return X, lb, ub

def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    round = np.random.rand()
    for i in range(pop):
        for j in range(dim):
            round = 4 * round * (1 - round)  # Logistic混沌映射
            X[i, j] = round * (ub[j] - lb[j]) + lb[j]
    return X, lb, ub

def logistic_tent_map(x, r=4, a=0.5):
    logistic_value = r * x * (1 - x)
    if logistic_value < a:
        tent_value = logistic_value / a
    else:
        tent_value = (1 - logistic_value) / (1 - a)
    return tent_value

def initial1(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    round_value = np.random.rand()
    for i in range(pop):
        for j in range(dim):
            round_value = logistic_tent_map(round_value)
            X[i, j] = round_value * (ub[j] - lb[j]) + lb[j]
    return X, lb, ub

'''梯度检测边界更新函数'''


def dynamic_range_scaling(lb, ub, p):
    lb = lb - p * (ub - lb)
    ub = ub + p * (ub - lb)
    return lb, ub


'''边界检查函数'''


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                # X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]  # 超过边界后随机
                X[i, j] = ub[j]  # 超过边界后极值替换
            elif X[i, j] < lb[j]:
                # X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
                X[i, j] = lb[j]
    return X


'''计算适应度函数'''


def CalculateFitness(X, fun1, dim):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun1(X[i, :])
    # fitAvg = fitness.sum() / fitness.shape[0]
    # x1 = np.zeros([1, dim])
    # x1 = np.squeeze(x1)
    # round = np.random.rand()
    # for i in range(pop):
    #     if fitness[i] < fitAvg:
    #         for j in range(dim):
    #             x1[j] = X[i, j] * (random.gauss(0, 1) + 1)
    #         funx1 = fun(x1)
    #         if funx1 < fitness[i]:
    #             X[i] = x1
    #             fitness[i] = funx1
    #     else:
    #         for j in range(dim):
    #             round = 4 * round * (1 - round)  # Logistic混沌映射
    #             x1[j] = (X[i, j] + round) / 2
    #         funx1 = fun(x1)
    #         if funx1 < fitness[i]:
    #             X[i] = x1
    #             fitness[i] = funx1
    return fitness, X

# def CalculateFitness2(X, fun2, dim):
#     pop = X.shape[0]
#     fitness = np.zeros([pop, 1])
#     for i in range(pop):
#         fitness[i] = fun2(X[i, :])
#     return fitness, X

def discover_factor(N, PD, p):
    a = PD - np.power(-N ** 0.2, 3)
    F = (1 / (p - a) ** 3) + 1
    F_num = np.array(F)
    return F_num

def discover_factor1(N, PD, p):
    a = PD - np.power(-N ** 0.3, 1)
    F = (1 / (p - a) ** 1) + 1
    F_num = np.array(F)
    return F_num

def joiner_factor(N, PD, i):
    a = PD - np.power(- N ** 0.05, 5)
    F = (1 / (2 * PD - i - a) ** 5) + 1
    F_num = np.array(F)
    return F_num

def joiner_factor1(N, PD, i):
    a = PD - np.power(- N ** 0.3, 1)
    F = (1 / (2 * PD - i - a) ** 1) + 1
    F_num = np.array(F)
    return F_num

def spec_factor(ti, Max_iter):
    # a = np.array(1.5 * (ti - Max_iter))
    a = np.array((ti - Max_iter))
    F = 1 / (np.exp(a) + 1)
    F_num = np.array(F)
    return F_num

'''适应度排序'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)  # 按列升序
    index = np.argsort(Fit, axis=0)  # 按列升序，返回排序后的索引值的数组
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


'''麻雀发现者勘探更新'''


def PDUpdate(X, PDNumber, ST, Max_iter, dim):
    # 创建一个新的麻雀位置矩阵，与原始矩阵相同
    X_new = copy.copy(X)
    # 根据算法设计，取PDNumber的20%作为状态sta
    sta = 0.2 * PDNumber

    # 遍历部分发现者
    for p in range(PDNumber):
        # 随机生成一个0到1之间的随机数R2
        R2 = np.random.rand(1)  # 预警值
        # 遍历发现者的每个维度
        for j in range(dim):
            # 如果R2小于预警值ST，或者当前发现者序号小于1，即处于初始阶段
            if R2 < ST or p < 1:
                # 对当前维度进行位置更新，使用指数衰减函数，使得位置逐渐收敛到0
                X_new[p, j] = X[p, j] * np.exp(-p / (random.random() * Max_iter))
            else:
                # 如果R2大于预警值ST，并且当前发现者序号大于等于1
                # 则根据高斯分布随机数进行位置更新
                X_new[p, j] = X[p, j] + random.gauss(0, 1)  # random.gauss(0, 1)服从正态分布的均值为0，标准差为1的随机数
    return X_new


'''麻雀加入者更新'''


def JDUpdate(X, PDNumber, pop, dim):
    X_new = copy.copy(X)
    # 产生-1，1的随机数
    A = np.ones([dim, 1])
    round = np.random.rand()
    for a in range(dim):
        round = 4 * round * (1 - round)
        if (round > 0.5):
            A[a] = -1
    aa = np.linalg.inv(np.matrix(A.T) * np.matrix(A))
    for i in range(PDNumber + 1, pop):
        for j in range(dim):
            round = 4 * round * (1 - round)
            if i > (pop - PDNumber) / 2 + PDNumber:
                # 第i个加入者适应度较低且没有获得食物，处于十分饥饿的状态，需要飞往其它区域以补充能量
                X_new[i, j] = random.gauss(0, 1) * np.exp((X[-1, j] - X[i, j]) / i ** 2)
            else:
                # 当i<0.5n时，第i个加入者将在Xp附近随机觅食,此处Xp=X0
                AA = np.mean(np.abs(X[i, j] - X[0, :]) * A * np.array(aa))
                X_new[i, j] = X[0, j] - AA
    return X_new

def SDUpdate(X, pop, SDNumber, fitness, BestF):
    # 创建一个新的麻雀位置矩阵，与原始矩阵相同
    X_new = copy.copy(X)
    # 获取种群维度
    dim = X.shape[1]
    # 生成一个长度为 pop 的列表，存放从 0 到 pop-1 的整数
    Temp = range(pop)
    # 从 Temp 中随机抽取 pop 个不重复的元素
    RandIndex = random.sample(Temp, pop)
    # 获取随机抽取的前 SDNumber 个索引，即要更新的麻雀的索引
    SDchooseIndex = RandIndex[0:SDNumber]

    # 遍历要更新的麻雀
    for i in range(SDNumber):
        # 遍历麻雀的每个维度
        for j in range(dim):
            if fitness[SDchooseIndex[i]] > BestF:
                # 如果麻雀的适应度高于最佳适应度，则认为该麻雀处于种群的边缘，容易受到捕食者的攻击
                # 会靠近或远离最优点，位置更新为高斯分布随机数加权重
                X_new[SDchooseIndex[i], j] = X[0, j] + random.gauss(0, 1) * np.abs(X[SDchooseIndex[i], j] - X[0, j])
            elif fitness[SDchooseIndex[i]] == BestF:
                # 如果麻雀的适应度等于最佳适应度，则认为该麻雀处于种群中间，需要接近种群中其它麻雀以降低被捕食的概率
                # 随机生成一个步长控制参数 K，然后根据该参数更新麻雀位置
                K = 4 * random.random() - 2
                # 计算位置更新量，根据适应度之间的差异和步长控制参数来调整
                X_new[SDchooseIndex[i], j] = X[SDchooseIndex[i], j] + K * (
                        np.abs(X[SDchooseIndex[i], j] - X[-1, j]) / (fitness[SDchooseIndex[i]] - fitness[-1] + 10E-8))
    return X_new

def PDUpdate1(X, PDNumber, ST, Max_iter, dim, pop):
    X_new = copy.copy(X)
    A = np.ones([dim, 1])
    round = np.random.rand()
    for a in range(dim):
        round = 4 * round * (1 - round)
        if (round > 0.5):
            A[a] = -1
    aa = np.linalg.inv(np.matrix(A.T) * np.matrix(A))

    for p in range(PDNumber):
        R2 = np.random.rand(1)  # 预警值
        for j in range(dim):
            # 如果R2小于预警值ST，或者当前发现者序号小于1，即处于初始阶段
            if R2 < ST or p < 1:
                X_new[p, j] = X[p, j] * np.exp(-p / (random.random() * Max_iter))
            else:
                AA = np.mean(np.abs(X[p, j] - X[0, :]) * A * np.array(aa))
                p1 = discover_factor(pop, PDNumber, p)
                X_new[p, j] = X[p, j] + p1 * random.gauss(0, 1) + (1 - p1) * (
                            X[0, j] - AA)  # random.gauss(0, 1)服从正态分布的均值为0，标准差为1的随机数
    return X_new


def JDUpdate1(X, PDNumber, pop, Max_iter, dim):
    X_new = copy.copy(X)
    # 产生-1，1的随机数
    A = np.ones([dim, 1])
    round = np.random.rand()
    for a in range(dim):
        round = 4 * round * (1 - round)
        if (round > 0.5):
            A[a] = -1
    aa = np.linalg.inv(np.matrix(A.T) * np.matrix(A))
    for i in range(PDNumber + 1, pop):
        for j in range(dim):
            round = 4 * round * (1 - round)
            if i > (pop - PDNumber) / 2 + PDNumber:
                # 第i个加入者适应度较低且没有获得食物，处于十分饥饿的状态，需要飞往其它区域以补充能量
                X_new[i, j] = random.gauss(0, 1) * np.exp((X[-1, j] - X[i, j]) / i ** 2)
            else:
                # 当i<0.5n时，第i个加入者将在Xp附近随机觅食,此处Xp=X0
                p2 = joiner_factor(pop, PDNumber, i)
                AA = np.mean(np.abs(X[i, j] - X[0, :]) * A * np.array(aa))
                X_new[i, j] = p2 * (X[0, j] - AA) + (1 - p2) * (X[i, j] * np.exp(-i / (random.random() * Max_iter)))
    return X_new


def PDUpdate2(X, PDNumber, ST, Max_iter, dim, pop):
    X_new = copy.copy(X)
    A = np.ones([dim, 1])
    round = np.random.rand()
    for a in range(dim):
        round = 4 * round * (1 - round)
        if (round > 0.5):
            A[a] = -1
    aa = np.linalg.inv(np.matrix(A.T) * np.matrix(A))

    for p in range(PDNumber):
        R2 = np.random.rand(1)  # 预警值
        for j in range(dim):
            # 如果R2小于预警值ST，或者当前发现者序号小于1，即处于初始阶段
            if R2 < ST or p < 1:
                X_new[p, j] = X[p, j] * np.exp(-p / (random.random() * Max_iter))
            else:
                AA = np.mean(np.abs(X[p, j] - X[0, :]) * A * np.array(aa))
                p1 = discover_factor1(pop, PDNumber, p)
                X_new[p, j] = X[p, j] + p1 * random.gauss(0, 1) + (1 - p1) * (
                            X[0, j] - AA)  # random.gauss(0, 1)服从正态分布的均值为0，标准差为1的随机数
    return X_new


def JDUpdate2(X, PDNumber, pop, Max_iter, dim):
    X_new = copy.copy(X)
    # 产生-1，1的随机数
    A = np.ones([dim, 1])
    round = np.random.rand()
    for a in range(dim):
        round = 4 * round * (1 - round)
        if (round > 0.5):
            A[a] = -1
    aa = np.linalg.inv(np.matrix(A.T) * np.matrix(A))
    for i in range(PDNumber + 1, pop):
        for j in range(dim):
            round = 4 * round * (1 - round)
            if i > (pop - PDNumber) / 2 + PDNumber:
                # 第i个加入者适应度较低且没有获得食物，处于十分饥饿的状态，需要飞往其它区域以补充能量
                X_new[i, j] = random.gauss(0, 1) * np.exp((X[-1, j] - X[i, j]) / i ** 2)
            else:
                # 当i<0.5n时，第i个加入者将在Xp附近随机觅食,此处Xp=X0
                p2 = joiner_factor1(pop, PDNumber, i)
                AA = np.mean(np.abs(X[i, j] - X[0, :]) * A * np.array(aa))
                X_new[i, j] = p2 * (X[0, j] - AA) + (1 - p2) * (X[i, j] * np.exp(-i / (random.random() * Max_iter)))
    return X_new

'''警戒者更新'''


def SDUpdate1(X, pop, SDNumber, fitness, BestF, ti, Max_iter):
    # 创建一个新的麻雀位置矩阵，与原始矩阵相同
    X_new = copy.copy(X)
    # 获取种群维度
    dim = X.shape[1]
    # 生成一个长度为 pop 的列表，存放从 0 到 pop-1 的整数
    Temp = range(pop)
    # 从 Temp 中随机抽取 pop 个不重复的元素
    RandIndex = random.sample(Temp, pop)
    # 获取随机抽取的前 SDNumber 个索引，即要更新的麻雀的索引
    SDchooseIndex = RandIndex[0:SDNumber]

    # 遍历要更新的麻雀
    for i in range(SDNumber):
        # 遍历麻雀的每个维度
        for j in range(dim):
            if fitness[SDchooseIndex[i]] > BestF:
                # 如果麻雀的适应度高于最佳适应度，则认为该麻雀处于种群的边缘，容易受到捕食者的攻击
                # 会靠近或远离最优点，位置更新为高斯分布随机数加权重
                X_new[SDchooseIndex[i], j] = X[0, j] + random.gauss(0, 1) * np.abs(X[SDchooseIndex[i], j] - X[0, j])
            elif fitness[SDchooseIndex[i]] == BestF:
                # 如果麻雀的适应度等于最佳适应度，则认为该麻雀处于种群中间，需要接近种群中其它麻雀以降低被捕食的概率
                # 随机生成一个步长控制参数 K，然后根据该参数更新麻雀位置
                p3 = spec_factor(ti, Max_iter)
                K = 4 * random.random() - 2
                # 计算位置更新量，根据适应度之间的差异和步长控制参数来调整
                X_new[SDchooseIndex[i], j] = X[SDchooseIndex[i], j] + p3 * K * (
                        np.abs(X[SDchooseIndex[i], j] - X[-1, j]) / (fitness[SDchooseIndex[i]] - fitness[-1] + 10E-8))
    return X_new


'''麻雀搜索算法'''


def Tent_SSA(pop, dim, lb, ub, Max_iter, fun1):
    '''
    输入：pop=>麻雀个体数量； dim=>目标函数变量空间的维数  lb=>下边界 ub=>上边界  fun=>适应度计算函数  Max_iter=>最大迭代次数
    返回：GbestScore=>全局最优适应度, GbestPositon=>最优参数, Curve=>全局最优适应度变化nparray
    '''

    # 初始化种群
    ST = 0.8  # 安全阈值Safe Threshold [0.5,1]
    PD = 0.3  # 发现者的比例，剩下的是加入者
    SD = 0.2  # 意识到有危险麻雀的比重
    PDNumber = int(pop * PD)  # 发现者数量
    SDNumber = int(pop * SD)  # 意识到有危险麻雀数量
    X, lb, ub = initial0(pop, dim, ub, lb)  # 初始化种群

    # 计算适应度值
    fitness, X = CalculateFitness(X, fun1, dim)  # 计算适应度值
    # fitness2, X2 = CalculateFitness2(X, fun2, dim)

    # 根据适应度排序，同时保留个体本身的顺序信息，防止在迭代过程中个体顺序混乱
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序

    # 初始化全局最优适应度值 GbestScore、全局最优参数值 GbestPositon 和适应度值变化曲线 Curve
    GbestScore = copy.copy(fitness[0])
    # GbestScore2 = copy.copy(fitness2[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon2 = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])
    GbestPositon2[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([Max_iter, 1])
    Curve2 = np.zeros([Max_iter, 1])

    for i in range(Max_iter):
        BestF = fitness[0]

        # lb, ub = dynamic_range_scaling(lb, ub, p)

        X = PDUpdate(X, PDNumber, ST, Max_iter, dim)  # 发现者更新
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        X = JDUpdate(X, PDNumber, pop, dim)  # 加入者更新
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        X = SDUpdate(X, pop, SDNumber, fitness, BestF)  # 警戒者更新
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        fitness, X = CalculateFitness(X, fun1, dim)  # 计算适应度值
        # fitness2, X2 = CalculateFitness2(X, fun2, dim)

        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        # fitness2, sortIndex2 = SortFitness(fitness2)
        X = SortPosition(X, sortIndex)  # 种群排序
        if (fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        # if (fitness2[0] <= GbestScore2):  # 更新全局最优
        #     GbestScore2 = copy.copy(fitness2[0])
        #     GbestPositon2[0, :] = copy.copy(X2[0, :])
        Curve[i] = GbestScore  # 最优适应度
        # Curve2[i] = GbestScore2
        if GbestScore[0] < 0.001:
            break
        # Curve[i] = ditAvg  # 平均适应度
        print(i)
    # best_fitness_history = np.array([best_fitness_history])
    # worst_fitness_history = np.array([worst_fitness_history])
    # with pd.ExcelWriter('fitness_history.xlsx', engine='xlsxwriter') as writer:
    #     pd.DataFrame({'Best Fitness': best_fitness_history[0], 'Worst Fitness': worst_fitness_history[0]}).to_excel(
    #         writer, index=False, float_format='%.8f')
    return GbestScore, GbestPositon, Curve


def Tent_SSA1(pop, dim, lb, ub, Max_iter, fun1):
    '''
    输入：pop=>麻雀个体数量； dim=>目标函数变量空间的维数  lb=>下边界 ub=>上边界  fun=>适应度计算函数  Max_iter=>最大迭代次数
    返回：GbestScore=>全局最优适应度, GbestPositon=>最优参数, Curve=>全局最优适应度变化nparray
    '''

    # 初始化种群
    ST = 0.7  # 安全阈值Safe Threshold [0.5,1]
    PD = 0.4  # 发现者的比例，剩下的是加入者
    SD = 0.2  # 意识到有危险麻雀的比重
    PDNumber = int(pop * PD)  # 发现者数量
    SDNumber = int(pop * SD)  # 意识到有危险麻雀数量
    '''种群初始化感觉可以做一些创新的地方'''
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群

    # 计算适应度值
    fitness, X = CalculateFitness(X, fun1, dim)  # 计算适应度值

    # 根据适应度排序，同时保留个体本身的顺序信息，防止在迭代过程中个体顺序混乱
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序

    # 初始化全局最优适应度值 GbestScore、全局最优参数值 GbestPositon 和适应度值变化曲线 Curve
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([Max_iter, 1])

    best_fitness_history = []
    worst_fitness_history = []

    for i in range(Max_iter):
        BestF = fitness[0]
        # WorstF = fitness[-1]
        best_fitness_history.append(BestF)
        # worst_fitness_history.append(WorstF)
        # lb, ub = dynamic_range_scaling(lb, ub, p)

        X = PDUpdate1(X, PDNumber, ST, Max_iter, dim, pop)  # 发现者更新
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        X = JDUpdate1(X, PDNumber, pop, Max_iter, dim)  # 加入者更新
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        X = SDUpdate1(X, pop, SDNumber, fitness, BestF, i, Max_iter)  # 警戒者更新
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        fitness, X = CalculateFitness(X, fun1, dim)  # 计算适应度值

        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if (fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[i] = GbestScore  # 最优适应度
        if GbestScore[0] < 0.001:
            break
        # Curve[i] = ditAvg  # 平均适应度
        print(i)
    # best_fitness_history = np.array([best_fitness_history])
    # worst_fitness_history = np.array([worst_fitness_history])
    # with pd.ExcelWriter('fitness_history.xlsx', engine='xlsxwriter') as writer:
    #     pd.DataFrame({'Best Fitness': best_fitness_history[0], 'Worst Fitness': worst_fitness_history[0]}).to_excel(
    #         writer, index=False, float_format='%.8f')
    # df = pd.DataFrame({'Best Fitness': best_fitness_history, 'Worst Fitness': worst_fitness_history})

    # df = pd.DataFrame({'Best Fitness': best_fitness_history})
    # df.to_excel('fitness_history.xlsx')

    return GbestScore, GbestPositon, Curve


def Tent_SSA2(pop, dim, lb, ub, Max_iter, fun1):
    '''
    输入：pop=>麻雀个体数量； dim=>目标函数变量空间的维数  lb=>下边界 ub=>上边界  fun=>适应度计算函数  Max_iter=>最大迭代次数
    返回：GbestScore=>全局最优适应度, GbestPositon=>最优参数, Curve=>全局最优适应度变化nparray
    '''

    # 初始化种群
    ST = 0.7  # 安全阈值Safe Threshold [0.5,1]
    PD = 0.3  # 发现者的比例，剩下的是加入者
    SD = 0.2  # 意识到有危险麻雀的比重
    PDNumber = int(pop * PD)  # 发现者数量
    SDNumber = int(pop * SD)  # 意识到有危险麻雀数量
    '''种群初始化感觉可以做一些创新的地方'''
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群

    # 计算适应度值
    fitness, X = CalculateFitness(X, fun1, dim)  # 计算适应度值

    # 根据适应度排序，同时保留个体本身的顺序信息，防止在迭代过程中个体顺序混乱
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序

    # 初始化全局最优适应度值 GbestScore、全局最优参数值 GbestPositon 和适应度值变化曲线 Curve
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([Max_iter, 1])

    best_fitness_history = []
    worst_fitness_history = []

    for i in range(Max_iter):
        BestF = fitness[0]
        # WorstF = fitness[-1]
        best_fitness_history.append(BestF)
        # worst_fitness_history.append(WorstF)
        # lb, ub = dynamic_range_scaling(lb, ub, p)

        X = PDUpdate2(X, PDNumber, ST, Max_iter, dim, pop)  # 发现者更新
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        X = JDUpdate2(X, PDNumber, pop, Max_iter, dim)  # 加入者更新
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        X = SDUpdate1(X, pop, SDNumber, fitness, BestF, i, Max_iter)  # 警戒者更新
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        fitness, X = CalculateFitness(X, fun1, dim)  # 计算适应度值

        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if (fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[i] = GbestScore  # 最优适应度
        if GbestScore[0] < 0.001:
            break
        # Curve[i] = ditAvg  # 平均适应度
        print(i)
    # best_fitness_history = np.array([best_fitness_history])
    # worst_fitness_history = np.array([worst_fitness_history])
    # with pd.ExcelWriter('fitness_history.xlsx', engine='xlsxwriter') as writer:
    #     pd.DataFrame({'Best Fitness': best_fitness_history[0], 'Worst Fitness': worst_fitness_history[0]}).to_excel(
    #         writer, index=False, float_format='%.8f')
    # df = pd.DataFrame({'Best Fitness': best_fitness_history, 'Worst Fitness': worst_fitness_history})

    # df = pd.DataFrame({'Best Fitness': best_fitness_history})
    # df.to_excel('fitness_history.xlsx')

    return GbestScore, GbestPositon, Curve
