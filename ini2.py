class Net(torch.nn.Module):
    def __init__(self, hiddenunitnum1=4, hiddenunitnum2=6, hiddenunitnum3=8, hiddenunitnum4=10, hiddenunitnum5=8,
                 hiddenunitnum6=6, hiddenunitnum7=4, maxepochs=10000, learnrate=0.001, findBest='SSA'):
        super(Net, self).__init__()  # 继承 __init__ 功能（固定）
        # 训练的次数；
        self.maxepochs = maxepochs
        # 学习率；
        self.learnrate = learnrate
        # 认为可以停止训练的理想误差，达到该误差值时停止训练；
        self.errorfinal = 10 ** -8
        # 搜索算法
        self.finfBest = findBest

        self.hiddenunitnum1 = hiddenunitnum1
        self.hiddenunitnum2 = hiddenunitnum2
        self.hiddenunitnum3 = hiddenunitnum3
        self.hiddenunitnum4 = hiddenunitnum4
        self.hiddenunitnum5 = hiddenunitnum5
        self.hiddenunitnum6 = hiddenunitnum6
        self.hiddenunitnum7 = hiddenunitnum7

        self.errhistory = []
        self.deltaStroy = np.array([])  # 存储梯度

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

    def forward(self, sampleinnorm):
        self.hiddenout11 = sigmoid((np.dot(self.w1, sampleinnorm).transpose() + self.b1.transpose())).transpose()
        self.hiddenout22 = sigmoid((np.dot(self.w2, self.hiddenout11).transpose() + self.b2.transpose())).transpose()
        self.hiddenout33 = sigmoid((np.dot(self.w3, self.hiddenout22).transpose() + self.b3.transpose())).transpose()
        self.hiddenout44 = sigmoid((np.dot(self.w4, self.hiddenout33).transpose() + self.b4.transpose())).transpose()
        self.hiddenout55 = sigmoid((np.dot(self.w5, self.hiddenout44).transpose() + self.b5.transpose())).transpose()
        self.hiddenout66 = sigmoid((np.dot(self.w6, self.hiddenout55).transpose() + self.b6.transpose())).transpose()
        self.hiddenout77 = sigmoid((np.dot(self.w7, self.hiddenout66).transpose() + self.b7.transpose())).transpose()

        # 定义激励函数(隐藏层的线性值)
        self.networkout = self.predict(self.hiddenout77)  # 输出层，输出值
        return self.networkout