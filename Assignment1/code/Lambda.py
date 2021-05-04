# * coding: utf8 *
import numpy as np
import matplotlib.pyplot as plt


class PolyFit:
    def __init__(self, order, path='./data.txt'):
        # 多项式阶数
        self.order = order
        # 多项式模型参数 (y=w0+w1*x+w2*x^2+w3*x^3+...) @Model
        self.w = np.zeros(order + 1).reshape(order + 1, 1)
        # 从文件读取数据 @BigData
        data = np.loadtxt(path)
        # 待拟合点个数
        self.m = len(data)
        # 自变量
        self.x = np.array(data[:, 0]).reshape(self.m, 1)
        self.x = np.hstack((np.ones((self.m, 1)), self.x))
        # 因变量
        self.y = np.array(data[:, 1]).reshape(self.m, 1)
        # 学习率
        self.alpha = 0.01

    # 损失函数 @Loss
    # 均方误差
    def loss_function(self):
        mse = 0
        for i in range(0, self.m):
            v = np.polyval(self.w[::-1], self.x[i])
            mse += (self.y[i] - v) ** 2
        return mse / (2 * self.m)

    # 计算梯度
    def gradient_function(self):
        diff = np.dot(self.x, self.w) - self.y
        return np.dot(np.transpose(self.x), diff) / self.m

    # 梯度下降 @Algorithm
    def gradient_descent(self):
        gradient = self.gradient_function()
        while not np.all(np.absolute(gradient) <= 1e-5):
            self.w = self.w - self.alpha * gradient
            gradient = self.gradient_function()


if __name__ == '__main__':
    my_fit = PolyFit(1, './data.txt')
    my_fit.gradient_descent()
    plt.scatter(my_fit.x[:, 1], my_fit.y)
    print(my_fit.w)
    func = np.poly1d(np.array([my_fit.w[1, 0], my_fit.w[0, 0]]).astype(float))
    x = np.linspace(-5, 10, 10)
    y = func(x)
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('PolyFit Result')
    plt.show()
