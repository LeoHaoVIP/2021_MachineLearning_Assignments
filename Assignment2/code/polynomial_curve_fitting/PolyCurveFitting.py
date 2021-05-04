# * coding: utf8 *
import numpy as np
import matplotlib.pyplot as plt
from polynomial_curve_fitting.PolynomialFeature import *
from polynomial_curve_fitting.LinearRegression import *
from polynomial_curve_fitting.RidgeRegression import *

np.random.seed(2)


def func(x):
    return np.sin(2 * np.pi * x)


def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))


# 生成训练数据
def create_toy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    # 在func(x)的基础上添加高斯噪声
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


# 普通最小二乘法多项式拟合
def simple_poly_fit():
    for i, degree in enumerate([0, 1, 3, 6, 9]):
        feature = PolynomialFeature(degree)
        X_train = feature.transform(x_train)
        X_test = feature.transform(x_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y = model.predict(X_test)
        plt.title('Curve Fitting Result, M=%d' % degree)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
        plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
        plt.plot(x_test, y, c="r", label="fitting")
        plt.ylim(-1.5, 1.5)
        plt.annotate("M={}".format(degree), xy=(-0.15, 1))
        plt.legend()
        plt.show()


# 最小二乘法多项式+正则化
def poly_fit_regularized():
    for i, degree in enumerate([0, 1, 3, 6, 9]):
        feature = PolynomialFeature(degree)
        X_train = feature.transform(x_train)
        X_test = feature.transform(x_test)

        model = RidgeRegression(alpha=1e-3)
        model.fit(X_train, y_train)
        y = model.predict(X_test)
        plt.title('Curve Fitting Result with Regularization, M=%d' % degree)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
        plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
        plt.plot(x_test, y, c="r", label="fitting")
        plt.ylim(-1.5, 1.5)
        plt.annotate("M={}".format(degree), xy=(-0.15, 1))
        plt.legend()
        plt.show()


# RMSE误差衡量1
def rmse_measure_1():
    training_errors = []
    test_errors = []
    for i in range(10):
        feature = PolynomialFeature(i)
        X_train = feature.transform(x_train)
        X_test = feature.transform(x_test)

        model = LinearRegression()
        # model = RidgeRegression()
        model.fit(X_train, y_train)
        y = model.predict(X_test)
        training_errors.append(rmse(model.predict(X_train), y_train))
        # test_errors.append(rmse(model.predict(X_test), y_test + np.random.normal(scale=0.25, size=len(y_test))))
        test_errors.append(rmse(model.predict(X_test), y_test))
    plt.title('Curve Fitting Result with Training Size=%d' % len(X_train))
    # plt.title('Curve Fitting Result with $ln(λ)$=%.2f' % 1.0)
    plt.plot(training_errors, 'o-', mfc="none", mec="b", ms=10, c="b", label="Training")
    plt.plot(test_errors, 'o-', mfc="none", mec="r", ms=10, c="r", label="Test")
    plt.legend()
    plt.xlabel("$degree$")
    plt.ylabel("$RMSE$")
    plt.show()


# RMSE误差衡量2
def rmse_measure_2():
    training_errors = []
    test_errors = []
    alpha_ln = np.linspace(-50, 0, 100)
    alpha = np.exp(alpha_ln)
    degree = 9
    for i in alpha:
        feature = PolynomialFeature(degree)
        X_train = feature.transform(x_train)
        X_test = feature.transform(x_test)

        # model = LinearRegression()
        model = RidgeRegression(i)
        model.fit(X_train, y_train)
        training_errors.append(rmse(model.predict(X_train), y_train))
        test_errors.append(rmse(model.predict(X_test), y_test))
    plt.title('Curve Fitting Result with $M$=%d' % degree)
    plt.plot(alpha_ln, training_errors, '-', mfc="none", mec="b", ms=10, c="b", label="Training")
    plt.plot(alpha_ln, test_errors, '-', mfc="none", mec="r", ms=10, c="r", label="Test")
    plt.xlim((-40, 0))
    plt.ylim((0, 1))
    plt.legend()
    plt.xlabel("$ln(λ)$")
    plt.ylabel("$RMSE$")
    plt.show()


if __name__ == '__main__':
    # 生成训练和测试数据
    x_train, y_train = create_toy_data(func, 10, 0.25)
    x_test = np.linspace(0, 1, 100)
    y_test = func(x_test)
    # 绘制训练数据
    plt.title('Training Data')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
    # 绘制理想目标函数模型
    plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    plt.legend()
    plt.show()
    # 普通最小二乘法多项式拟合
    simple_poly_fit()
    # RMSE误差衡量
    rmse_measure_1()
    rmse_measure_2()
    # 最小二乘法+正则化拟合
    poly_fit_regularized()
    pass
