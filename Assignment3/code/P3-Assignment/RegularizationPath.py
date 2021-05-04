# * coding: utf8 *
from polynomial_curve_fitting.PolynomialFeature import *
from polynomial_curve_fitting.RidgeRegression import *
import matplotlib.pyplot as plt

np.random.seed(2)


def func(x):
    return np.sin(2 * np.pi * x)


# 生成训练数据
def create_toy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    # 在func(x)的基础上添加高斯噪声
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))


# 绘制训练数据
def plot_train_data():
    # 绘制训练数据
    plt.title('Training Data')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")


# 绘制理想结果
def plot_ideal_result():
    # 绘制理想目标函数模型
    plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    plt.legend()
    plt.show()


# 绘制正则化曲线
def plot_regularization_path(degree):
    # 填充训练数据
    feature = PolynomialFeature(degree)
    X = feature.transform(x_train)
    # 未经正则化操作的w_infinite
    w_infinite = np.linalg.pinv(X) @ y_train
    # 正则项系数（范围可调节，0.04-100是较好的范围）
    alpha = np.linspace(0.04, 100, 10000)
    # 注意report中要对w随lambda的变化进行分析以及横坐标的范围
    w_lambda = None
    # 依次计算不同正则项系数下的 w_lambda
    for i in alpha:
        w_lambda_i = np.linalg.solve(i * np.eye(np.size(X, 1)) + X.T @ X, X.T @ y_train)
        w_lambda = w_lambda_i if w_lambda is None else np.vstack((w_lambda, w_lambda_i))
    # 横坐标 ||w_lambda||/||w_infinite||
    path_x = [np.linalg.norm(i) / np.linalg.norm(w_infinite) for i in w_lambda]
    # 绘制Regularization Path
    plt.title('Regularization Path (degree={0})'.format(degree))
    plt.xlabel('$||w_\lambda||/||w_∞||$')
    plt.ylabel('$w_i$')
    for i in range(0, degree + 1):
        # 依次绘制wi的正则化路径
        plt.plot(path_x, w_lambda[:, i], label='$w_{0}$'.format(i), linewidth='2')
        # plt.plot(path_x, abs(w_lambda[:, i]), label='$w_{0}$'.format(i), linewidth='2')
    plt.legend()
    plt.show()
    pass


# 绘制RMSE正则化曲线
def plot_rmse(degree):
    # 填充训练数据
    feature = PolynomialFeature(degree)
    X = feature.transform(x_train)
    # 未经正则化操作的w_infinite
    w_infinite = np.linalg.pinv(X) @ y_train
    # 正则项系数
    alpha = np.linspace(0.04, 100, 10000)
    # 注意report中要说明：lambda不能调过大，否则影响准确率（通过RMSE指标说明）
    # 不同正则项系数下的 w_lambda
    w_lambda = None
    # 训练集和测试集RMSE误差
    training_errors = []
    test_errors = []
    for i in alpha:
        w_lambda_i = np.linalg.solve(i * np.eye(np.size(X, 1)) + X.T @ X, X.T @ y_train)
        w_lambda = w_lambda_i if w_lambda is None else np.vstack((w_lambda, w_lambda_i))
        model = RidgeRegression(i)
        model.fit(X, y_train)
        training_errors.append(rmse(model.predict(X), y_train))
        test_errors.append(rmse(model.predict(feature.transform(x_test)), y_test))
    # 横坐标 ||w_lambda||/||w_infinite||
    path_x = [np.linalg.norm(i) / np.linalg.norm(w_infinite) for i in w_lambda]
    # 绘制RMSE曲线
    plt.title('RMSE (degree={0})'.format(degree))
    plt.xlabel('$||w_\lambda||/||w_∞||$')
    plt.ylabel("$RMSE$")
    plt.plot(path_x, training_errors, '-', mfc="none", mec="b", ms=10, c="b", label="Train")
    plt.plot(path_x, test_errors, '-', mfc="none", mec="r", ms=10, c="r", label="Test")
    plt.legend()
    plt.show()
    pass


if __name__ == '__main__':
    # 生成训练和测试数据
    x_train, y_train = create_toy_data(func, 10, 0.25)
    x_test = np.linspace(0, 1, 100)
    y_test = func(x_test)
    # 绘制正则化路径曲线
    plot_regularization_path(1)
    plot_rmse(1)
    plot_regularization_path(3)
    plot_rmse(3)
    plot_regularization_path(6)
    plot_rmse(6)
    plot_regularization_path(9)
    plot_rmse(9)
