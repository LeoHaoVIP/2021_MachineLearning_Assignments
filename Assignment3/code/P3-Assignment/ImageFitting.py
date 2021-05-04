# * coding: utf8 *
import numpy as np
from polynomial_curve_fitting.PolynomialFeature import *
from polynomial_curve_fitting.RidgeRegression import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))


# 读取图像数据
def load_data(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1], data[:, 2]


# 绘制原始图像
def plot_origin_image():
    plt.title('Pixel Sequence of Origin Image')
    plt.xlabel('row')
    plt.ylabel('column')
    # 绘制原始数据
    plt.plot(index, row, '.-', mfc="none", mec="r", ms=5, c='r', label='row-sequence-origin')
    plt.plot(index, column, '.-', mfc="none", mec="b", ms=5, c='b', label='column-sequence-origin')
    plt.plot(column, row, '.-', mfc="none", mec="y", ms=5, c='y', label='image-origin')
    # 注意原始数据从左上角开始编号，因此需要将纵坐标逆置
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()


# 绘制拟合结果
def plot_fitting_result():
    # 对图像横坐标执行拟合
    R, w_row = linear_fit(index, row, degree_row, alpha_row)
    # 对图像纵坐标执行拟合
    C, w_column = linear_fit(index, column, degree_column, alpha_column)
    plt.title('Linear Fitting Result')
    plt.xlabel('row')
    plt.ylabel('column')
    # 绘制拟合曲线
    plt.plot(index, R @ w_row, c="r", label="row-fitting-result")
    plt.plot(index, C @ w_column, c="b", label="column-fitting-result")
    # 注意原始数据从左上角开始编号，因此需要将纵坐标逆置
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()


# 由线性拟合结果生成图像
def fitting_result_to_image():
    # 对图像横坐标执行拟合
    x_row, w_row = linear_fit(index, row, degree_row, alpha_row)
    # 对图像纵坐标执行拟合
    x_column, w_column = linear_fit(index, column, degree_column, alpha_column)
    # 计算得到的纵坐标
    y_row = x_row @ w_row
    y_column = x_column @ w_column
    plt.title('Image Fitting Result')
    plt.xlabel('row')
    plt.ylabel('column')
    # 以y_column、y_row分为图像横坐标和纵坐标绘图
    plt.plot(y_column, y_row, label='image-fitting-result')
    plt.xlim((0, 100))
    # 注意原始数据从左上角开始编号，因此需要将纵坐标逆置
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
    # 图像生成
    image = np.ones((int(max(y_row)) + 1, int(max(y_column)) + 1), dtype=np.int8)
    for i in range(0, len(index)):
        image[int(y_row[i]), int(y_column[i])] = 0
    plt.title('Comparison between Origin and Fitting Result')
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.scatter(column, row, marker='.', c='y')
    patches = [mpatches.Patch(color='k', label='image-fitting-result'),
               mpatches.Patch(color='y', label='image-origin')]
    plt.legend(handles=patches, bbox_to_anchor=(0.8, 1))
    plt.show()
    pass


# 线性拟合，返回多项式模型参数
def linear_fit(x, y, degree, alpha):
    # 填充数据
    feature = PolynomialFeature(degree)
    X = feature.transform(x)
    # 引入正则化后，求解得到的参数w_lambda
    w_lambda = np.linalg.solve(alpha * np.eye(np.size(X, 1)) + X.T @ X, X.T @ y)
    return X, w_lambda


# 基于RMSE指标寻找最佳阶数degree
def find_best_degree_based_on_rmse():
    row_training_errors = []
    column_training_errors = []
    degrees = np.arange(2, 20, 2)
    for degree in degrees:
        X = PolynomialFeature(degree).transform(index)
        model_row = RidgeRegression(alpha_row)
        model_column = RidgeRegression(alpha_column)
        model_row.fit(X, row)
        model_column.fit(X, column)
        row_training_errors.append(rmse(model_row.predict(X), row))
        column_training_errors.append(rmse(model_column.predict(X), column))
    plt.title('Linear Fitting Result ($RMSE$)')
    plt.plot(degrees, row_training_errors, 'o-', mfc="none", mec="r", ms=10, c="r",
             label="row-fitting-RMSE ($\lambda=%.3f$)" % alpha_row)
    plt.plot(degrees, column_training_errors, 'o-', mfc="none", mec="b", ms=10, c="b",
             label="column-fitting-RMSE ($\lambda=%.3f$)" % alpha_column)
    plt.xlabel("$degree$")
    plt.ylabel("$RMSE$")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 读取数据
    index, row, column = load_data('./ImageData.txt')
    # 拟合模型阶数和对应的正则项系数
    degree_row, degree_column, alpha_row, alpha_column = 10, 10, 1e-3, 1e-3
    # 绘制原始图像
    plot_origin_image()
    # 获取拟合效果最佳的degree
    find_best_degree_based_on_rmse()
    # 绘制拟合结果
    plot_fitting_result()
    # 绘制拟合得到的目标图像
    fitting_result_to_image()
    pass
