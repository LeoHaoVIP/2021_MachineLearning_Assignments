# * coding: utf8 *
import numpy as np
import collections
import logistic_regression.DatasetUtils as DataSetUtils
import logistic_regression.LogisticUtils as LogisticUtils
import logistic_regression.PlotUtils as PlotUtils


def random_select(train_data, test_data, label_train, label_test):
    train_index = np.loadtxt('./dataset/train_index.txt', dtype=np.int32)
    test_index = np.loadtxt('./dataset/test_index.txt', dtype=np.int32)
    train_data = train_data[train_index]
    label_train = label_train[train_index]
    test_data = test_data[test_index]
    label_test = label_test[test_index]
    return train_data, test_data, label_train, label_test


if __name__ == '__main__':
    # 加载训练和测试数据
    # data:28x28; label: 1
    print('-> loading dataset...')
    train_data, test_data, label_train, label_test = DataSetUtils.load_data_set()
    # 使用预生成的随机序列选取7000个训练样本和3000个测试样本
    train_data, test_data, label_train, label_test = random_select(train_data, test_data, label_train, label_test)
    print(collections.Counter(label_train))
    print(collections.Counter(label_test))
    # 创建Logistic回归实例
    print('-> creating model...')
    # logistic_instance = LogisticUtils.LogisticRegressionML(train_data, test_data, label_train, label_test,
    #                                                        epochs=5, batch_size=10, lr=0.001, gd_type='normal')
    logistic_instance = LogisticUtils.LogisticRegressionML(train_data, test_data, label_train, label_test, epochs=1,
                                                           batch_size=10, lr=0.1, _lambda=1e-5,
                                                           gd_type='regularization')
    # logistic_instance = LogisticUtils.LogisticRegressionML(train_data, test_data, label_train, label_test,
    #                                                        epochs=1, batch_size=10, lr=0.1, gd_type='noise')
    if logistic_instance.model_exists():
        print('local model already exists')
        print('-> loading models...')
        logistic_instance = logistic_instance.load_model()
        print(logistic_instance.accuracy_on_train())
        print(logistic_instance.accuracy_on_test())
    else:
        print('-> training model...')
        logistic_instance.train()
        print('-> saving model...')
        logistic_instance.save_model()
        print(logistic_instance.accuracy_on_train())
        print(logistic_instance.accuracy_on_test())
    # 绘制各个类别的ROC曲线
    PlotUtils.plot_roc_curve(logistic_instance)
    # 绘制训练过程记录
    PlotUtils.plot_train_record(logistic_instance)
