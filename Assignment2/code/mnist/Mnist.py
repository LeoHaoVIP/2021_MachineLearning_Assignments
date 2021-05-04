# * coding: utf8 *
import numpy as np
import mnist.Utils as Utils

train_images_ubyte_path = 'mnist/mnist-ubyte/train-images.idx3-ubyte'
train_labels_ubyte_path = 'mnist/mnist-ubyte/train-labels.idx1-ubyte'
test_images_ubyte_path = 'mnist/mnist-ubyte/t10k-images.idx3-ubyte'
test_labels_ubyte_path = 'mnist/mnist-ubyte/t10k-labels.idx1-ubyte'


class MachineLearning:
    def __init__(self, train_set, test_set, train_label, test_label):
        # =========@BigData=========
        # 导入训练和测试数据
        self.train_set = train_data
        self.test_set = test_data
        self.train_label = train_label
        self.test_label = test_label
        # =========@Model=========
        # 模型参数
        self.weight = np.random.random((3, 5))

    # =========@Loss=========
    # 损失函数，计算预测结果与实际标签的距离
    def loss_function(self):
        loss = 0
        return loss

    # =========@Algorithm=========
    def update_model_params(self):
        # 更新模型参数
        self.weight += np.random.random((3, 5))
        pass

    # =========@Application=========
    # 模型应用，输出预测结果
    def model_test(self, sample):
        random_label = np.random.randint(0, 10)
        return random_label

    # =========@Evaluation=========
    # 模型评估 （针对训练集和测试集各自的准确率、召回率等指标）
    def model_evaluate(self):
        pass


# 加载训练和测试数据
def load_data():
    # 读取训练数据
    images_train, num_train = Utils.decode_idx3_ubyte(train_images_ubyte_path)
    # 读取测试数据
    images_test, num_test = Utils.decode_idx3_ubyte(test_images_ubyte_path)
    # 读取训练标签
    train_label = Utils.decode_idx1_ubyte(train_labels_ubyte_path)
    # 读取测试标签
    test_label = Utils.decode_idx1_ubyte(test_labels_ubyte_path)
    return images_train, images_test, train_label, test_label


if __name__ == '__main__':
    # 加载训练和测试数据
    train_data, test_data, label_train, label_test = load_data()
    # 创建MachineLearning对象
    ml = MachineLearning(train_data, test_data, label_train, label_test)
    # 训练模型（更新模型参数）
    print('--->> start model training...')
    ml.update_model_params()
    print('--->> model training finished, final w= \r\n', ml.weight)
    # 模型评估
    ml.model_evaluate()
    # 随机读取训练集图片，输出识别结果
    test_index = np.random.randint(0, len(label_test))
    label_predict = ml.model_test(sample=test_data[test_index])
    print('--->> {0}; ideal label: {1}; predict label: {2}'.format(label_test[test_index] == label_predict,
                                                                   label_test[test_index], label_predict))