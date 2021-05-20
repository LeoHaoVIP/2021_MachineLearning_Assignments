# * coding: utf8 *
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

CLASS_NUM = 10


def random_select(train_data, test_data, label_train, label_test):
    train_index = np.loadtxt('./dataset/train_index.txt', dtype=np.int32)
    test_index = np.loadtxt('./dataset/test_index.txt', dtype=np.int32)
    train_data = train_data[train_index]
    label_train = label_train[train_index]
    test_data = test_data[test_index]
    label_test = label_test[test_index]
    return train_data, test_data, label_train, label_test


# 绘制roc曲线
def plot_roc_curve(logistic_instance):
    # 对测试样本所属类别进行one-hot编码
    test_labels = label_binarize(logistic_instance.test_label, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 获取每个测试样本所属类别得分
    test_scores = logistic_instance.score_on_test()
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure()
    for i in range(CLASS_NUM):
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        lw = 2
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class %d (area = %0.3f)' % (i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('ROC\n\n{0}'.format(logistic_instance.save_path.split('/')[2][:-4]))
    plt.show()


# 绘制训练过程记录
def plot_train_record(logistic_instance):
    # 获取训练记录
    train_record = np.array(logistic_instance.train_record)
    fig, ax1 = plt.subplots()
    # 创建第二个坐标轴
    iter_index = np.arange(1, len(train_record) + 1)
    acc_train = np.array(train_record)[:, 2]
    acc_test = train_record[:, 3]
    train_loss = train_record[:, 4]
    # 绘图
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Train Loss')
    ax1.plot(iter_index, train_loss, '--', c='orange', label='train-loss', linewidth=2)
    plt.legend(loc=2)
    ax2 = ax1.twinx()
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.plot(iter_index, acc_train, '--', c='blue', label='acc_train', linewidth=2)
    ax2.plot(iter_index, acc_test, '--', c='green', label='acc_test', linewidth=2)
    plt.legend(loc=1)
    plt.title('Training Process\n\n{0}'.format(logistic_instance.save_path.split('/')[2][:-4]))
    plt.show()
