# * coding: utf8 *
import numpy as np
import struct

train_images_ubyte_path = '../MNIST/raw/train-images.idx3-ubyte'
train_labels_ubyte_path = '../MNIST/raw/train-labels.idx1-ubyte'
test_images_ubyte_path = '../MNIST/raw/t10k-images.idx3-ubyte'
test_labels_ubyte_path = '../MNIST/raw/t10k-labels.idx1-ubyte'


# 读取ubyte格式图像文件
def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    _, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images, num_images


# 读取ubyte格式标签文件
def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    _, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images, np.int8)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


# 加载训练和测试数据
def load_data_set():
    # 读取训练数据
    images_train, num_train = decode_idx3_ubyte(train_images_ubyte_path)
    # 读取测试数据
    images_test, num_test = decode_idx3_ubyte(test_images_ubyte_path)
    # 读取训练标签
    train_label = decode_idx1_ubyte(train_labels_ubyte_path)
    # 读取测试标签
    test_label = decode_idx1_ubyte(test_labels_ubyte_path)
    # min-max归一化
    images_train = images_train / 255
    images_test = images_test / 255
    return images_train, images_test, train_label, test_label
