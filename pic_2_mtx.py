import cv2
import numpy as np
import sys

import read_picture


def pic_2_mtx(train_image_path, train_labels_path, test_image_path):
    sys.path.append("../read_picture/")

    train_image, train_label = read_picture.read_image_data(train_image_path, train_labels_path)

    test_image = []

    for i in range(0, 64):
        img = cv2.imread(test_image_path + str(i) + '.png', 0)
        matrix = np.asarray(img)
        test_image[i, :] = matrix

    test_image = np.reshape(test_image, (-1, 784))

    return train_image, train_label, test_image

# C:\\Users\\HP\\PycharmProjects\\my_code\\mnist_data\\train-images.idx3-ubyte
# C:\\Users\\HP\\PycharmProjects\\my_code\\mnist_data\\train-labels.idx1-ubyte
# C:\\Users\\HP\\PycharmProjects\\my_code\\auto_grader\\image\\
