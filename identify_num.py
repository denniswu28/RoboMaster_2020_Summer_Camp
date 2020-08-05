import numpy as np
import sys
import read_picture
import cv2
from sklearn.decomposition import PCA
import kNN
from keras.datasets import mnist

def identify_numbers(test_image):

    (train_image, train_label), (x_test, y_test) = mnist.load_data()
    train_image_arr = np.array(np.concatenate( [train_image , x_test], axis = 0 )) #array
    train_label_arr = np.array(np.concatenate( [train_label , y_test], axis = 0 ))



    # train_image_arr = np.array(train_image)  # array
    # train_label_arr = np.array(train_label)
    large_test_image_arr = np.array(x_test)
    large_test_label_arr = np.array(y_test)
    test_image_arr = np.array(test_image)


    # print(len(train_label_arr))
    # print(len(train_image_arr))


    train_image_arr = np.reshape(train_image_arr, (-1, 784))
    test_image_arr = np.reshape(test_image_arr, (-1, 784))
    large_test_image_arr = np.reshape(large_test_image_arr, (-1, 784))

    # mean_num = np.mean(train_image_arr, axis=0)
    # # print(mean_num)
    # train_image_arr = train_image_arr - mean_num
    # large_test_image_arr = large_test_image_arr - mean_num
    # test_image_arr = test_image_arr - mean_num

    # pca = PCA(n_components=100)
    #
    # pca.fit(train_image_arr)  # fit PCA with training data instead of the whole dataset
    # train_image_pca = pca.transform(train_image_arr)
    # test_image_pca = pca.transform(test_image_arr)
    # large_test_image_pca = pca.transform(large_test_image_arr)


    # print(train_image_arr[0])
    # train_image_arr[train_image_arr <= 0] = 0  # 转化为黑白图像
    train_image_arr[train_image_arr > 0] = 1  # 转化为黑白图像


    # test_image_arr[test_image_arr <= 0] = 0  # 转化为黑白图像
    test_image_arr[test_image_arr > 0] = 1  # 转化为黑白图像

    # large_test_image_arr[large_test_image_arr <= 0] = 0  # 转化为黑白图像
    large_test_image_arr[large_test_image_arr > 0] = 1  # 转化为黑白图像


    f = kNN.KNNClassify()
    f.fit(train_image_arr, train_label_arr)

    # y_pre_1 = f.predict_y(large_test_image_pca[0:1000])
    #
    # print(f.score(y_pre_1, large_test_label_arr[0:1000]))

    y_pre = f.predict_y(test_image_arr)

    return y_pre