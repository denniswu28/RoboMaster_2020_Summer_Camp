import numpy as np
from sklearn.decomposition import PCA

import kNN
import sys
sys.path.append("../read_picture/")
import read_picture
from keras.datasets import mnist

def identify_numbers(test_image):



    train_image, train_label = read_picture.read_image_data('./mnist_data/train-images.idx3-ubyte',
                                                            './mnist_data/train-labels.idx1-ubyte')

    x_test, y_test = read_picture.read_image_data('./mnist_data/t10k-images.idx3-ubyte',
                                                            './mnist_data/t10k-labels.idx1-ubyte')

    # (train_image, train_label), (x_test, y_test) = mnist.load_data()


    # train_image_arr = np.array(np.concatenate( [train_image , x_test], axis = 0 )) #array
    # train_label_arr = np.array(np.concatenate( [train_label , y_test], axis = 0 ))


    train_image_arr = np.array(np.concatenate( [train_image , x_test[0:5000]], axis = 0 )) #array
    train_label_arr = np.array(np.concatenate( [train_label , y_test[0:5000]], axis = 0 ))


    # train_image_arr = np.array(train_image)  # before submission, comment these two lines
    # train_label_arr = np.array(train_label)  # before submission, comment these two lines

    large_test_image_arr = np.array(x_test)
    large_test_label_arr = np.array(y_test)
    test_image_arr = np.array(test_image)


    # print(len(train_label_arr))
    # print(len(train_image_arr))


    train_image_arr = np.reshape(train_image_arr, (-1, 784))
    test_image_arr = np.reshape(test_image_arr, (-1, 784))
    large_test_image_arr = np.reshape(large_test_image_arr, (-1, 784))

    # print(train_image_arr[0])
    # train_image_arr[train_image_arr <= 0] = 0  # 转化为黑白图像
    train_image_arr[train_image_arr > 0] = 255  # 转化为黑白图像
    #
    #
    # # test_image_arr[test_image_arr <= 0] = 0  # 转化为黑白图像
    test_image_arr[test_image_arr > 0] = 255  # 转化为黑白图像
    #
    # # large_test_image_arr[large_test_image_arr <= 0] = 0  # 转化为黑白图像
    large_test_image_arr[large_test_image_arr > 0] = 255  # 转化为黑白图像

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

    # train_image_pca[train_image_pca > 0] = 255  # 转化为黑白图像
    # test_image_pca[test_image_pca > 0] = 255  # 转化为黑白图像
    # large_test_image_pca[large_test_image_pca > 0] = 255  # 转化为黑白图像


    f = kNN.KNNClassify()
    f.fit(train_image_arr, train_label_arr)
    # f.fit(train_image_pca, train_label_arr)

    # train_image_arr_e, train_label_arr_e = f.edit(train_image_arr, train_label_arr)
    # train_image_arr_e, train_label_arr_e = f.edit(train_image_pca, train_label_arr)

    # train_image_arr_e = np.array(train_image_arr_e)
    # train_label_arr_e = np.array(train_label_arr_e)

    # edited_X = open("X_train_edited.txt", mode='wb')
    # edited_y = open("y_train_edited.txt", mode='wb')
    # X_dtype = train_image_arr_e.dtype
    # y_dtype = train_label_arr_e.dtype
    # X_shape = train_image_arr_e.shape
    # y_shape = train_label_arr_e.shape
    # print(X_dtype)
    # print(y_dtype)
    # print(X_shape)
    # print(y_shape)
    # edited_X.write(train_image_arr_e.tostring())
    # edited_y.write(train_label_arr_e.tostring())
    # train_image_arr, train_label_arr = f.edit(train_image_pca, train_label_arr)

    # edited_X = open("X_train_edited.txt", mode='rb')
    # edited_y = open("y_train_edited.txt", mode='rb')
    # train_image_arr = np.fromstring(edited_X.read(), dtype=getattr(np, "uint8")).reshape(70000,784)
    # train_label_arr = np.fromstring(edited_y.read(), dtype=getattr(np, "int32")).reshape(70000,)


    y_pre_1 = f.predict_y(large_test_image_arr[5000:6000], train_image_arr, train_label_arr)
    #
    print(f.score(y_pre_1, large_test_label_arr[5000:6000]))

    # y_pre = f.predict_y(test_image_pca, train_image_pca, train_label_arr)
    # y_pre = f.predict_y(test_image_arr, train_image_arr_e, train_label_arr_e)
    y_pre = f.数字识别(test_image_arr, train_image_arr, train_label_arr)

    return y_pre