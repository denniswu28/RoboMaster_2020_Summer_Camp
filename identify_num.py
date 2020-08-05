import numpy as np
import sys
import read_picture
import kNN
from keras.datasets import mnist

def identify_numbers(test_image):

    (train_image, train_label), (x_test, y_test) = mnist.load_data()
    train_image_arr = np.array(np.concatenate( [train_image , x_test], axis = 0 )) #array
    train_label_arr = np.array(np.concatenate( [train_label , y_test], axis = 0 ))

    # print(len(train_label_arr))
    # print(len(train_image_arr))


    train_image_arr = np.reshape(train_image_arr, (-1,784))

    # print(train_image_arr[0])
    train_image_arr[train_image_arr > 0] = 1 #转化为二进制

    f = kNN.KNNClassify()
    f.fit(train_image_arr, train_label_arr)

    y_pre = f.predict_y(test_image)

    return y_pre