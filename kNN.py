import operator

import numpy as np


class KNNClassify():

    def __init__(self, k=3, p=2):

        self.k = k
        self.p = p
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):

        self._X_train = X_train
        self._y_train = y_train
        return self

    def gaussian(self, dist, sigma=10.0):
        """ Input a distance and return it`s weight"""
        weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
        return weight

    def predict_y(self, X_test, X_train, y_train):
        # X_test[X_test > 0] = 1  # 转化为二进制
        m = X_train.shape[0]
        y_pre = []
        for intX in X_test:
            # print(np.tile(intX, (m, 1)))
            # print(m)
            minus_mat = np.tile(intX, (m, 1)) - X_train
            sq_minus_mat = minus_mat ** self.p
            sq_distance = sq_minus_mat.sum(axis=1)
            diff_sq_distance = sq_distance ** float(1 / self.p)

            sorted_distance_index = diff_sq_distance.argsort()
            class_count = {}
            dist = []
            for i in range(self.k):
                weight = self.gaussian(sorted_distance_index[i])
                dist = y_train[sorted_distance_index[i]]
                class_count[dist] = class_count.get(dist, 0) + weight

            sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
            y_pre.append((sorted_class_count[0][0]))

        return np.array(y_pre)

    def score(self, y_pre, y_test):

        j = 0
        for i in range(len(y_pre)):
            if y_pre[i] == y_test[i]:
                j += 1
        return ('accuracy: {:.10%}'.format(j / len(y_test)))

    def edit(self, X_train_e, y_train_e):

        X_edited_train = X_train_e
        y_edited_train = y_train_e
        deleted = []

        for i in range(1):
            left = i * 1000
            right = left + 1000
            X_edit_train = np.concatenate((X_train_e[0:left] , X_train_e[right:len(X_train_e)]), axis=0)
            y_edit_train = np.concatenate((y_train_e[0:left] , y_train_e[right:len(y_train_e)]), axis=0)
            X_edit_test = X_train_e[left:right]
            y_edit_test = y_train_e[left:right]

            print(i)

            y_edit_pre = self.predict_y(X_edit_test, X_edit_train, y_edit_train)

            for j in range(len(y_edit_pre) - 1, 0, -1):
                if y_edit_pre[j] != y_edit_test[j]:
                    # X_edit_test.pop(j)
                    deleted.append(left + j)



        deleted.sort(reverse=True)
        print("We have " + str(len(deleted)) + " graphs deleted.")
        for i in range(len(deleted)):
            np.delete(X_edited_train, deleted[i], axis=0)
            np.delete(y_edited_train, deleted[i], axis=0)

        return X_edited_train, y_edited_train
