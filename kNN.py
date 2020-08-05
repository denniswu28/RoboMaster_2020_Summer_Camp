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

    def predict_y(self, X_test):
        # X_test[X_test > 0] = 1  # 转化为二进制
        m = self._X_train.shape[0]
        y_pre = []
        for intX in X_test:
            # print(np.tile(intX, (m, 1)))
            # print(m)
            minus_mat = np.tile(intX, (m, 1)) - self._X_train
            sq_minus_mat = minus_mat ** self.p
            sq_distance = sq_minus_mat.sum(axis=1)
            diff_sq_distance = sq_distance ** float(1 / self.p)

            sorted_distance_index = diff_sq_distance.argsort()
            class_count = {}
            dist = []
            for i in range(self.k):
                dist = self._y_train[sorted_distance_index[i]]
                class_count[dist] = class_count.get(dist, 0) + 1

            sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
            y_pre.append((sorted_class_count[0][0]))

        return np.array(y_pre)

    def score(self, y_pre, y_test):

        j = 0
        for i in range(len(y_pre)):
            if y_pre[i] == y_test[i]:
                j += 1
        return ('accuracy: {:.10%}'.format(j / len(y_test)))

