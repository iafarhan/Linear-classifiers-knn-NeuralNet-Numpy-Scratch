from builtins import range
from builtins import object
import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):

        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):

                dists[i,j]=np.sqrt(np.sum(np.square(X[i]-self.X_train[j])))

        return dists

    def compute_distances_one_loop(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):

            dists[i,:]=np.sqrt((np.sum(np.square(self.X_train-X[i]),axis=1)))            
        return dists

    def compute_distances_no_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        test_norm=np.sum(X**2,axis=1,keepdims=True) #e.g 500*1
        train_norm=np.sum(self.X_train**2,axis=1) #5000,
        train_dot_test=np.dot(X,self.X_train.T) #500*5000
        dists=-2*train_dot_test+test_norm+train_norm
        return dists

    def predict_labels(self, dists, k=1):

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            idx_nearest=np.argpartition(dists[i],k)[0:k]
            closest_y=self.y_train[idx_nearest]

            (labels,counts)=np.unique(closest_y,return_counts=True)
            idx=np.argmax(counts)
            y_pred[i]=labels[idx]

        return y_pred
