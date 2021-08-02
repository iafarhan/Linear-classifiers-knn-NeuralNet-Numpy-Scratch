from builtins import range
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape) # initialize the gradient as zero

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                dW[:,j] += X[i]*1
                dW[:,y[i]] += X[i]*-1
              
                loss += margin
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW+=reg*2*W
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):

    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    num_train = X.shape[0]
    scores = np.dot(X,W)
    correct_class_scores = scores[np.arange(num_train),y].reshape(num_train,1)
    margins = scores-correct_class_scores+1
    margins[np.arange(num_train),y] = 0
    margins[margins < 0] = 0
    loss_example_wise = np.sum(margins,axis=1)
    loss = np.sum(loss_example_wise)
    loss /= num_train
    loss += reg * np.sum(W*W)
    margins[margins>0]=1
    # as in gradient we saw that dervivate due to one margin voilation for
    # a single class was -1 * X[i]
    #so we are counting the margins 
    num_wrong_classifers=-1*margins.sum(1)
    margins[np.arange(num_train),y]=num_wrong_classifers
    dW+=np.dot(margins.T,X).T
    dW /= num_train
    dW += reg * 2 * W


    return loss, dW