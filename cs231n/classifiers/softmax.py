from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):

    # Initialize the loss and gradient to zero.
    loss = 0.0
    num_train= X.shape[0]
    num_classes=W.shape[1]
    dW = np.zeros_like(W)    
    for i in range(num_train):
        scores=np.dot(X[i],W)
        max_score=scores.max()
        scores -= max_score
        scores=np.exp(scores)
        probs = scores/np.sum(scores)
        loss += - np.log(probs[y[i]])
        # computing the gradients now
        # dl/dfyi = -1 + prob of correct class
        # dl/dfij = prob of that ij classifier
        probs[y[i]] += -1
        # update the params
        for j in range(num_classes):
            dW[:,j] +=probs[j]*X[i]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss /= num_train
    loss += reg*np.sum(W*W)
    dW /= num_train
    dW += reg*2*W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    scores = np.dot(X,W)
    num_train= X.shape[0]

    max_scores=scores.max(axis=1).reshape(-1,1) #max scores each axis
    scores -= max_scores
    unnorm_prob = np.exp(scores)
    # now convert un_norm probs to probs
    probs= unnorm_prob/np.sum(unnorm_prob,axis=1).reshape(-1,1)
    correct_class_probs=probs[np.arange(num_train),y]
    loss_exps= -np.log(correct_class_probs)
    loss = np.sum(loss_exps)
    #loss is simple -log(correct class prob)
    loss /= num_train
    loss += reg* np.sum(W*W)
    
    probs[np.arange(num_train),y] -= 1
    dW += np.dot(X.T,probs)
    dW/=num_train
    dW += reg*2*W

    return loss, dW
