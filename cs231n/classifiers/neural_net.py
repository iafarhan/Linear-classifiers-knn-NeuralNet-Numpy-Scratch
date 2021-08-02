from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

def affine_forward(X,W,b):

  return np.dot(X,W)+b

def affine_backward(up_grad,X,W,b ):
  dW = np.dot(X.T,up_grad) 
  dX = np.dot(up_grad,W.T)
  db = np.ones(b.shape)
  db = np.sum(up_grad,axis=0)
  return dX, dW, db

def relu_forward(fc_scores):
  return np.maximum(0,fc_scores)

def relu_backward(up_grad,fc_scores):
  dfc_scores = np.where(fc_scores>0,1,0)
  return dfc_scores*up_grad

def softmax_with_cross_entropy(scores,y):
  scores -= np.max(scores,axis=1,keepdims=True)
  unorm_probs = np.exp(scores)
  probs = unorm_probs/np.sum(unorm_probs,axis=1,keepdims=True)
  N=probs.shape[0]
  loss = -np.log(probs[np.arange(N),y])
  loss = np.sum(loss)
  loss /= N

  #dscores
  dScores=np.zeros(scores.shape)
  probs[np.arange(N),y] -= 1
  dScores=probs
  dScores /= N
  return loss, dScores
  



class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        scores = None

        fc1_scores=affine_forward(X,W1,b1)
        fc1_relu=relu_forward(fc1_scores)
        scores=affine_forward(fc1_relu,W2,b2)

        if y is None:
            return scores

        loss = None

        loss,dScores = softmax_with_cross_entropy(scores,y)
        #regularization
        
        loss += reg*np.sum(W1*W1)
        loss += reg* np.sum(W2*W2)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        grads = {}

        dfc1_relu,dW2,db2 = affine_backward(dScores,fc1_relu,W2,b2)
        grads['W2']=dW2+ reg*2*W2
        grads['b2']=db2

        dfc1_scores = relu_backward(dfc1_relu,fc1_scores)
        dX,dW1,db1 = affine_backward(dfc1_scores,X,W1,b1)
        grads['W1']=dW1+ reg*2*W1
        grads['b1']=db1
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
      
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None
            batch_idxs=np.random.choice(num_train,batch_size)
            X_batch=X[batch_idxs]
            y_batch=y[batch_idxs] 

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            for param in self.params:
                self.params[param] -= learning_rate*grads[param]
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):

        y_pred = None

        fwd_pass_scores=self.loss(X)
        y_pred = np.argmax(fwd_pass_scores,axis=1)

        return y_pred
