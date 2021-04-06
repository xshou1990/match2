"""
Ablation Study 1: feed forward binary classification model.
Author: Xiao Shou

"""
# Utilities
#import csv
#import argparse
from time import time
# Tensorflow/Keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np



def BinCla(encoder_dims, act='relu', init='glorot_uniform'):
    """
    Fully connected feed forward binary classification model.
    # Arguments
        encoder_dims: list of number of units in each layer of encoder. encoder_dims[0] is input dim, encoder_dims[-1] is units in hidden layer (latent dim).
        act: activation of feed forward neural network intermediate layers, not applied to Input, Hidden and Output layers
        init: initialization of layers
    # Return
        (BinCla, encoder): Binary Classifier and encoder models
    """
    
    n_stacks = len(encoder_dims) - 1

    # Input
    x = Input(shape=(encoder_dims[0],), name='input')
    # Internal layers in encoder
    encoded = x
    for i in range(n_stacks-1):
        encoded = Dense(encoder_dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(encoded)
    # Hidden layer (latent space)
    encoded = Dense(encoder_dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(encoded) # hidden layer, latent representation is extracted from here
    
    encoder = Model(inputs=x, outputs=encoded, name='encoder')
    
    encoded_prob = Dense(1,activation='sigmoid',name = 'softmax') (encoded)
    # classifier model
    BinCla = Model(inputs=x, outputs=encoded_prob , name='BinCla')

    return BinCla, encoder



class MATCHLG:
    """
    Deep Embedded Self-Organizing Map for Covariate Matching (MATCHSOM) model
    # Example
        ```
        matchsom = MATCHSOM(encoder_dims=[784, 500, 500, 2000, 10], map_size=(10,10))
        ```
    # Arguments
        encoder_dims: list of numbers of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer (latent dim)
        map_size: tuple representing the size of the rectangular map. Number of prototypes is map_size[0]*map_size[1]
    """

    def __init__(self, encoder_dims ):
        self.encoder_dims = encoder_dims
        self.input_dim = self.encoder_dims[0]
        self.BinCla = None
        self.encoder = None
        self.model = None

    
    def initialize(self, ae_act='relu', ae_init='glorot_uniform'):
        """
        Create MATCHLG model
        # Arguments
            ae_act: activation for encoder intermediate layers
            ae_init: initialization of encoder layers
        """
        # Create binary classifier model
        self.BinCla, self.encoder = BinCla(self.encoder_dims, ae_act, ae_init)
        # Create MATCHSOM model
        self.model = Model(inputs=self.BinCla.input,
                           outputs=[self.BinCla.output])
        

    def compile(self, optimizer):
        """
        Compile MATCHSOM model
        # Arguments
            gamma: coefficient of SOM loss
            optimizer: optimization algorithm
        """
        self.model.compile(loss={'softmax': 'binary_crossentropy'},
                           optimizer=optimizer)

    def encode(self, x):
        """
        Encoding function. Extract latent features from hidden layer
        # Arguments
            x: data point
        # Return
            encoded (latent) data point
        """
        return self.encoder.predict(x)
    
    def fit(self, X_train, y_train=None,
            X_val=None, y_val=None,
            iterations=3000,
            eval_interval=100,
            batch_size=64):
        """
        Training procedure
        # Arguments
           X_train: training set
           y_train: (optional) training labels
           X_val: (optional) validation set
           y_val: (optional) validation labels
           iterations: number of training iterations
           som_iterations: number of iterations where SOM neighborhood is decreased
           eval_interval: evaluate metrics on training/validation batch every eval_interval iterations
           save_epochs: save model weights every save_epochs epochs
           batch_size: training batch size
        """

        # Set and compute some initial values
        index = 0
        bce_val_hist = []
        som_val_hist = []
        
        if X_val is not None:
            index_val = 0

        for ite in range(iterations):
            # Get training and validation batches
            if (index + 1) * batch_size > X_train.shape[0]:
                X_batch = X_train[index * batch_size::]
                if y_train is not None:
                    y_batch = y_train[index * batch_size::]
                index = 0
            else:
                X_batch = X_train[index * batch_size:(index + 1) * batch_size]
                if y_train is not None:
                    y_batch = y_train[index * batch_size:(index + 1) * batch_size]
                index += 1
            if X_val is not None:
                if (index_val + 1) * batch_size > X_val.shape[0]:
                    X_val_batch = X_val[index_val * batch_size::]
                    if y_val is not None:
                        y_val_batch = y_val[index_val * batch_size::]
                    index_val = 0
                else:
                    X_val_batch = X_val[index_val * batch_size:(index_val + 1) * batch_size]
                    if y_val is not None:
                        y_val_batch = y_val[index_val * batch_size:(index_val + 1) * batch_size]
                    index_val += 1
                    
            if (X_batch.shape[0] > 0) and (X_val_batch.shape[0] > 0) : 

                # Train on batch
                loss = self.model.train_on_batch(x=X_batch, y=y_batch)

                if ite % eval_interval == 0:

                    if X_val is not None:
                        val_loss = self.model.test_on_batch(X_val_batch, y_val_batch)
                        bce_val_hist.append(val_loss)

                        # terminate if we have validation loss decrease or not increase more than 1e-3
                        if ite > 100 :
                            #print(bce_val_hist[-2], bce_val_hist[-1], som_val_hist[-2] , som_val_hist[-1])
                            if bce_val_hist[-2] - bce_val_hist[-1] < 1e-3 :
                                break

