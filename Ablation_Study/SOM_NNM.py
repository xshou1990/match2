"""
Ablation Study 2: Self organizing map only (the binary classification part is disabled)

Author: Xiao Shou, some codes inspired and adapted from Florent Forest 's DESOM model

"""
# Utilities
import csv
#import argparse
from time import time
# Tensorflow/Keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from SOM import SOMLayer
#from metrics import quantization_error,topographic_error
import numpy as np
from sklearn.metrics import roc_auc_score



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


def som_loss(weights, distances):
    """
    SOM loss
    # Arguments
        weights: weights for the weighted sum, Tensor with shape `(n_samples, n_prototypes)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, Tensor with shape `(n_samples, n_prototypes)`
    # Return
        SOM reconstruction loss
    """
    return tf.reduce_mean(tf.reduce_sum(weights*distances, axis=1))


class MATCHSOM:
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

    def __init__(self, encoder_dims, map_size):
        self.encoder_dims = encoder_dims
        self.input_dim = self.encoder_dims[0]
        self.map_size = map_size
        self.n_prototypes = map_size[0]*map_size[1]
        self.BinCla = None
        self.encoder = None
        self.model = None

    
    def initialize(self, ae_act='relu', ae_init='glorot_uniform'):
        """
        Create MATCHSOM model
        # Arguments
            ae_act: activation for encoder intermediate layers
            ae_init: initialization of encoder layers
        """
        # Create binary classifier model
        
        self.BinCla, self.encoder = BinCla(self.encoder_dims, ae_act, ae_init)
        som_layer = SOMLayer(self.map_size, name='SOM')(self.encoder.input)
        # Create MATCHSOM model
        self.model = Model(inputs=self.BinCla.input,
                           outputs=[som_layer])
        
    @property
    def prototypes(self):
        """
        Returns SOM code vectors
        """
        return self.model.get_layer(name='SOM').get_weights()[0]

    def compile(self, optimizer):
        """
        Compile MATCHSOM model
        # Arguments
            gamma: coefficient of SOM loss
            optimizer: optimization algorithm
        """
        self.model.compile(loss={ 'SOM': som_loss},
                           optimizer=optimizer)
    
    def load_weights(self, weights_path):
        """
        Load pre-trained weights of MATCHSOM model
        # Arguments
            weight_path: path to weights file (.h5)
        """
        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_bincla_weights(self, bincla_weights_path):
        """
        Load pre-trained weights of AE
        # Arguments
            ae_weight_path: path to weights file (.h5)
        """
        self.BinCla.load_weights(bincla_weights_path)
        self.pretrained = True

    def init_som_weights(self, X):
        """
        Initialize with a sample w/o remplacement of encoded data points.
        # Arguments
            X: numpy array containing training set or batch
        """

        sample = X[np.random.choice(X.shape[0], size=self.n_prototypes, replace=False)]
        encoded_sample = self.encode(sample)
        self.model.get_layer(name='SOM').set_weights([encoded_sample])

    def encode(self, x):
        """
        Encoding function. Extract latent features from hidden layer
        # Arguments
            x: data point
        # Return
            encoded (latent) data point
        """
        return self.encoder.predict(x)

#     def predict(self, x):
#         """
#         Predict best-matching unit using the output of SOM layer
#         # Arguments
#             x: data point
#         # Return
#             index of the best-matching unit
#         """
#         _, d = self.model.predict(x, verbose=0)
#         return d.argmin(axis=1)

    def map_dist(self, y_pred):
        """
        Calculate pairwise Manhattan distances between cluster assignments and map prototypes (rectangular grid topology)
        
        # Arguments
            y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
        # Return
            pairwise distance matrix (map_dist[i,k] is the Manhattan distance on the map between assigned cell of data point i and cell k)
        """
        labels = np.arange(self.n_prototypes)
        tmp = np.expand_dims(y_pred, axis=1)
        d_row = np.abs(tmp-labels) // self.map_size[1]
        d_col = np.abs(tmp % self.map_size[1] - labels % self.map_size[1])
        return d_row + d_col

    @staticmethod
    def neighborhood_function(d, T, neighborhood='gaussian'):
        """
        SOM neighborhood function (gaussian neighborhood)
        # Arguments
            x: distance on the map
            T: temperature parameter
        # Return
            neighborhood weight
        """
        if neighborhood == 'gaussian':
            return np.exp(-(d ** 2) / (T ** 2))
        elif neighborhood == 'window':
            return (d <= T).astype(np.float32)
    
    
    
    def fit(self, X_train, y_train=None,
            X_val=None, y_val=None,
            iterations=10000,
            som_iterations=10000,
            eval_interval=10,
            save_epochs=5,
            batch_size=256,
            Tmax=10,
            Tmin=0.1,
            decay='exponential'):
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
           Tmax: initial temperature parameter
           Tmin: final temperature parameter
           decay: type of temperature decay ('exponential' or 'linear')
           save_dir: path to existing directory where weights and logs are saved
        """
    
        
        # Set and compute some initial values
        index = 0
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
                # Compute best matching units for batches
                d = self.model.predict(X_batch)
                d_pred = d.argmin(axis=1)
                if X_val is not None:
                    d_val = self.model.predict(X_val_batch)
                    d_val_pred = d_val.argmin(axis=1)

                # Update temperature parameter
                if ite < som_iterations:
                    if decay == 'exponential':
                        T = Tmax*(Tmin/Tmax)**(ite/(som_iterations-1))
                    elif decay == 'linear':
                        T = Tmax - (Tmax-Tmin)*(ite/(som_iterations-1))
                    else:    
                        T = decay
                # Compute topographic weights batches
                w_batch = self.neighborhood_function(self.map_dist(d_pred), T )

                if X_val is not None:
                    w_val_batch = self.neighborhood_function(self.map_dist(d_val_pred), T)

                # Train on batch
                loss = self.model.train_on_batch(x=X_batch, y= w_batch)


                if ite % eval_interval == 0:

                    if X_val is not None:
                        val_loss = self.model.test_on_batch(X_val_batch,  w_val_batch)
                        som_val_hist.append(val_loss)

                        # terminate if we have validation loss decrease or not increase more than 1e-3
                        if ite > 100 :
                            if   som_val_hist[-2] - som_val_hist[-1] < 1e-3:
                                break

