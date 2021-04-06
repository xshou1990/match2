"""
Implementation of Match2: Hybrid Self-Organizing Map and Neural Network Strategies for Treatment Effect Estimation

Author: Xiao Shou, some codes inspired and adapted from Florent Forest 's DESOM model

"""

import csv
from time import time
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from SOM import SOMLayer
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
        self.pretrained = False
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
        som_layer = SOMLayer(self.map_size, name='SOM')(self.encoder.output)
        # Create MATCHSOM model
        self.model = Model(inputs=self.BinCla.input,
                           outputs=[self.BinCla.output, som_layer])
        
    @property
    def prototypes(self):
        """
        Returns SOM code vectors
        """
        return self.model.get_layer(name='SOM').get_weights()[0]

    def compile(self, gamma, optimizer):
        """
        Compile MATCHSOM model
        # Arguments
            gamma: coefficient of SOM loss
            optimizer: optimization algorithm
        """
        self.model.compile(loss={'softmax': 'binary_crossentropy', 'SOM': som_loss},
                           loss_weights=[1, gamma],
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

    def predict(self, x):
        """
        Predict best-matching unit using the output of SOM layer
        # Arguments
            x: data point
        # Return
            index of the best-matching unit
        """
        _, d = self.model.predict(x, verbose=0)
        return d.argmin(axis=1)

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
    
    def pretrain(self, X, y,
                 Xval,yval,
                 optimizer='adam',
                 epochs=1000,
                 batch_size=64,
                 save_dir='../../results/tmp'):
        """
        Pre-train the binary classifier using only binary cross entropy loss
        Saves weights in h5 format.
        # Arguments
            X: training set
            y: label
            optimizer: optimization algorithm
            epochs: number of pre-training epochs
            batch_size: training batch size
            save_dir: path to existing directory where weights will be saved
        """
        print('Pretraining...')
        self.BinCla.compile(optimizer=optimizer, loss='binary_crossentropy')

        # Begin pretraining
        t0 = time()
        self.BinCla.fit(X, y, batch_size=batch_size, epochs=epochs, validation_data = (Xval,yval), callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)])
        print('Pretraining time: ', time() - t0)
        testloss = self.BinCla.evaluate(Xval,yval)
        self.BinCla.save_weights('{}/bincla_weights-epoch{}.h5'.format(save_dir, epochs))
        print('Pretrained weights are saved to {}/bincla_weights-epoch{}.h5'.format(save_dir, epochs))
        self.pretrained = True
        return testloss
    
    
    def fit(self, X_train, y_train=None,
            X_val=None, y_val=None,
            iterations=10000,
            som_iterations=10000,
            eval_interval=10,
            save_epochs=5,
            batch_size=256,
            Tmax=10,
            Tmin=0.1,
            decay='exponential',
           save_dir='results/tmp'):
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
        
        if not self.pretrained:
            print('Classifier was not pre-trained!')

        
        # Logging file
        logfile = open(save_dir + '/matchsom_log.csv', 'w')
        fieldnames = ['iter', 'T', 'L', 'Lc', 'Lsom']
        if X_val is not None:
            fieldnames += ['L_val', 'Lc_val', 'Lsom_val']

        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()
        
        
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
                # Compute best matching units for batches
                _, d = self.model.predict(X_batch)
                d_pred = d.argmin(axis=1)
                if X_val is not None:
                    _, d_val = self.model.predict(X_val_batch)
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
                loss = self.model.train_on_batch(x=X_batch, y=[y_batch, w_batch])

                if ite % eval_interval == 0:

                    # Initialize log dictionary
                    logdict = dict(iter=ite, T=T)


                    logdict['L'] = loss[0]
                    logdict['Lc'] = loss[1]
                    logdict['Lsom'] = loss[2]


                    if X_val is not None:
                        val_loss = self.model.test_on_batch(X_val_batch, [y_val_batch, w_val_batch])
                        logdict['L_val'] = val_loss[0]
                        logdict['Lc_val'] = val_loss[1]
                        logdict['Lsom_val'] = val_loss[2]

                        bce_val_hist.append(val_loss[1])
                        som_val_hist.append(val_loss[2])

                        # terminate if we have validation loss decrease or not increase more than 1e-3
                        if ite > 100 :
                            #print(bce_val_hist[-2], bce_val_hist[-1], som_val_hist[-2] , som_val_hist[-1])
                            if bce_val_hist[-2] - bce_val_hist[-1] < 1e-3 and  som_val_hist[-2] - som_val_hist[-1] < 1e-3:
                                break

                    logwriter.writerow(logdict)
