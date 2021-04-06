import numpy as np


def neuron_idx(net_dim):
    '''
    unravel the neuron coordinates for a given network topology 
    i.e. net_dim = [3,3], it returns [[0,0],[0,1],[0,2],[1,0],...]] 
    '''
    neuron_idx = []
    tot_neurons = net_dim[0]*net_dim[1]
    for i in range(tot_neurons):
        neuron_idx.append (np.concatenate(np.where (np.eye(tot_neurons)[i].reshape(net_dim) != 0) ))
    return np.array(neuron_idx)

def idx_convert(bmu,neuron_map):
    '''
    find the index/coordinate on the map for a given number
    bmu : best match unit
    neuron_map: indices of SOM 
    '''
    neuron_coord = []
    for i in bmu:
        neuron_coord.append(neuron_map[i])
    return neuron_coord