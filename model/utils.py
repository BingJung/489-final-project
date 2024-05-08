from typing import Sequence
from math import ceil

def get_dims(input_dim: int, output_dim: int, layer_num: int) -> Sequence:
    '''
    automatically calculate dimension of each layer given
    input dimension, output dimension, and number of layers
    '''
    total_dim_difference = input_dim - output_dim

    unit_diff = ceil(total_dim_difference / sum([i+1 for i in range(layer_num)]))
    dim_differences = [(layer_num - layer_id) * unit_diff for layer_id in range(layer_num)]
    dims = [input_dim - sum(dim_differences[:i]) for i in range(layer_num)]
    
    # ensure input and output dims
    dims[0] = input_dim
    dims[-1] = output_dim
    return dims

def get_one_hot(id, dim):
    '''
    get one-hot vector
    input:
    id - the position to flip to 1
    dim - dimension of the required vector
    '''
    oh = [0 for _ in range(dim)]
    oh[id] = 1
    return oh