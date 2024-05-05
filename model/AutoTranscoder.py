import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import get_dataset
from .BPNet import BPNet

from typing import Sequence
from math import ceil, log
import random
from tqdm import tqdm

class VAT:
    def __init__(self, dataset: str, layer_num: int=4, joint: bool=False) -> None:
        self.data = get_dataset(dataset + "_filtered")
        
        dim_map = lambda x : ceil(x / log(x))
        self.entity_dim = self.data.entity_num
        self.relation_dim = self.data.relation_num
        self.latent_dim = ceil((dim_map(self.entity_dim) + dim_map(self.relation_dim)) / 2)
        print(f"start building VAT with {self.entity_dim} entities, {self.relation_dim} relations, and latent dimension {self.latent_dim}.")

        # make nets; ignore joint first
        print("setting entity encoder...")
        self.entity_encoder = BPNet(get_dims(self.entity_dim, self.latent_dim, layer_num))
        print("setting entity decoder...")
        self.entity_decoder = BPNet(get_dims(self.latent_dim, self.entity_dim, layer_num))
        print("setting relation encoder...")
        self.relation_encoder = BPNet(get_dims(self.relation_dim, self.latent_dim, layer_num))
        print("setting relation decoder...")
        self.relation_decoder = BPNet(get_dims(self.latent_dim, self.relation_dim, layer_num))
        print("setting disturbers...")
        self.entity_noise = BPNet(get_dims(self.latent_dim, 1, 4))
        self.relation_noise = BPNet(get_dims(self.latent_dim, 1, 4))
        
        print("done initialization.")


    def train(self, epochs=100, batch_size=100, lr=0.01, momentum=0):
        '''
        training for mixed prediction
        '''
        print("start training.")
        # set learning rates and momentums
        self.entity_encoder.set_learning_rate(lr)
        self.entity_decoder.set_learning_rate(lr)
        self.relation_encoder.set_learning_rate(lr)
        self.relation_decoder.set_learning_rate(lr)
        self.entity_encoder.set_momentum(momentum)
        self.entity_decoder.set_momentum(momentum)
        self.relation_encoder.set_momentum(momentum)
        self.relation_decoder.set_momentum(momentum)

        # start training
        for round in range(epochs):
            total_loss = 0
            for num, data in tqdm(enumerate(self.data.train_data()), total=self.data.train_num):
                # ignore some samples
                if data[0] > 2000 or data[1] > 2000:
                    continue
                # conver input into one-hot vectors:
                data = [
                    get_one_hot(data[0], self.entity_dim),
                    get_one_hot(data[2], self.relation_dim),
                    get_one_hot(data[1], self.entity_dim),
                    ]

                # get outputs
                encoded = [
                    self.entity_encoder.forward(data[0]), 
                    self.relation_encoder.forward(data[1]), 
                    self.entity_encoder.forward(data[2])
                    ]
                noise_levels = [
                    self.entity_noise.forward(encoded[0]), 
                    self.relation_noise.forward(encoded[1]), 
                    self.entity_noise.forward(encoded[2])
                    ]
                noises = [
                    [random.random() for _ in range(self.latent_dim)] # noises for each vector
                    for _ in range(3)
                    ]
                disturbed = [
                    [encoded[i][d] + noise_levels[i][0] * noises[i][d] for d in range(self.latent_dim)]
                    for i in range(3)
                ]
                predicted = [
                    self.entity_decoder.forward([disturbed[2][d] - disturbed[1][d] for d in range(self.latent_dim)]),
                    self.relation_decoder.forward([disturbed[2][d] - disturbed[0][d] for d in range(self.latent_dim)]),
                    self.entity_decoder.forward([disturbed[0][d] + disturbed[1][d] for d in range(self.latent_dim)])
                ]

                # TODO: compute errors
                
                total_loss += 0

                # TODO: back propagate errors
                for i in range(3):
                    pass

                # TODO: update weights if reaching batch size
                if (num+1) % batch_size:
                    pass

            # update weights for the rest data
            if (num+1) % batch_size:
                pass



    def test(self):
        # TODO: test on relation and entity prediction and record average rankings of expected output
        pass

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