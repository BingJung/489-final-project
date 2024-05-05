import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import get_dataset
from .BPNet import BPNet

from typing import Sequence
from math import ceil, log, copysign
import random
from tqdm import tqdm

class VAT:
    def __init__(self, dataset: str, layer_num: int=4) -> None:
        self.data_name = dataset
        self.data = get_dataset(dataset.upper() + "_filtered")
        self.errors = []
        
        dim_map = lambda x : ceil(x / log(x))
        self.entity_dim = self.data.entity_num
        self.relation_dim = self.data.relation_num
        self.latent_dim = ceil((dim_map(self.entity_dim) + dim_map(self.relation_dim)) / 2)
        print(f"start building VAT with {self.entity_dim} entities, {self.relation_dim} relations, and latent dimension {self.latent_dim}.")

        # build nets
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


    def train(self, epochs=100, batch_size=20, lr=0.005, momentum=0.01, save=False):
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
        self.errors = []
        with open('errors.tmp', 'w') as file: # clear temporary error record
            pass
        for round in range(epochs):
            # total_error = {"encoder": 0, "decoder": 0, "noise_gen": 0}
            batch_errors = []
            batch_error = 0
            for num, data in tqdm(enumerate(self.data.train_data()), total=self.data.train_num, desc=f"epochs {round+1}/{epochs}"):
                # conver input into one-hot vectors:
                data = [
                    get_one_hot(data[0], self.entity_dim), # h
                    get_one_hot(data[2], self.relation_dim), # r
                    get_one_hot(data[1], self.entity_dim), # t
                    ]

                # get outputs
                encoded, noise_levels, norm_noises, disturbed, translated, decoded = self.get_outputs(data)
                desired_noises = [
                    [copysign(
                        log(abs(
                                sum(
                                    [
                                        (disturbed[i][d] - translated[i][d]) / norm_noises[i][d] 
                                        for d in range(self.latent_dim)
                                        ]
                                )
                            )) / self.latent_dim,
                        sum(
                            [
                                (disturbed[i][d] - translated[i][d]) / norm_noises[i][d] 
                                for d in range(self.latent_dim)
                                ]
                        )
                    )]
                    for i in range(3)
                ]

                # # compute errors
                # encoded_error = [
                #     [disturbed[i][d] - translated[i][d] for d in range(self.latent_dim)]
                #     for i in range(3)
                # ]
                # decoded_error = [ # TODO: this can be improved
                #     [data[0][d] - decoded[0][d] for d in range(self.entity_dim)],
                #     [data[1][d] - decoded[1][d] for d in range(self.relation_dim)],
                #     [data[2][d] - decoded[2][d] for d in range(self.entity_dim)]
                # ]
                # noise_error = [ # TODO: this can be double checked
                #     sum([1 / (disturbed[i][d] - translated[i][d]) * norm_noises[i][d] for d in range(self.latent_dim)]) / self.latent_dim
                #     for i in range(3)
                # ]
                # total_error["encoder"] += sum([sum(i) for i in encoded_error])
                # total_error["decoder"] += sum([sum(i) for i in decoded_error])
                # total_error["noise_gen"] += sum(noise_error)

                # back propagate errors
                
                for i in range(3):
                    if i == 0 or i == 2: # head or tail
                        batch_error += self.entity_encoder.backward(disturbed[i], translated[i])
                        self.entity_encoder.update_weight_changes()
                        batch_error += self.entity_decoder.backward(data[i], decoded[i])
                        self.entity_decoder.update_weight_changes()
                        batch_error += self.entity_noise.backward(desired_noises[i], noise_levels[i])
                        self.entity_noise.update_weight_changes()
                    else: # relation
                        batch_error += self.relation_encoder.backward(disturbed[i], translated[i])
                        self.relation_encoder.update_weight_changes()
                        batch_error += self.relation_decoder.backward(data[i], decoded[i])
                        self.relation_decoder.update_weight_changes()
                        batch_error += self.relation_noise.backward(desired_noises[i], noise_levels[i])
                        self.relation_noise.update_weight_changes()

                # update weights if reaching batch size
                if (num+1) % batch_size == 0:
                    self.entity_encoder.update_weights()
                    self.entity_decoder.update_weights()
                    self.entity_noise.update_weights()
                    self.relation_encoder.update_weights()
                    self.relation_decoder.update_weights()
                    self.relation_noise.update_weights()
                    batch_error = batch_error/batch_size
                    batch_errors.append(batch_error)

                    # save batch error and reset
                    with open('errors.tmp', 'a') as file:
                        file.write(str(batch_error) + ', ')
                    batch_error = 0
                    

            ### at the end of each epochs:                    
            # update weights for the rest data
            self.entity_encoder.update_weights()
            self.entity_decoder.update_weights()
            self.entity_noise.update_weights()
            self.relation_encoder.update_weights()
            self.relation_decoder.update_weights()
            self.relation_noise.update_weights()

            # create new line in error.tmp
            with open('errors.tmp', 'a') as file:
                file.write('\n')

            # refresh process bar
            print(end='\r')

            # save parameters
            if save == True: # mind the order
                parameters = [
                    self.entity_encoder.get_weights(),
                    self.entity_decoder.get_weights(),
                    self.entity_noise.get_weights(),
                    self.relation_encoder.get_weights(),
                    self.relation_decoder.get_weights(),
                    self.relation_noise.get_weights()
                ]

                save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
                os.makedirs(save_dir, exist_ok=True)

                save_name = f'./checkpoints/{self.data_name}-lr{lr}-momen{momentum}-{epochs}.pkl'
                previous_name = f'./checkpoints/{self.data_name}-lr{lr}-momen{momentum}-{epochs-1}.pkl'
                if os.path.exists(os.path.join(save_dir, previous_name)):
                    os.remove(os.path.join(save_dir, previous_name))
                with open(os.path.join(save_dir, save_name), 'wb') as file:
                    pickle.dump(parameters, file)

    def load_parameters(self, name: str):
        save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        with open(os.path.join(save_dir, name), 'rb') as file:
            parameters = pickle.load(file)
        self.entity_encoder.load_weights(parameters),
        self.entity_decoder.load_weights(parameters),
        self.entity_noise.load_weights(parameters),
        self.relation_encoder.load_weights(parameters),
        self.relation_decoder.load_weights(parameters),
        self.relation_noise.load_weights(parameters)

    def test(self):
        '''
        test on relation and entity prediction and record average rankings of expected output
        '''
        rankings = [[], [], []]
        for data in self.data.test_data():
            # conver input into one-hot vectors:
            data_oh = [
                get_one_hot(data[0], self.entity_dim), # h
                get_one_hot(data[2], self.relation_dim), # r
                get_one_hot(data[1], self.entity_dim), # t
                ]
            
            # get outputs and record rankings
            encoded, noise_levels, norm_noises, disturbed, translated, decoded = self.get_outputs(data)
            for i, prediction in enumerate(decoded):
                confidence = prediction(data[i])
                sorted_pred = sorted(prediction)
                rankings[i].append(sorted_pred.index(confidence)+1)
                
    def get_outputs(self, data: Sequence):
        '''
        get full outputs of the model
        input: data - (h, r, t) in one-hot vectors
        '''
        encoded = [ # directly mapped representations
            self.entity_encoder.forward(data[0]), 
            self.relation_encoder.forward(data[1]), 
            self.entity_encoder.forward(data[2])
            ]
        noise_levels = [
            self.entity_noise.forward(encoded[0]), 
            self.relation_noise.forward(encoded[1]), 
            self.entity_noise.forward(encoded[2])
            ]
        norm_noises = [ # noise to be added to encoded representations
            [random.normalvariate() for _ in range(self.latent_dim)] # noises for each vector
            for _ in range(3)
            ]
        disturbed = [ # noisy representations
            [encoded[i][d] + noise_levels[i][0] * norm_noises[i][d] for d in range(self.latent_dim)]
            for i in range(3)
        ]
        translated = [ # predicted latent representations
            [disturbed[2][d] - disturbed[1][d] for d in range(self.latent_dim)], # h
            [disturbed[2][d] - disturbed[0][d] for d in range(self.latent_dim)], # r
            [disturbed[0][d] + disturbed[1][d] for d in range(self.latent_dim)]  # t
        ]
        decoded = [
            self.entity_decoder.forward(translated[0]),
            self.relation_decoder.forward(translated[1]),
            self.entity_decoder.forward(translated[2])
        ]

        return encoded, noise_levels, norm_noises, disturbed, translated, decoded 


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