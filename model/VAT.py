import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import get_dataset
from .BPNet import BPNet
from .utils import get_dims, get_one_hot

from typing import Sequence
from math import ceil, log
import random
from tqdm import tqdm

class VAT:
    def __init__(self, dataset: str, layer_num: int=4) -> None:
        self.data_name = dataset
        self.data = get_dataset(dataset.upper() + "_filtered")
        self.errors = []
        self.epochs = 0
        self.best = (0, 10000) # (epoch, loss)
        
        # calculate dimensions
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
        self.entity_noise = BPNet(get_dims(self.latent_dim, self.latent_dim, 4))
        self.relation_noise = BPNet(get_dims(self.latent_dim, self.latent_dim, 4))
        
        print("done initialization.")


    def train(self, epochs=100, batch_size=25, lr=0.01, momentum=0, save=False):
        '''
        training for mixed prediction
        '''
        assert epochs > self.epochs, f"this model is already at epoch {self.epochs}"
        print("start training.")

        # set learning rate and momentum
        self.set_hyper(lr, momentum)

        if save:
            # set and make saving directories
            save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', f'VAT-{self.data_name}-bs_{batch_size}-lr_{lr}-mt_{momentum}')
            os.makedirs(save_dir, exist_ok=True)

            # save error history if a checkpoint is loaded
            if self.epochs > 0:
                with open(os.path.join(save_dir, 'errors.tmp'), 'w') as file:
                    for error in self.errors:
                        file.write(str(error) + '\n')
            # make new error record otherwise
            else: 
                with open(os.path.join(save_dir, 'errors.tmp'), 'w') as file:
                    pass

        # train
        for round in range(epochs - self.epochs):
            epoch = round + self.epochs + 1 # current actual epoch
            batch_errors = [] # total error records for each batch

            # loop batches
            progress_bar = tqdm(total=self.data.train_num)
            for num, data in enumerate(self.data.train_data()):
                progress_bar.set_description(f"epochs {epoch}/{epochs}; batch")
                batch_error = 0 # summed error for each batch

                # conver input into one-hot vectors:
                data = [
                    get_one_hot(data[0], self.entity_dim), # h
                    get_one_hot(data[2], self.relation_dim), # r
                    get_one_hot(data[1], self.entity_dim), # t
                    ]
                
                # get outputs of model
                encoded, noise_levels, norm_noises, disturbed, translated, decoded = self.get_outputs(data)
                desired_noise_levels = [
                    [(translated[i][d] - encoded[i][d]) / norm_noises[i][d] 
                    for d in range(self.latent_dim)] for i in range(3)
                ]

                # back propagate errors
                for i in range(3):
                    if i == 0 or i == 2: # head or tail
                        # encoder
                        batch_error += abs(self.entity_encoder.backward(translated[i], encoded[i]))
                        self.entity_encoder.update_weight_changes()

                        # decoder
                        batch_error += abs(self.entity_decoder.backward(data[i], decoded[i]))
                        self.entity_decoder.update_weight_changes()

                        # noise generator
                        batch_error += abs(self.entity_noise.backward(desired_noise_levels[i], noise_levels[i]))
                        self.entity_noise.update_weight_changes()

                    else: # relation
                        # encoder
                        batch_error += abs(self.relation_encoder.backward(translated[i], encoded[i]))
                        self.relation_encoder.update_weight_changes()

                        # decoder
                        batch_error += abs(self.relation_decoder.backward(data[i], decoded[i]))
                        self.relation_decoder.update_weight_changes()

                        # noise generator
                        batch_error += abs(self.relation_noise.backward(desired_noise_levels[i], noise_levels[i]))
                        self.relation_noise.update_weight_changes()
                    
                # update progress bar
                progress_bar.update(1)

                # at the end of each batch:
                if (num+1) % batch_size == 0 or num == self.data.train_num - 1:
                    # update weights
                    self.entity_encoder.update_weights()
                    self.entity_decoder.update_weights()
                    self.entity_noise.update_weights()
                    self.relation_encoder.update_weights()
                    self.relation_decoder.update_weights()
                    self.relation_noise.update_weights()

                    # save batch error and report
                    batch_errors.append(batch_error)
                    progress_bar.set_postfix_str(f"avg batch error: {batch_error/batch_size}")

            # at the end of each epoch:                    
            # update epoch error (total)
            epoch_error = sum(batch_errors)
            self.errors.append(epoch_error)
            if abs(epoch_error) < abs(self.best[-1]):
                self.best = (epoch, epoch_error)

            # save parameters
            if save:
                parameters = [
                    self.entity_encoder.get_weights(),
                    self.entity_decoder.get_weights(),
                    self.entity_noise.get_weights(),
                    self.relation_encoder.get_weights(),
                    self.relation_decoder.get_weights(),
                    self.relation_noise.get_weights()
                ] # order matters

                # file names
                save_name = f'{epoch}.pkl'
                previous_name = f'{epoch-1}.pkl'

                # keep checkpoints every 10 epochs
                if os.path.exists(os.path.join(save_dir, previous_name)) and (epoch-1) % 10 != 0:
                    os.remove(os.path.join(save_dir, previous_name))
                with open(os.path.join(save_dir, save_name), 'wb') as file:
                    pickle.dump(parameters, file)

                # write epoch error
                with open(os.path.join(save_dir, 'errors.tmp'), 'a') as file:
                    file.write(f"{str(epoch_error)}\n")

        # end of training
        progress_bar.close()
        self.epochs += epoch
        print("best epoch and error:", self.best)

    def test(self):
        '''
        test on relation and entity prediction and record average rankings of expected output
        '''
        rankings = [[], [], []]
        top_10p_hits = [0, 0, 0]
        top_1_hits = [0, 0, 0]
        test_data = [i for i in self.data.test_data()] + [i for i in self.data.valid_data()]
        for data in tqdm(test_data):
            # convert data and create one-hot vectors
            data = [
                data[0], data[2], data[1]
            ]
            data_oh = [
                get_one_hot(data[0], self.entity_dim), # h
                get_one_hot(data[1], self.relation_dim), # r
                get_one_hot(data[2], self.entity_dim), # t
                ]
            
            # get outputs and record rankings
            encoded, noise_levels, norm_noises, disturbed, translated, decoded = self.get_outputs(data_oh, noisy=False)

            # test
            for i, prediction in enumerate(decoded):
                confidence = prediction[data[i]]
                sorted_pred = sorted(prediction, reverse=True)
                rank = sorted_pred.index(confidence) + 1
                total = self.data.relation_num if i == 1 else self.data.entity_num
                rankings[i].append(rank/total * 100)
                if rank <= total/10:
                    top_10p_hits[i] += 1 / len(test_data) * 100
                if rank == 1:
                    top_1_hits[i] += 1 / len(test_data) * 100
        
        avg_rankings = [sum(rank) / len(rank) for rank in rankings]
        print("average ranking percentages:", avg_rankings)
        print("top 10 percent hits:", top_10p_hits)
        print("accurate hits:", top_1_hits)

        return rankings
    
    def load_parameters(self, epoch: int, bs: int=None, lr: float=None, mt: float=None):
        if bs is None:
            save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', f'VAT-{self.data_name}')
        else:
            save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', f'VAT-{self.data_name}-bs_{bs}-lr_{lr}')
        with open(os.path.join(save_dir, f'{epoch}.pkl'), 'rb') as file:
            parameters = pickle.load(file)
            assert len(parameters) == 6

        # load parameters
        self.entity_encoder.load_weights(parameters[0])
        self.entity_decoder.load_weights(parameters[1])
        self.entity_noise.load_weights(parameters[2])
        self.relation_encoder.load_weights(parameters[3])
        self.relation_decoder.load_weights(parameters[4])
        self.relation_noise.load_weights(parameters[5])

        # load epochs and errors
        self.epochs = epoch
        with open(os.path.join(save_dir, 'errors.tmp'), 'r') as file:
            errors = file.readlines()
            for error in errors:
                self.errors.append(eval(error))

    def get_outputs(self, data: Sequence, noisy=True):
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
            self.entity_noise.forward([d / sum(encoded[0]) for d in encoded[0]]), 
            self.relation_noise.forward([d / sum(encoded[1]) for d in encoded[1]]), 
            self.entity_noise.forward([d / sum(encoded[2]) for d in encoded[2]])
            ]
        norm_noises = [ # noise to be added to encoded representations
            [random.normalvariate() for _ in range(self.latent_dim)] # noises for each vector
            for _ in range(3)
            ]
        if noisy:
            disturbed = [ # noisy representations
                [encoded[i][d] + noise_levels[i][d] * norm_noises[i][d] for d in range(self.latent_dim)]
                for i in range(3)
            ]
        else:
            disturbed = encoded
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
    
    def set_hyper(self, lr, mt):
        '''
        set learning rate and momentum for all nets
        '''
        self.entity_encoder.set_learning_rate(lr)
        self.entity_decoder.set_learning_rate(lr)
        self.relation_encoder.set_learning_rate(lr)
        self.relation_decoder.set_learning_rate(lr)
        self.entity_noise.set_learning_rate(lr)
        self.relation_noise.set_learning_rate(lr)
        self.entity_encoder.set_momentum(mt)
        self.entity_decoder.set_momentum(mt)
        self.relation_encoder.set_momentum(mt)
        self.relation_decoder.set_momentum(mt)
        self.entity_noise.set_momentum(mt)
        self.relation_noise.set_momentum(mt)
