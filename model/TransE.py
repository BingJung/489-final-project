import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import get_dataset

from typing import Sequence
from math import ceil, log, sqrt, copysign
import random
from tqdm import tqdm

class TransE:
    def __init__(self, dataset: str):
        self.data_name = dataset
        self.data = get_dataset(dataset.upper() + "_filtered")
        self.errors = []

        # get input and latent dimensions
        dim_map = lambda x : ceil(x / log(x))
        self.entity_dim = self.data.entity_num
        self.relation_dim = self.data.relation_num
        self.latent_dim = ceil((dim_map(self.entity_dim) + dim_map(self.relation_dim)) / 2)
        print(f"building TransE with {self.entity_dim} entities, {self.relation_dim} relations, and latent dimension {self.latent_dim}.")

        # initialize representations
        self.entities = [[random.normalvariate(0, sqrt(2 / latent_dim))] for _ in range(entity_dim) for _ in range(self.entity_dim)]
        self.relations = [[random.normalvariate(0, sqrt(2 / latent_dim))] for _ in range(relations) for _ in range(self.relation_dim)]

    def train(self, epochs=100, batch_size=20, lr=0.005, save=False):
        # start training
        self.errors = []
        with open(f'TransE_{self.data_name}_errors.tmp', 'w') as file: # clear temporary error record
            pass
        for round in range(epochs):
            # total error records for each batch:
            batch_errors = [] 

            # summed errors for update in each batch:
            entity_diff_sum = [[0 for _ in range(self.latent_dim)] for _ in range(self.entity_dim)] 
            relation_diff_sum = [[0 for _ in range(self.latent_dim)] for _ in range(self.relation_dim)]
            for num, data in tqdm(enumerate(self.data.train_data()), total=self.data.train_num, desc=f"epochs {round+1}/{epochs}"):
                # label data
                data = {'h': data[0], 'r': data[2], 't': data[1]}
                
                # make predictions and record difference
                translated = [
                    [
                        entities[data['t']][d] - entities[data['r']][d] for d in range(self.latent_dim)
                        ],
                    [
                        entities[data['t']][d] - entities[data['h']][d] for d in range(self.latent_dim)
                        ],
                    [
                        entities[data['h']][d] + entities[data['r']][d] for d in range(self.latent_dim)
                        ]
                ]
                for key in data: # h, r, t
                    # update each dimension in vector
                    for d in range(self.latent_dim):
                        if key == 'r':
                            relation_diff_sum[data[key]][d] += (translated[data[key]][d] - relations[data[key]][d])
                        else: # h or t
                            entity_diff_sum[data[key]][d] += (translated[data[key]][d] - entities[data[key]][d])

                # udpate representations for each batch
                if (num+1) % batch_size == 0 or num == self.data.train_num - 1:
                    for d in range(self.latent_dim):
                        relations[data[key]][d] += relation_diff_sum[data[key]][d] / batch_size * lr
                        entities[data[key]][d] += entity_diff_sum[data[key]][d] / batch_size * lr
                    batch_errors.append(
                        sum(
                            [
                                sum(relation_diff_sum[id][d]) for d in range(self.latent_dim)
                            ]
                            for id in range(self.relation_dim)
                        ) / batch_size
                        +
                        sum(
                            [
                                sum(entity_diff_sum[id][d]) for d in range(self.latent_dim)
                            ]
                            for id in range(self.entity_dim)
                        ) / batch_size
                    )
                    # initialize error sum
                    entity_diff_sum = [[0 for _ in range(self.latent_dim)] for _ in range(self.entity_dim)] 
                    relation_diff_sum = [[0 for _ in range(self.latent_dim)] for _ in range(self.relation_dim)]

                    # save batch error and reset
                    with open(f'TransE_{self.data_name}_errors.tmp', 'a') as file:
                        file.write(str(batch_errors[-1]) + ', ')
                
            # at the end of each epoch:
            # create new line in error.tmp
            with open(f'TransE_{self.data_name}_errors.tmp', 'a') as file:
                file.write('\n')

            # refresh process bar
            print(end='\r')

            # save parameters
            if save == True: # mind the order
                parameters = [
                    self.entities, 
                    self.relations
                ]

                save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
                os.makedirs(save_dir, exist_ok=True)

                save_name = f'./checkpoints/TransE_{self.data_name}-lr{lr}-{epochs}.pkl'
                previous_name = f'./checkpoints/TransE_{self.data_name}-lr{lr}-{epochs-1}.pkl'
                if os.path.exists(os.path.join(save_dir, previous_name)):
                    os.remove(os.path.join(save_dir, previous_name))
                with open(os.path.join(save_dir, save_name), 'wb') as file:
                    pickle.dump(parameters, file)