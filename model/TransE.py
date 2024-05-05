import sys, os, pickle, re
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
        self.epochs = 0 # pretrained-epoches
        self.best = (0, 10000) # (epoch, total error)
        self.errors = []

        # get input and latent dimensions
        dim_map = lambda x : ceil(x / log(x))
        self.entity_dim = self.data.entity_num
        self.relation_dim = self.data.relation_num
        self.latent_dim = ceil((dim_map(self.entity_dim) + dim_map(self.relation_dim)) / 2)
        print(f"building TransE with {self.entity_dim} entities, {self.relation_dim} relations, and latent dimension {self.latent_dim}.")

        # initialize representations
        self.entities = [[random.normalvariate(0, sqrt(2 / self.latent_dim)) for _ in range(self.entity_dim)] for _ in range(self.entity_dim)]
        self.relations = [[random.normalvariate(0, sqrt(2 / self.latent_dim)) for _ in range(self.relation_dim)] for _ in range(self.relation_dim)]

    def train(self, epochs=100, batch_size=20, lr=0.01, momentum=0, save=False):
        assert epochs > self.epochs, f"this model is already at epoch {self.epochs}"
        print("start training.")
        save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', f'TransE-{self.data_name}-bs_{batch_size}-lr_{lr}')
        os.makedirs(save_dir, exist_ok=True)
        self.errors = []
        if self.epochs == 0:
            with open(os.path.join(save_dir, 'errors.tmp'), 'w') as file: # clear temporary error record
                pass
        progress_bar = tqdm(total=epochs - self.epochs, desc="epochs")
        for round in range(epochs - self.epochs):
            epoch = round + self.epochs + 1

            # total error records for each batch:
            batch_errors = [] 

            # summed errors for update in each batch:
            entity_diff_sum = [[0 for _ in range(self.latent_dim)] for _ in range(self.entity_dim)] 
            relation_diff_sum = [[0 for _ in range(self.latent_dim)] for _ in range(self.relation_dim)]
            for num, data in enumerate(self.data.train_data()):
                # label data
                data = {'h': data[0], 'r': data[2], 't': data[1]}
                
                # make predictions and record difference
                translated = [
                    [
                        self.entities[data['t']][d] - self.entities[data['r']][d] for d in range(self.latent_dim)
                        ],
                    [
                        self.entities[data['t']][d] - self.entities[data['h']][d] for d in range(self.latent_dim)
                        ],
                    [
                        self.entities[data['h']][d] + self.entities[data['r']][d] for d in range(self.latent_dim)
                        ]
                ]
                for i, key in enumerate(data): # h, r, t
                    # update each dimension in vector
                    for d in range(self.latent_dim):
                        if key == 'r':
                            relation_diff_sum[data[key]][d] += (translated[i][d] - self.relations[data[key]][d])
                        else: # h or t
                            entity_diff_sum[data[key]][d] += (translated[i][d] - self.entities[data[key]][d])

                # udpate representations for each batch
                if (num+1) % batch_size == 0 or num == self.data.train_num - 1:
                    for d in range(self.latent_dim):
                        for i in range(self.relation_dim):
                            self.relations[i][d] += relation_diff_sum[i][d] / batch_size * lr
                        for i in range(self.entity_dim):
                            self.entities[i][d] += entity_diff_sum[i][d] / batch_size * lr
                    batch_errors.append(
                        sum([
                                sum(relation_diff_sum[id]) for id in range(self.relation_dim)
                                ]) / batch_size
                        +
                        sum([
                                sum(entity_diff_sum[id]) for id in range(self.entity_dim)
                            ]) / batch_size
                    )

                    # reset error sum
                    entity_diff_sum = [[0 for _ in range(self.latent_dim)] for _ in range(self.entity_dim)] 
                    relation_diff_sum = [[0 for _ in range(self.latent_dim)] for _ in range(self.relation_dim)]
                
            # at the end of each epoch:
            # create new line in error.tmp
            with open(os.path.join(save_dir, 'errors.tmp'), 'a') as file:
                file.write('\n')

            # update errors and progress bar
            progress_bar.update(1)
            self.errors.append(batch_errors)
            epoch_error = sum(batch_errors)
            progress_bar.set_postfix_str(f"total error: {epoch_error}", False)
            if abs(epoch_error) < abs(self.best[-1]):
                self.best = (epoch, epoch_error)
            with open(os.path.join(save_dir, 'errors.tmp'), 'a') as file:
                file.write(f"{epoch_error}\n")

            # save parameters
            if save == True: # mind the order
                parameters = [
                    self.entities, 
                    self.relations
                ]

                save_name = f'{epoch}.pkl'
                previous_name = f'{self.epochs+round}.pkl'
                if os.path.exists(os.path.join(save_dir, previous_name)) and (epoch - 1) % 50 != 0:
                    os.remove(os.path.join(save_dir, previous_name))
                with open(os.path.join(save_dir, save_name), 'wb') as file:
                    pickle.dump(parameters, file)

        progress_bar.close()
        self.epochs += epochs
        print("best epoch and error:", self.best)

    def test(self):
        rankings = [[], [], []]
        top_10_hits = [0, 0, 0]
        for data in tqdm(self.data.test_data(), total=self.data.test_num):
            # convert data and get representations
            data = [
                data[0], data[2], data[1]
            ]
            representations = [
                self.entities[data[0]],
                self.relations[data[1]],
                self.entities[data[2]]
            ]
            expected = [
                [representations[2][d] - representations[1][d] for d in range(self.latent_dim)],
                [representations[2][d] - representations[0][d] for d in range(self.latent_dim)],
                [representations[0][d] + representations[1][d] for d in range(self.latent_dim)]
                ]
            print(self.relations)
            for i in range(3):
                # rank
                if i == 1:
                    metrics = [sum([(self.relations[id][d] - expected[i][d])**2 for d in range(self.latent_dim)]) for id in range(self.relation_dim)]
                else: 
                    metrics = [sum([(self.entities[id][d] - expected[i][d])**2 for d in range(self.latent_dim)]) for id in range(self.entity_dim)]

                inconfidence = metrics[data[i]]
                sorted_pred = sorted(metrics)
                rank = sorted_pred.index(inconfidence) + 1
                rankings[i].append(rank)
                if rank <= 10:
                    top_10_hits[i] += 1 / self.data.test_num * 100

        avg_rankings = [sum(rank) / len(rank) for rank in rankings]
        print(avg_rankings)
        print(top_10_hits)

        return rankings

    def load_parameters(self, bs: int, lr: float, mt, epoch: int):
        save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', f'TransE-{self.data_name}-bs_{bs}-lr_{lr}')
        with open(os.path.join(save_dir, f'{epoch}.pkl'), 'rb') as file:
            parameters = pickle.load(file)
            assert len(parameters) == 2
        self.entities = parameters[0]
        self.relations = parameters[1]
        self.epochs = epoch