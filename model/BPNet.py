# requirement: customizing learning rate, momentum
from .Connection import Connection
from .Unit import Unit

from typing import Sequence
from itertools import product, pairwise # "pairwise" requires python 3.10 or later

class BPNet:
    def __init__(self, units_nums: Sequence, learning_rate = 0.01, momentum = 0) -> None:
        '''
        input: units_nums - a list specifying number of units in each layer of the network;
                            length of the list will be the number of layeres of the network
                            e.g. [2, 4, 8, 16]
        '''
        self.units = [[Unit() for _ in range(num)] for num in units_nums]
        self.layer_num = len(units_nums) - 1
        self.unit_nums = units_nums
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.Gs = [] # global errors; only modified by train_until_settle

        # build connections and add to units
        self.connections_forward = [{f"({unit1}, {unit2})" : Connection(self.units[layer][unit1], self.units[layer+1][unit2]) for unit1, unit2 in product(range(units_nums[layer]), range(units_nums[layer+1]))} for layer in range(self.layer_num)]
        # connection is a list of dictionaries each of which refers to a layer

        for layer1, layer2 in pairwise(range(self.layer_num + 1)):
            for unit1, unit2 in product(range(units_nums[layer1]), range(units_nums[layer2])):
                self.units[layer1][unit1].add_connection(self.connections_forward[layer1][f"({unit1}, {unit2})"])
                self.units[layer2][unit2].add_connection(self.connections_forward[layer1][f"({unit1}, {unit2})"])

    # forward activation
    def forward(self, pattern: Sequence) -> list: 
        '''
        input & output: vector as a list
        '''
        assert len(pattern) == self.unit_nums[0], f"input size is supposed to be {self.unit_nums[0]} rather than {len(pattern)}"
        # set activations in the first layer; inputs are not set
        for i in range(len(pattern)):
            self.units[0][i].set_activation(pattern[i])

        # update the rest layers
        for layer_id, layer in enumerate(self.units):
            if layer_id == 0:
                continue
            for unit in layer:
                unit.get_input()
                unit.update_activation()

        # if bin: # return binary outputs
        #     return [1 if unit.get_activation() > 0.5 else 0 for unit in self.units[-1]]
        # else:
        return self.get_output()

    def backward(self, target: Sequence, result: Sequence=None):
        '''
        back-propagate the error
        input: 
        target - the expected result of the network
        result (optional) - the result to compute error with
        '''
        assert len(target) == self.unit_nums[-1], f"target size is supposed to be {self.unit_nums[-1]}"
        if result != None:
            assert len(result) == self.unit_nums[-1], f"result size is supposed to be {self.unit_nums[-1]}"
            self.set_output(result)

        # update errors in the last layer
        for id, unit in enumerate(self.units[-1]):
            unit.update_unit_error(target[id])

        # back propagates errors
        for layer in reversed(self.units[:-1]):
            for unit in layer:
                unit.sum_forward_error()

    def update_weight_changes(self):
        '''
        update expected change of each connection in the network
        '''
        for weight_layer in self.connections_forward:
            for conn in weight_layer.values():
                conn.update_weight_change(self.learning_rate)

    def update_weights(self):
       '''
       update weight of each connection
       (with momentum wrt the weight change in last update)
       '''
       for weight_layer in self.connections_forward:
            for conn in weight_layer.values():
                conn.update_weight(self.momentum) 
    
    def set_learning_rate(self, learning_rate) -> None:
        self.learning_rate = learning_rate

    def set_momentum(self, momentum) -> None:
        self.momentum = momentum
    
    def get_output(self) -> list:
        return [unit.get_activation() for unit in self.units[-1]]
    
    def set_output(self, activations) -> None:
        for id, unit in enumerate(self.units[-1]):
            unit.set_activation(activations[id])
    
    def get_weights(self) -> list:
        return [[c.get_weight() for c in layer.values()] for layer in self.connections_forward]
    
    def load_weights(self, weights: Sequence) -> None:
        '''
        load a set of weights in network
        input: weights: a list of layer weights
                        (can consists of more or less layers of weights than this network
                        as long as number of units matches)
        '''
        # check if weight size matches first
        length = min(self.layer_num, len(weights))
        for i in range(length):
            assert isinstance(weights, Sequence), f"Expecting a sequence at layer {i}."
            assert len(self.connections_forward[i]) == len(weights[i]), \
                f"At layer {i} expecting size {len(self.connections_forward[i])} but get {len(weights[i])}."
            
        # load weights
        for layer in range(length):
            for i, key in enumerate(self.connections_forward[layer]):
                self.connections_forward[layer][key] = weights[layer][i]

    def init(self):
        for conns in self.connections_forward.values():
            for c in conns:
                c.init()
        self.training_patterns = []
        self.Gs = []


