from .Unit import Unit

import random
from math import sqrt

class Connection:
    # allowing unknown input units for construction
    def __init__(
            self, 
            sender: Unit = None, 
            recipient: Unit = None, 
            input_num: int = None, 
            output_num: int = None
            ):
        self.sender = sender
        self.recipient = recipient
        if input_num == None:
            self.weight = random.random()
        else: # kaiming initialization (without which complicated optimization won't work at all...fantastic)
            self.weight = random.normalvariate(0, sqrt(2 / input_num))
        self.input_num = input_num
        self.ouput_num = output_num
        self.weight_change = 0

    def set_recipient(self, unit):
        self.recipient = unit

    def set_sender(self, unit):
        self.sender = unit

    def set_weight(self, weight):
        self.weight = weight
    
    # returning the current weight change (not including momentum)
    def update_weight_change(self, learning_rate = 0.5) -> float:
        self.weight_change += learning_rate * self.recipient.error * self.sender.activation
        return self.weight_change

    def update_weight(self, momentum = 0):
        self.weight += self.weight_change
        self.weight_change *= momentum
        return self.weight
    
    def init(self):
        self.weight = 0
        self.weight_change = 0
        self.momentum = 0

    def get_weight(self) -> float:
        return self.weight
    
    def get_weight_change(self) -> float:
        return self.weight_change
    
    def same_activations(self) -> bool:
        return self.recipient.get_activation() == self.sender.get_activation()