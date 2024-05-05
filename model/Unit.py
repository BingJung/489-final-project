from math import exp

class Unit:
    def __init__(self, layer_size = None, next_layer_size = None) -> None:
        self.input = 0.0
        self.activation = 0.0
        self.error = 0
        self.incoming_connections = []
        self.outgoing_connections = []

        # not used for now
        self.layer_size = layer_size
        self.next_layer_size = next_layer_size

    # question: how to enable class constraint here and avoid circular import
    def add_connection(self, connection):
        assert connection.recipient == self or connection.sender == self, \
        "target unit is not involved in the input connection."
        assert connection.recipient != None and connection.sender != None, \
        "the connection is not complete."
        # allowing for self-self connection
        if connection.recipient == self:
            self.incoming_connections.append(connection)
        if connection.sender == self:
            self.outgoing_connections.append(connection)
        
    def get_input(self):
        self.input = sum(i.sender.activation * i.weight for i in self.incoming_connections)
        return self.input
    
    def get_activation(self):
        return self.activation
    
    def update_unit_error(self, aim: float) -> float:
        self.error = (aim - self.activation) * self.activation * (1 - self.activation)
        return self.error
    
    def get_error(self) -> float:
        return self.error
    
    def sum_forward_error(self) -> float:
        weighted_errors = [conn.get_weight() * conn.recipient.get_error() * self.activation * (1 - self.activation) for conn in self.outgoing_connections]
        self.error = sum(weighted_errors)
        return self.error

    def update_activation(self) -> bool:
        # print("unit input:", self.input, "unit activation:", orig)
        # self.activation = 1 / (1 + exp(-self.input))
        self.activation = 1 / (1 + exp(-self.input)) # Leaky ReLU
        return self.activation
    
    def set_activation(self, activation):
        self.activation = activation
    
    def init_io(self):
        self.input = 0
        self.activation = 0



    