import torch

class Optimizer():
    def __init__(self, config: dict):
        self.optimizer = config['optimizer']['name']
        self.learning_rate = config['optimizer']['lr']
        self.momentum = config['optimizer']['momentum']
        self.weight_decay = config['optimizer']['weight_decay']

    def optim(self, model):
        if self.optimizer == 'Adam':
            return torch.optim.Adam(model.parameters(), self.learning_rate)
        elif self.optimizer == 'SGD':
            return torch.nn.optim.SGD(model.parameters(), self.learning_rate)
        else:
            return torch.optim.RMSprop(model.parameters(), self.learning_rate)