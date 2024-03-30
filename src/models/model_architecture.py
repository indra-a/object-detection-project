import torch.nn as nn
import torch.nn.init as init

class LenetModel(nn.Module):
    """
    Model architecture (Lenet)
    """
    def __init__(self, config: dict):
        super(LenetModel, self).__init__()
        self.channels = config['model']['channels']
        self.kernel_size = config['model']['kernel_size']
        self.input_size = config['model']['input_size']
        self.output_size = config['model']['output_size']
        self.dropout = config['model']['dropout']

        def init_weights_zeros(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.zeros_(m.weight.data)
                if m.bias is not None:
                    init.zeros_(m.bias.data)

        self.model = nn.Sequential(
            nn.Conv2d(3, self.channels, self.kernel_size, 1, 1),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(self.channels, self.channels*2 , self.kernel_size, 1, 1),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(self.channels*2, self.channels*2*2 , self.kernel_size, 1, 1),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Dropout(self.dropout),

            nn.Flatten(),
            nn.Linear(self.channels*2*2 * ((self.input_size // 8)-2) * ((self.input_size // 8)-2), 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size)
        )

    def forward(self, x):
        return self.model(x)