import torch.nn as nn
import torch.nn.init as init

class LenetModel(nn.Module):
    """
    Model architecture (Lenet)
    """
    def __init__(self, channels, kernel_size, input_size, output_size, dropout):
        super(LenetModel, self).__init__()

        def init_weights_zeros(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.zeros_(m.weight.data)
                if m.bias is not None:
                    init.zeros_(m.bias.data)

        self.model = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, 1, 1),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(channels, channels*2 , kernel_size, 1, 1),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(channels*2, channels*2*2 , kernel_size, 1, 1),
            nn.tanh(),
            nn.AvgPool2d(2, 2),

            nn.Dropout(dropout),

            nn.Flatten(),
            nn.Linear(channels*2*2 * (input_size // 8) * (input_size // 8), 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.model(x)