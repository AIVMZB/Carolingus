import torch
from torch import nn


class SyntaxEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super(SyntaxEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 30, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(30, 60, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(60, 100, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.head = nn.Sequential(
            nn.Linear(2500, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.cnn(x)
        net = torch.flatten(net, 1)
        
        return self.head(net)
