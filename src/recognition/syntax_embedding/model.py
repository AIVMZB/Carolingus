import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50


class FCResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self._dim = dim
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        res = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return F.relu(res + x)


class SyntaxEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super(SyntaxEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        resnet = resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        self.head = nn.Sequential(
            FCResBlock(2048),
            nn.Linear(2048, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.cnn(x)
        net = torch.flatten(net, 1)
        
        return self.head(net)
