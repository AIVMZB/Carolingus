import torch
from torch import nn
from torchvision.models import resnet50


class WordClassifier(nn.Module):
    def __init__(self, num_of_classes: int):
        super().__init__()
        self.num_of_classes = num_of_classes
        resnet = resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        self.head = nn.Sequential(
            nn.Linear(2048),
            nn.ReLU(),
            nn.Linear(2048),
            nn.ReLU(),
            nn.Linear(2048),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        features: torch.Tensor = self.cnn(x).flatten()
        return self.head(features)
