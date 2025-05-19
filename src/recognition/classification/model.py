import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten_size = 64 * 56 * 56  
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  
        x = self.dropout(F.relu(self.fc1(x)))
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(1024, num_classes)  

    def forward(self, x):
        return self.fc2(x)


class FullModel(nn.Module):
    def __init__(self, num_classes):
        super(FullModel, self).__init__()
        resnet = resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1]) 
        self.flatten_size = 2048  
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.flatten_size, 2048),
            nn.ReLU(),
            nn.Linear(self.flatten_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        features = self.cnn(x)  # Output shape: (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # Flatten to (batch_size, 2048)
        output = self.classifier(features)  # Pass through classifier
        return output
