from torchvision import transforms
from dataset import CustomImageDataset
from torch.utils.data import random_split, DataLoader
import json
import os

def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


def get_dataloader(train_dir, val_dir, batch_size=32):
    transform = transforms.Compose(
        [   transforms.Resize((200, 200)),
            #transforms.RandomCrop(size=120),
            transforms.ElasticTransform(alpha=50.0, sigma=2.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    label_to_idx_path = "train-labels.json"
    if os.path.exists(label_to_idx_path):
        with open(label_to_idx_path, "r") as f:
            label_to_idx = json.load(f)
    else:
        label_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(os.listdir(train_dir)))}
        with open(label_to_idx_path, "w") as f:
            json.dump(label_to_idx, f, indent=2)

    train_dataset = CustomImageDataset(root_dir=train_dir, label_to_idx_path=label_to_idx_path, transform=transform)
    val_dataset = CustomImageDataset(root_dir=val_dir, label_to_idx_path=label_to_idx_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

