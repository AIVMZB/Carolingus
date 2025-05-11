import os
from PIL import Image
from torch.utils.data import Dataset
from typing import NamedTuple
import random
import json


class ImageLabel(NamedTuple):
    image: str
    label: int


class CustomImageDataset(Dataset):
    def __init__(
        self, root_dir, label_to_idx_path=None, transform=None, n_samples: int = 50
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.data: list[ImageLabel] = []
        self.n_samples = n_samples

        # Отримуємо реальні класи з `val_dir`
        actual_classes = {d.name for d in os.scandir(root_dir) if d.is_dir()}

        if label_to_idx_path:
            with open(label_to_idx_path, "r") as f:
                full_label_to_idx = json.load(f)  # Завантажуємо всі мапінги

            self.label_to_idx = {
                cls: idx
                for cls, idx in full_label_to_idx.items()
                if cls in actual_classes
            }
        else:
            self.label_to_idx = {cls: i for i, cls in enumerate(sorted(actual_classes))}

        for class_name in self.label_to_idx:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                img_paths = [
                    os.path.join(class_dir, img_file)
                    for img_file in os.listdir(class_dir)
                ]

                # Sample data to deal with unbalanced data
                sampled_images = self._sample_images(img_paths)

                for sampled_image_path in sampled_images:
                    self.data.append(
                        ImageLabel(sampled_image_path, self.label_to_idx[class_name])
                    )

        self._shuffle_data()

    def _shuffle_data(self):
        #combined = list(zip(self.image_paths, self.labels))
        random.shuffle(self.data)
        #self.image_paths, self.labels = zip(*combined) if combined else ([], [])

    def _sample_images(self, images: list) -> list:
        if self.n_samples is None:
            return images

        if len(images) > self.n_samples:
            return random.sample(images, self.n_samples)
        elif len(images) == self.n_samples:
            return images
        elif len(images) < self.n_samples:
            return [random.choice(images) for _ in range(self.n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
