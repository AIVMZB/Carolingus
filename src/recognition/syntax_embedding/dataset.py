from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Pad, Resize, RandomRotation, GaussianBlur, Compose, ToDtype, Normalize
import torch
import pandas as pd
from typing import Literal, Union
import numpy as np
import random
from os.path import join as opj
import os
import cv2


class RandomGaussianBlur:
    def __init__(self, probability: float = 0.5):
        self._probability = probability
        self._tranformation = GaussianBlur(5, 0.1)

    def __call__(self, image):
        if np.random.rand() >= self._probability:
            return self._tranformation.forward(image)

        return image


class RandomPad:
    def __init__(self):
        self.max_pad = 10
        self.min_pad = 1
        self.transform = Pad(self.min_pad)

    def __call__(self, image):
        padding_size = np.random.randint(self.min_pad, self.max_pad)
        self.transform.padding = padding_size
        return self.transform.forward(image)


class SyntaxEmbeddingTripletDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        positive_pairs_df: pd.DataFrame,
        output_size: tuple[int, int] = (120, 120),
        image_format: Literal["BGR", "RGB", "GRAY"] = "BGR",
        seed: int = 42,
        device: Union[str, torch.device] = "cuda"
    ) -> None:
        self._dataset_root = dataset_root
        self._image_format = image_format
        self._output_size = output_size
        self._positive_pairs_df = positive_pairs_df
        self._images = set(positive_pairs_df["anchor"])
        random.seed(seed)

        self._device = device
        self._augmentation = Compose([
            RandomRotation(degrees=15),
            Resize(self._output_size),
            ToDtype(torch.float32),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def read_image(self, relative_image_path: str) -> torch.Tensor:
        image_path = os.path.join(self._dataset_root, relative_image_path)
        np_image = cv2.imread(image_path)
        if self._image_format == "RGB":
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        elif self._image_format == "GRAY":
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
            np_image = np.expand_dims(np_image, 2)

        image = torch.from_numpy(np_image.transpose(2, 0, 1))
        image = self._augmentation(image).to(self._device)

        return image

    def __getitem__(self, index) -> tuple[str, np.ndarray]:
        row = self._positive_pairs_df.iloc[index]
        anchor = row["anchor"]
        positives = row["positive"].split(":")
        positive = random.choice(positives)
        negatives = self._images - set(positives) - set([anchor])
        negative = random.choice(list(negatives))

        anchor_image = self.read_image(anchor)
        positive_image = self.read_image(positive)
        negative_image = self.read_image(negative)

        return anchor_image, positive_image, negative_image

    def __len__(self):
        return len(self._positive_pairs_df)

    @property
    def dataset_root(self) -> str:
        return self._dataset_root

    @property
    def output_size(self) -> tuple[int, int]:
        return self._output_size

    @staticmethod
    def get_word_from_image_name(image_name: str) -> str:
        basename = os.path.basename(image_name)
        return basename.split("-")[1].split(".")[0].lower()


class ImgToWordDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        image_format: Literal["RGB", "BGR"] = "RGB",
        img_size: int = 120,
        images: list | None = None
    ):
        super().__init__()
        self._image_format = image_format
        self._dataset_root = dataset_root
        self._img_size = (img_size, img_size)
        if images is None:
            self._image_paths: list = []
            for document in os.listdir(dataset_root):
                for image in os.listdir(opj(dataset_root, document)):
                    self._image_paths.append(
                        opj(dataset_root, document, image)
                    )
        else:
            self._image_paths = images

        self.transform = Compose([
            ToDtype(torch.float32),
            Resize(self._img_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self._image_paths) - 1

    @staticmethod
    def get_word_from_image_name(image_name: str) -> str:
        basename = os.path.basename(image_name)
        return basename.split("-")[1].split(".")[0].lower()

    def __getitem__(self, index):
        image_path = os.path.join(self._dataset_root, self._image_paths[index])
        word = self.get_word_from_image_name(image_path)
        image = cv2.imread(image_path)
        if self._image_format == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = torch.unsqueeze(image, 0).cuda()

        image = self.transform(image)

        return {
            "word": word,
            "image": image,
            "image_path": image_path
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = SyntaxEmbeddingTripletDataset(
        r"E:\Dyploma\Carolingus\Carolingus\datasets\word_dataset",
        (120, 120),
        "RGB"
    )

    data = dataset[np.random.randint(0, len(dataset))]

    first_image = data["second_image"].cpu(
    ).numpy().transpose(1, 2, 0).astype(np.uint8)

    plt.imshow(first_image)
    plt.show()


# Mean height = 120
# Mean width = 103
