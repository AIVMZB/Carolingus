from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import (
    Pad,
    Resize,
    RandomRotation,
    GaussianBlur,
    Compose,
    ToDtype,
    Normalize,
    RandomPerspective,
)
import torch
import pandas as pd
from tqdm import tqdm
from typing import Literal, Union
import numpy as np
import random
from os.path import join as opj
import os
import cv2

from .common import string_distance, get_word_from_image_name


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
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self._dataset_root = dataset_root
        self._image_format = image_format
        self._output_size = output_size
        self._positive_pairs_df = positive_pairs_df
        self._images = set(positive_pairs_df["anchor"])
        random.seed(seed)

        self._device = device
        self._augmentation = Compose(
            [
                RandomRotation(degrees=15),
                RandomPerspective(),
                Resize(self._output_size),
                ToDtype(torch.float32),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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


class ImgToWordDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        images: list | None,
        image_format: Literal["RGB", "BGR"] = "RGB",
        img_size: int = 120,
        add_batch_dim: bool = False,
        augment: bool = False,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self._image_format = image_format
        self._dataset_root = dataset_root
        self._img_size = (img_size, img_size)
        self._add_batch_dim = add_batch_dim
        self._device = device
        self._image_paths = images

        transforms = [
            ToDtype(torch.float32),
            Resize(self._img_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if augment:
            augmentations = [RandomRotation(degrees=15), RandomPerspective()]
            transforms = augmentations + transforms

        self.transform = Compose(transforms)

    def __len__(self) -> int:
        return len(self._image_paths) - 1

    def __getitem__(self, index):
        image_path = os.path.join(self._dataset_root, self._image_paths[index])
        word = get_word_from_image_name(image_path)
        image = cv2.imread(image_path)
        if self._image_format == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = torch.from_numpy(image.transpose(2, 0, 1)).to(self._device)

        if self._add_batch_dim:
            image = torch.unsqueeze(image, 0)

        image = self.transform(image)

        return {"word": word, "image": image, "image_path": image_path}
    
    @property
    def dataset_root(self) -> str:
        return self._dataset_root


class HardTripletsDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        embeddings: dict,
        str_dist_threshold: int,
        subset_images: list,
        image_format: str = "RGB",
        img_size: int = 120,
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        self._dataset_root = dataset_root
        self._subset_images = subset_images
        self._image_format = image_format
        self._image_size = (img_size, img_size)
        self._device = device

        self._triplets = find_hardest_triplets_in_embeddings(
            embeddings, self._subset_images, str_dist_threshold
        )

        self._augmentation = Compose(
            [
                RandomRotation(degrees=15),
                RandomPerspective(),
                Resize(self._image_size),
                ToDtype(torch.float32),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self._triplets)

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

    def __getitem__(self, index):
        triplet = self._triplets[index]
        anchor = triplet["anchor"]
        positive = triplet["positive"]
        negative = triplet["negative"]

        anchor_image = self.read_image(anchor)
        positive_image = self.read_image(positive)
        negative_image = self.read_image(negative)

        return anchor_image, positive_image, negative_image


def find_hardest_triplets_in_embeddings(
    embeddings_per_image: dict, image_names: list[str], threshold: int
) -> list:
    hardest_triplets = []

    vectors = [embeddings_per_image[image_name] for image_name in image_names]
    vectors = torch.tensor(vectors)

    distances = torch.cdist(vectors, vectors)

    print("Searching for hard pairs...")
    for i, anchor_image_name in enumerate(tqdm(image_names)):
        anchor_word = get_word_from_image_name(anchor_image_name)
        furthest_positive: int | None = None
        nearest_negative: int | None = None

        for j, positive_image_name in enumerate(image_names):
            positive_word = get_word_from_image_name(positive_image_name)
            if (
                positive_image_name == anchor_image_name
                or string_distance(anchor_word, positive_word) > threshold
            ):
                continue

            if furthest_positive is None or distances[i, j] > furthest_positive:
                furthest_positive = j
        if furthest_positive is None:
            continue

        for k, negative_image_name in enumerate(image_names):
            negative_word = get_word_from_image_name(negative_image_name)
            if (
                negative_image_name == anchor_image_name
                or string_distance(positive_word, negative_word) <= threshold
            ):
                continue

            if nearest_negative is None or distances[i, k] < nearest_negative:
                nearest_negative = k
        if nearest_negative is None:
            continue

        hardest_triplets.append(
            {
                "anchor": anchor_image_name,
                "positive": image_names[furthest_positive],
                "negative": image_names[nearest_negative],
            }
        )

    return hardest_triplets


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = SyntaxEmbeddingTripletDataset(
        r"E:\Dyploma\Carolingus\Carolingus\datasets\word_dataset", (120, 120), "RGB"
    )

    data = dataset[np.random.randint(0, len(dataset))]

    first_image = data["second_image"].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

    plt.imshow(first_image)
    plt.show()


# Mean height = 120
# Mean width = 103
