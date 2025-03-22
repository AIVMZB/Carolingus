from torch.utils.data import Dataset
from torchvision.transforms.v2 import (
    Pad,
    Resize,
    RandomRotation,
    GaussianBlur,
    Compose,
    ToDtype,
    # Normalize,
    RandomPerspective,
    ColorJitter,
    Lambda
)
import torch
import pandas as pd
from tqdm import tqdm
from typing import Literal, Union
import numpy as np
import random
from os.path import join as opj
import cv2
from functools import lru_cache

from .common import string_distance, get_word_from_image_name, IndexedEmbeddings


BALANCED_N_SAMPLES = 50

AUGMENTATIONS = (
    RandomRotation(degrees=15),
    RandomPerspective(),
    ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
)


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
        """
        Dataset for training triplet loss model for syntax embedding.
        Args:
            dataset_root (str): Path to the dataset root directory.
            positive_pairs_df (pd.DataFrame): DataFrame with positive pairs of images.
            output_size (tuple[int, int], optional): Output size of the images. Defaults to (120, 120).
            image_format (Literal["BGR", "RGB", "GRAY"], optional): Format of the images. Defaults to "BGR".
            seed (int, optional): Random seed. Defaults to 42.
            device (Union[str, torch.device], optional): Device to use. Defaults to "cuda".
        """
        self._dataset_root = dataset_root
        self._image_format = image_format
        self._output_size = output_size
        self._positive_pairs_df = positive_pairs_df
        self._images = set(positive_pairs_df["anchor"])
        random.seed(seed)

        self._device = device
        self._transforms = Compose(
            list(AUGMENTATIONS)
            + [
                Resize(self._output_size),
                ToDtype(torch.float32),
                Lambda(lambda x: x / 255.0)
                # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def read_image(self, relative_image_path: str) -> torch.Tensor:
        image_path = opj(self._dataset_root, relative_image_path)
        np_image = cv2.imread(image_path)
        if self._image_format == "RGB":
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        elif self._image_format == "GRAY":
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
            np_image = np.expand_dims(np_image, 2)

        image = torch.from_numpy(np_image.transpose(2, 0, 1))
        image = self._transforms(image).to(self._device)

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
        images: list,
        image_format: Literal["RGB", "BGR"] = "RGB",
        img_size: int = 120,
        add_batch_dim: bool = False,
        augment: bool = False,
        device: str | torch.device = "cuda",
    ):
        """
        Dataset to map images to words, written on them.
        Args:
            dataset_root (str): Path to the dataset root directory.
            images (list): List of image paths.
            image_format (Literal["RGB", "BGR"], optional): Format of the images. Defaults to "RGB".
            img_size (int, optional): Size of the images. Defaults to 120.
            add_batch_dim (bool, optional): Whether to add batch dimension to the images. Defaults to False.
            augment (bool, optional): Whether to apply augmentations to the images. Defaults to False.
            device (str | torch.device, optional): Device to use. Defaults to "cuda".
        """
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
            Lambda(lambda x: x / 255.0)
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if augment:
            augmentations = [RandomRotation(degrees=15), RandomPerspective()]
            transforms = augmentations + transforms

        self.transform = Compose(transforms)

    def __len__(self) -> int:
        return len(self._image_paths) - 1

    def __getitem__(self, index):
        image_path = opj(self._dataset_root, self._image_paths[index])
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
        balanced: bool = True,
        image_format: str = "RGB",
        img_size: int = 120,
        device: str | torch.device = "cuda",
    ):
        """
        Dataset to train triplet loss model with hard triplets.
        Args:
            dataset_root (str): Path to the dataset root directory.
            embeddings (dict): Embeddings of the images.
            str_dist_threshold (int): Threshold for string distance between image names.
            subset_images (list): List of image paths.
            balanced (bool, optional): Whether to balance the triplets. Defaults to True.
            image_format (str, optional): Format of the images. Defaults to "RGB".
            img_size (int, optional): Size of the images. Defaults to 120.
            device (str | torch.device, optional): Device to use. Defaults to "cuda".
        """
        super().__init__()

        self._dataset_root = dataset_root
        self._subset_images = subset_images
        self._image_format = image_format
        self._image_size = (img_size, img_size)
        self._balanced = balanced
        self._device = device

        self._triplets: pd.DataFrame = find_hardest_triplets_in_embeddings(
            embeddings, self._subset_images, str_dist_threshold
        )
        if self._balanced:
            self._triplets = balance_triplets(self._triplets, BALANCED_N_SAMPLES)

        self._transforms = Compose(
            list(AUGMENTATIONS)
            + [
                Resize(self._image_size),
                ToDtype(torch.float32),
                Lambda(lambda x: x / 255.0)
                # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self._triplets)

    def read_image(self, relative_image_path: str) -> torch.Tensor:
        image_path = opj(self._dataset_root, relative_image_path)
        np_image = cv2.imread(image_path)
        if self._image_format == "RGB":
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        elif self._image_format == "GRAY":
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
            np_image = np.expand_dims(np_image, 2)

        image = torch.from_numpy(np_image.transpose(2, 0, 1))
        image = self._transforms(image).to(self._device)

        return image

    def __getitem__(self, index):
        triplet = self._triplets.iloc[index]
        anchor = triplet["anchor"]
        positive = triplet["positive"]
        negative = triplet["negative"]

        anchor_image = self.read_image(anchor)
        positive_image = self.read_image(positive)
        negative_image = self.read_image(negative)

        return anchor_image, positive_image, negative_image


@lru_cache
def get_string_distances(image_names: tuple) -> np.ndarray:
    distance_matrix = np.zeros((len(image_names), len(image_names)), dtype=int)

    print("Creating string distance matrix...")
    for i in tqdm(range(len(distance_matrix))):
        first_image_name = image_names[i]
        first_word = get_word_from_image_name(first_image_name)
        for j in range(i + 1, len(distance_matrix)):
            second_image_name = image_names[j]
            second_word = get_word_from_image_name(second_image_name)

            distance = string_distance(first_word, second_word)

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def find_hardest_triplets_in_embeddings(
    embeddings_per_image: dict, image_names: list[str], threshold: int
) -> list:
    hardest_triplets = []

    vectors = [embeddings_per_image[image_name] for image_name in image_names]
    vectors = torch.tensor(vectors)

    l2_distances = torch.cdist(vectors, vectors)
    string_distances = get_string_distances(tuple(image_names))

    print("Searching for hard pairs...")
    for i, anchor_image_name in enumerate(tqdm(image_names)):
        furthest_positive: int | None = None
        nearest_negative: int | None = None

        for j, positive_image_name in enumerate(image_names):
            if i == j or string_distances[i, j] > threshold:
                continue

            if furthest_positive is None or l2_distances[i, j] > furthest_positive:
                furthest_positive = j
        if furthest_positive is None:
            continue

        for k, negative_image_name in enumerate(image_names):
            if i == k or string_distances[i, k] <= threshold:
                continue

            if nearest_negative is None or l2_distances[i, k] < nearest_negative:
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

    return pd.DataFrame(hardest_triplets)


def balance_triplets(triplets: pd.DataFrame, n_samples: int = 50) -> pd.DataFrame:
    triplets["anchor_word"] = triplets["anchor"].apply(get_word_from_image_name)
    words = set(triplets["anchor_word"].tolist())

    balanced_data = []
    for word in words:
        word_df = triplets[triplets["anchor_word"] == word]
        for _ in range(n_samples):
            triplet_idx = random.randint(0, len(word_df) - 1)
            triplet = triplets.iloc[triplet_idx]
            balanced_data.append(
                {
                    "anchor": triplet["anchor"],
                    "positive": triplet["positive"],
                    "negative": triplet["negative"],
                }
            )

    return pd.DataFrame(balanced_data)
