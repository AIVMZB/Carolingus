from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Pad, Resize, RandomRotation, GaussianBlur, Compose, ToDtype, Normalize
import torch
from typing import Literal, Union
import numpy as np
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


class SyntaxEmbeddingDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        output_size: tuple[int, int] = (120, 120),
        image_format: Literal["BGR", "RGB", "GRAY"] = "BGR",
        image_paths: list | None = None,
        device: Union[str, torch.device] = "cuda"
    ) -> None:
        self._dataset_root = dataset_root
        self._image_format = image_format
        self._output_size = output_size
        self._images_paths = [] if image_paths is None else image_paths
        self._device = device
        self._augmentation = Compose([
            RandomPad(),
            RandomRotation(degrees=15),
            RandomGaussianBlur(),
            Resize(self._output_size),
            ToDtype(torch.float32),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for document in os.listdir(dataset_root):
            for image in os.listdir(opj(dataset_root, document)):
                self._images_paths.append(
                    opj(dataset_root, document, image)
                )

        self._pairs = []
        for i in range(len(self._images_paths)):
            for j in range(i, len(self._images_paths)):
                self._pairs.append(
                    [self._images_paths[i], self._images_paths[j]]
                )

    def with_pairs(self, pairs: list[list]) -> "SyntaxEmbeddingDataset":
        self._pairs = pairs
        return self

    def get_word_data(self, image_path) -> tuple[str, np.ndarray]:
        word_value = self.get_word_from_image_name(image_path)
        numpy_image = cv2.imread(image_path)
        if self._image_format == "RGB":
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        if self._image_format == "GRAY":
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)
            numpy_image = np.expand_dims(numpy_image, 2)

        image = torch.from_numpy(numpy_image.transpose(2, 0, 1))
        image = self._augmentation(image).to(self._device)

        return word_value, image

    def __getitem__(self, index) -> tuple[str, np.ndarray]:
        first_path, second_path = self._pairs[index]
        first_word, first_image = self.get_word_data(first_path)
        second_word, second_image = self.get_word_data(second_path)
        str_distance = self.string_distance(first_word, second_word)
        str_distance = torch.tensor(str_distance).to(self._device)
        
        return {
            "str_distance": str_distance,
            "first_image": first_image,
            "second_image": second_image
        }

    def __len__(self):
        return len(self._pairs)
    
    @property
    def pairs(self) -> list:
        return self._pairs
    
    @property
    def dataset_root(self) -> str:
        return self._dataset_root
    
    @property
    def output_size(self) -> tuple[int, int]:
        return self._output_size

    @staticmethod
    def get_word_from_image_name(image_name: str) -> str:
        basename = os.path.basename(image_name)
        return basename.split("-")[1].split(".")[0]

    @staticmethod
    def string_distance(str1: str, str2: str):
        len_str1 = len(str1) + 1
        len_str2 = len(str2) + 1

        distance_matrix = [[0] * len_str2 for _ in range(len_str1)]

        for i in range(len_str1):
            distance_matrix[i][0] = i
        for j in range(len_str2):
            distance_matrix[0][j] = j

        for i in range(1, len_str1):
            for j in range(1, len_str2):
                if str1[i - 1] == str2[j - 1]:
                    cost = 0
                else:
                    cost = 1

                distance_matrix[i][j] = min(distance_matrix[i - 1][j] + 1,
                                            distance_matrix[i][j - 1] + 1,
                                            distance_matrix[i - 1][j - 1] + cost)

        return distance_matrix[-1][-1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = SyntaxEmbeddingDataset(
        r"E:\Dyploma\Carolingus\Carolingus\datasets\word_dataset",
        (120, 120),
        "RGB"
    )

    data = dataset[np.random.randint(0, len(dataset))]

    first_image = data["first_image"].cpu().numpy().transpose(1, 2, 0)

    plt.imshow(first_image)
    plt.show()


### Mean height = 120
### Mean width = 103