import json
import torch
import numpy as np
from torchvision import transforms

from .model import FullModel


class WordClassifier:
    def __init__(self, weights: str, labels: str | dict, device: str = "cpu"):
        if isinstance(labels, str):
            labels = json.load(open(labels))
        else:
            labels = labels

        self._labels = self.reverse_labels(labels)

        self._device = device

        self._model = FullModel(len(labels)).to(self._device)
        self._model.load_state_dict(torch.load(weights, weights_only=True, map_location=self._device))

        self.transform = transforms.Compose(
            [
                transforms.Resize((200, 200)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @staticmethod
    def reverse_labels(labels: dict[str, int]) -> dict[int, str]:
        idx_to_label = {}
        for word, idx in labels.items():
            idx_to_label[idx] = word

        return idx_to_label

    def preprocess(self, image: np.ndarray):
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0).to(torch.float32)
        image = self.transform(image).to(self._device)

        return image

    def classify(self, image: np.ndarray) -> tuple[str, float]:
        image = self.preprocess(image)

        model_output: torch.Tensor = self._model(image)[0]
        probs = model_output.softmax(dim=0)

        max_class_idx = probs.argmax(0).item()

        return self._labels[max_class_idx], probs[max_class_idx]
