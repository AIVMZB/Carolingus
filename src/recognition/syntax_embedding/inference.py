import numpy as np
import torch
from .model import SyntaxEncoder
from .common import IndexedEmbeddings
from torchvision.transforms.v2 import Resize, Compose, ToDtype, Normalize
from typing import Union
import json


class SyntaxEncoderInference:
    def __init__(
        self,
        weights: str,
        embeddings: Union[str, dict[str, np.ndarray]],
        device: str = "cuda",
    ):
        if isinstance(embeddings, str):
            embeddings = json.load(open(embeddings))
            for key, value in embeddings.items():
                embeddings[key] = torch.tensor(value, dtype=torch.float64)

        self._embeddings: dict[str, np.ndarray] = embeddings
        self._indexed_embeddings = IndexedEmbeddings(embeddings)

        self._output_dim: int = list(embeddings.values())[0].shape[0]

        self._device = device

        self._model = SyntaxEncoder(output_dim=self._output_dim).to(self._device)
        self._model.load_state_dict(torch.load(weights, weights_only=True))
        self._model.eval()

        self._transforms = Compose(
            [
                Resize((120, 120)),
                ToDtype(torch.float32),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = self._transforms(image).to(self._device)
        image = torch.unsqueeze(image, 0)
        return image

    def inference(self, image: np.ndarray, max_words: int = 1) -> list[str]:
        self._model.eval()
        with torch.no_grad():
            image = self._preprocess(image)
            vector = self._model(image)[0]
        return self._indexed_embeddings.find_closest_words(vector, max_words, margin=2)
