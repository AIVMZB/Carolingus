import numpy as np
import torch
from .model import SyntaxEncoder
from .common import IndexedEmbeddings
from typing import List


def preprocess(image: np.ndarray) -> torch.Tensor:
    image = torch.from_numpy(image.transpose(2, 0, 1)).cuda()
    return torch.unsqueeze(image, 0)


def inference(image: np.ndarray, embeddings: dict[str: list], weights: str, output_dim: int = 32) -> List[str]:
    model = SyntaxEncoder(output_dim)
    model.load_state_dict(torch.load(weights, weights_only=True))

    image = preprocess(image)
    image_vector = model(image)[0].cpu().numpy()

    indexed_embeddings = IndexedEmbeddings(embeddings)

    return indexed_embeddings.find_closest_words(image_vector)
