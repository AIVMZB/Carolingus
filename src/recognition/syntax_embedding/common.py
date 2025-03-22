import os
import faiss
import torch
from typing import List
import numpy as np
import numba as nb

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class IndexedEmbeddings:
    def __init__(self, embeddings: dict[str, torch.Tensor]):
        """
        Class to index embeddings and find closest words to a given vector faster than using broot force search.
        """
        self._image_names = [image_name for image_name in embeddings.keys()]
        self._words = list(map(get_word_from_image_name, self._image_names))

        vectors = np.array(
            [vector.cpu().numpy().astype(np.float32) for vector in embeddings.values()]
        )
        self._indexed_embeddings = faiss.IndexFlatL2(vectors.shape[1])
        self._indexed_embeddings.add(vectors)

    def find_closest_words(
        self, vector: np.ndarray | torch.Tensor, max_words: int, margin: float = 2
    ) -> List[str]:
        """
        Finds closest words to a given vector in the indexed embeddings.
        Args:
            vector (np.ndarray | torch.Tensor): Vector to find closest words to.
            max_words (int): Maximum number of words to return.
            margin (float): Maximum distance between the vector and the word embedding.
        Returns:
            List[str]: List of closest words to the vector.
        """
        if isinstance(vector, torch.Tensor):
            vector = vector.cpu().numpy()

        vector = vector.reshape(1, -1).astype(np.float32)

        distances, indeces = self._indexed_embeddings.search(vector, max_words)
        words = []
        for i, distance in enumerate(distances[0]):
            # if distance < margin:
            #     words.append(self._words[indeces[0][i]])
            words.append(self._words[indeces[0][i]])

        return words

    @property
    def words(self) -> list:
        return self._words

    @property
    def image_names(self) -> list:
        return self._image_names


@nb.njit(cache=True)
def string_distance(str1: str, str2: str):
    """
    Computes the Levenshtein distance between two strings.
    Args:
        str1 (str): First string.
        str2 (str): Second string.
    Returns:
        int: Levenshtein distance between the two strings.
    """
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    distance_matrix = [[0] * len_str2 for _ in range(len_str1)]

    for i in range(len_str1):
        distance_matrix[i][0] = i
    for j in range(len_str2):
        distance_matrix[0][j] = j

    for i in nb.prange(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1

            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + 1,
                distance_matrix[i][j - 1] + 1,
                distance_matrix[i - 1][j - 1] + cost,
            )

    return distance_matrix[-1][-1]


def get_word_from_image_name(image_name: str) -> str:
    """
    Extracts a word from image path.
    Args:
        image_name (str): Path to the image.
    Returns:
        str: Word extracted from the image path.
    """
    if "-" not in image_name and "." not in image_name:
        return image_name.lower()

    basename = os.path.basename(image_name)
    return basename.split("-")[1].split(".")[0].lower()


def get_create_embeddings_dir(save_dir: str) -> str:
    """
    Creates a directory for embeddings.
    Args:
        save_dir (str): Directory to save embeddings to.
    Returns:
        str: Path to the embeddings directory.
    """
    embeddings_dir = os.path.join(save_dir, "embeddings")
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    return embeddings_dir
