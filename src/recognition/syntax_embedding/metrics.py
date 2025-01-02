from .common import string_distance, get_word_from_image_name

import torch
from torch.nn import functional as F
from tqdm import tqdm


def nearest_precision(nearest_words: list[str], gt_nearest_words: list[str]):
    if len(nearest_words) == 0 and len(gt_nearest_words) == 0:
        return 1
    elif len(nearest_words) == 0 and len(gt_nearest_words) > 0:
        return 0

    count = 0
    for word in nearest_words:
        if word in gt_nearest_words:
            count += 1

    return count / len(nearest_words)


def find_closest_words(
    word_vec: torch.Tensor,
    embeddings_dict: dict,
    margin: float = 0.3,
    max_words: int = 5,
) -> list:
    def criterion(vector) -> float:
        vec = torch.Tensor(vector)
        return F.pairwise_distance(word_vec, vec)

    words = []
    for image_name, vector in embeddings_dict.items():
        if criterion(vector) <= margin:
            words.append(get_word_from_image_name(image_name))

        if len(words) == max_words:
            break

    return words


def find_gt_closest_words(
    word: str, candidates: list, threshold: int = 2, sort: bool = False
) -> list:
    gt_closest_words = set()
    word = get_word_from_image_name(word)

    for candidate in candidates:
        candidate_word = get_word_from_image_name(candidate)
        if string_distance(candidate_word, word) <= threshold:
            gt_closest_words.add(candidate_word)

    if sort:
        return sorted(gt_closest_words, key=lambda x: string_distance(word, x))
    return gt_closest_words


def measure(
    val_embeddings: dict[str, torch.Tensor],
    train_embeddings: dict[str, torch.Tensor],
    margin: float,
    threshold: int,
    max_words: int,
) -> float:
    mean_nearest_precision = 0

    candidates = list(train_embeddings.keys())

    for image_name, val_vector in tqdm(val_embeddings.items()):
        pred_closest_words = find_closest_words(
            val_vector, train_embeddings, margin, max_words
        )
        gt_closest_words = find_gt_closest_words(
            image_name, candidates, threshold, sort=True
        )
        mean_nearest_precision += nearest_precision(
            pred_closest_words, gt_closest_words
        )

    mean_nearest_precision /= len(val_embeddings)

    return mean_nearest_precision
