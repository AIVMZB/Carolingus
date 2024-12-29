from .model import SyntaxEncoder
from .dataset import ImgToWordDataset
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


def find_closest_words(word_vec: torch.Tensor, embeddings_dict: dict, margin: float = 0.3, max_words: int = 5) -> list:
    def criterion(vector) -> float:
        vec = torch.Tensor(vector).cuda()
        return F.pairwise_distance(word_vec, vec)

    words = []
    for word, vector in embeddings_dict.items():
        if criterion(vector) <= margin:
            words.append(word)

        if len(words) == max_words:
            break
    
    return words


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


def find_gt_closest_words(word: str, vocabulary: list, threshold: int = 2, sort: bool = False) -> list:
    gt_closest_words = []
    for candidate in vocabulary:
        if string_distance(candidate, word) <= threshold:
            gt_closest_words.append(candidate)

    if sort:
        return sorted(gt_closest_words, key=lambda x: string_distance(word, x))
    return gt_closest_words


def measure(model: SyntaxEncoder, embeddings: dict[str, list], val_dataset: ImgToWordDataset) -> float:
    mean_nearest_precision = 0

    for data in tqdm(val_dataset):
        word = data["word"]
        image = data["image"]

        emb_word = model(image)
        pred_closest_words = find_closest_words(emb_word, embeddings, margin=2, max_words=5)
        gt_closest_words = find_gt_closest_words(word, list(embeddings.keys()), threshold=2)

        mean_nearest_precision += nearest_precision(pred_closest_words, gt_closest_words)
    
    mean_nearest_precision /= len(val_dataset)

    return mean_nearest_precision
