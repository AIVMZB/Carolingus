import os


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


def get_word_from_image_name(image_name: str) -> str:
    if "-" not in image_name and "." not in image_name:
        return image_name.lower()

    basename = os.path.basename(image_name)
    return basename.split("-")[1].split(".")[0].lower()


def get_create_embeddings_dir(save_dir: str) -> str:
    embeddings_dir = os.path.join(save_dir, "embeddings")
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
    
    return embeddings_dir
