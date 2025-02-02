from .shapes import Obb, Bbox
import numpy as np


def find_left_top(box: Obb) -> tuple[float, float]:
    left_1 = 0
    for i in range(2, 8, 2):
        if box[i] < box[left_1]:
            left_1 = i

    left_2 = 0 if left_1 != 0 else 2
    for i in range(0, 8, 2):
        if box[i] < box[left_2] and i != left_1:
            left_2 = i

    if box[left_1 + 1] < box[left_2 + 1]:
        return box[left_1], box[left_1 + 1]

    return box[left_2], box[left_2 + 1]


def find_left_bottom(box: Obb) -> tuple[float, float]:
    left_1 = 0
    for i in range(0, 8, 2):
        if box[i] < box[left_1]:
            left_1 = i

    left_2 = 0 if left_1 != 0 else 2
    for i in range(0, 8, 2):
        if box[i] < box[left_2] and i != left_1:
            left_2 = i

    if box[left_1 + 1] > box[left_2 + 1]:
        return box[left_1], box[left_1 + 1]

    return box[left_2], box[left_2 + 1]


def find_right_top(box: Obb) -> tuple[int, int]:
    right_1 = 0

    for i in range(2, 8, 2):
        if box[i] > box[right_1]:
            right_1 = i

    right_2 = 0 if right_1 != 0 else 2
    for i in range(0, 8, 2):
        if box[i] > box[right_2] and i != right_1:
            right_2 = i

    if box[right_1 + 1] < box[right_2 + 1]:
        return box[right_1], box[right_1 + 1]

    return box[right_2], box[right_2 + 1]


def find_right_bottom(box: Obb) -> tuple[int, int]:
    right_1 = 0

    for i in range(2, 8, 2):
        if box[i] > box[right_1]:
            right_1 = i

    right_2 = 0 if right_1 != 0 else 2
    for i in range(0, 8, 2):
        if box[i] > box[right_2] and i != right_1:
            right_2 = i

    if box[right_1 + 1] > box[right_2 + 1]:
        return box[right_1], box[right_1 + 1]

    return box[right_2], box[right_2 + 1]


def extend_lines_to_corners(lines: list[Obb]) -> list[Obb]:
    extended_lines = []
    for line in lines:
        extended_lines.append(extend_line_to_corners(line))

    return extended_lines


def extend_line_to_corners(line: Obb) -> Obb:
    left_top = find_left_top(line)
    right_top = find_right_top(line)
    left_bottom = find_left_bottom(line)
    right_bottom = find_right_bottom(line)

    top_direction = np.array([right_top[0] - left_top[0], right_top[1] - left_top[1]])
    bottom_direction = np.array(
        [right_bottom[0] - left_bottom[0], right_bottom[1] - left_bottom[1]]
    )
    avg_direction = (top_direction + bottom_direction) / 2

    x3 = x4 = 0
    x2 = x1 = 1
    y3 = left_bottom[1] - left_bottom[0] * avg_direction[1] / avg_direction[0]
    y4 = left_top[1] - left_top[0] * avg_direction[1] / avg_direction[0]

    y1 = right_top[1] + avg_direction[1] * (1 - right_top[0]) / avg_direction[0]
    y2 = right_bottom[1] + avg_direction[1] * (1 - right_bottom[0]) / avg_direction[0]

    return Obb(x1, y1, x2, y2, x3, y3, x4, y4)


def save_model_results(results: np.ndarray) -> None:
    with open("results.txt", "w") as f:
        for i in range(results.shape[0]):
            box = results[i]
            f.write(
                f"{box[0, 0]} {box[0, 1]} {box[1, 0]} {box[1, 1]} {box[2, 0]} {box[2, 1]} {box[3, 0]} {box[3, 1]}\n"
            )


def obbs_from_file(filename: str = "results.txt") -> list[Obb]:
    obbs = []
    with open(filename, "r") as f:
        for line in f:
            values = [float(value) for value in line.split(" ")]
            obbs.append(Obb(*values))

    return obbs


def model_results_to_obbs(results: np.ndarray) -> list[Obb]:
    obbs = []
    for i in range(results.shape[0]):
        box = results[i]
        obbs.append(
            Obb(
                box[0, 0],
                box[0, 1],
                box[1, 0],
                box[1, 1],
                box[2, 0],
                box[2, 1],
                box[3, 0],
                box[3, 1],
            )
        )

    return obbs


def line_angle(line: Obb) -> float:
    left_top = find_left_top(line)
    right_top = find_right_top(line)
    left_bottom = find_left_bottom(line)
    right_bottom = find_right_bottom(line)

    top_direction = np.array([right_top[0] - left_top[0], right_top[1] - left_top[1]])
    bottom_direction = np.array(
        [right_bottom[0] - left_bottom[0], right_bottom[1] - left_bottom[1]]
    )
    avg_direction = (top_direction + bottom_direction) / 2
    avg_direction = avg_direction / np.sqrt((avg_direction @ avg_direction.T))

    radians = np.arctan(avg_direction[1] / avg_direction[0])

    return np.degrees(radians)


def obb_to_bbox(obb: Obb) -> Bbox:
    cx = (obb.x1 + obb.x2 + obb.x3 + obb.x4) / 4
    cy = (obb.y1 + obb.y2 + obb.y3 + obb.y4) / 4

    left, top = find_left_top(obb)
    right, bottom = find_right_bottom(obb)

    width = right - left
    height = bottom - top

    return Bbox(cx, cy, width, height)
