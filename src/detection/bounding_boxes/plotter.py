import cv2
import numpy as np
from typing import List

from .shapes import Obb, Bbox, find_line_for_word


def plot_obb_on_image(
    image: np.ndarray, obb: Obb, color: tuple = (0, 0, 255)
) -> np.ndarray:
    im_obb = np.zeros((4, 2), dtype=np.int32)
    width = image.shape[0]
    height = image.shape[1]

    for i in range(0, len(obb), 2):
        im_obb[i // 2][0] = round(obb[i] * height) if obb[i] > 0 else 0
        im_obb[i // 2][1] = round(obb[i + 1] * width) if obb[i + 1] > 0 else 0

    im_obb = im_obb.reshape((-1, 1, 2))

    return cv2.polylines(image, [im_obb], True, color, 3)


def plot_obbs_on_image(
    image: np.ndarray, obbs: List[Obb], color: tuple = (0, 0, 255)
) -> np.ndarray:
    plotted_image = image.copy()
    for obb in obbs:
        plotted_image = plot_obb_on_image(image, obb, color)

    return plotted_image


def plot_word_and_line(image: np.ndarray, word: Bbox, lines: list[Obb]) -> np.ndarray:
    line = find_line_for_word(word, lines)

    image = plot_obb_on_image(image, line)
    return plot_obb_on_image(image, word, (255, 0, 0))
