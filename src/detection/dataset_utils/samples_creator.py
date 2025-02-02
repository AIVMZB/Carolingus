from detection.bounding_boxes.shapes import (
    to_bbox,
    map_words_to_lines,
    sort_words_by_lines,
    Bbox,
)
from PIL import Image
import numpy as np
import os


def process_lines_and_terms(words, lines, image, file_path):
    line_to_words = map_words_to_lines(words, lines, image)
    sorted_line_to_words = sort_words_by_lines(line_to_words, words)

    with open(file_path, "r", encoding="utf-8") as file:
        terms = file.read().split()

    return sorted_line_to_words, terms


def assign_terms_to_words(sorted_line_to_words, words, terms, image, output_folder):
    number_to_term = {}
    term_index = 0

    for key, number_list in sorted_line_to_words.items():
        for number in number_list:
            if term_index < len(terms):
                term = terms[term_index]
                number_to_term[number] = term
                term_index += 1

                word = words[number]
                filename = f"{number}-{term}"

                crop_image(image, to_bbox(word), output_folder, filename)


def crop_image(image: np.ndarray, crop_params: Bbox, output_folder: str, filename: str):
    image = Image.fromarray(image)

    x, y, w, h = crop_params

    width = int(w * image.width)
    height = int(h * image.height)

    left = int(x * image.width - width / 2)
    top = int(y * image.height - height / 2)
    right = left + width
    bottom = top + height

    cropped_image = image.crop((left, top, right, bottom))

    output_path = os.path.join(output_folder, f"{filename}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cropped_image.save(output_path)
