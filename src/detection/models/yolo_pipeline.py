from detection.bounding_boxes.word_to_lines import crop_line_from_image
from detection.bounding_boxes.obb_helper import extend_lines_to_corners
from detection.bounding_boxes.plotter import plot_obbs_on_image
from detection.intersect_resolver import (
    IntersectionResolver,
    resolve_intersected_objects,
    build_resolver_by_name,
    tensor_to_boxes,
)
from detection.bounding_boxes import Obb, Bbox, obb_to_bbox, obb_to_image_coords
from .base import WordDetector

from ultralytics import YOLO
from typing import List
import numpy as np
import shutil
import torch
import cv2
import os


def obb_center(obb: Obb) -> tuple[float, float]:
    return (
        (obb.x1 + obb.x2 + obb.x3 + obb.x4) / 4,
        (obb.y1 + obb.y2 + obb.y3 + obb.y4) / 4,
    )


class LineWordPipeline(WordDetector):
    def __init__(
        self,
        line_detection_model: str,
        word_detection_model: str,
        line_conf: float,
        word_conf: float,
        line_int_resolver: IntersectionResolver | None,
        word_int_resolver: IntersectionResolver | None,
        rotate_lines: bool = False,
        device: str = "cuda:0",
    ):
        self._device = torch.device(device)
        self._line_model = YOLO(line_detection_model).to(self._device)
        self._word_model = YOLO(word_detection_model).to(self._device)

        self._line_int_resolver = line_int_resolver
        self._word_int_resolver = word_int_resolver

        self._line_conf = line_conf
        self._word_conf = word_conf
        self._rotate_lines = rotate_lines

    def _resolve_line_intersections(
        self, lines: torch.Tensor | np.ndarray, line_confs: torch.Tensor | np.ndarray
    ):
        if self._line_int_resolver is None:
            return tensor_to_boxes(lines)
        resolved_lines = resolve_intersected_objects(
            lines, line_confs, 0.15, self._line_int_resolver
        )
        return extend_lines_to_corners(resolved_lines)

    def _resolve_word_intersections(
        self, words: torch.Tensor | np.ndarray, word_confs: torch.Tensor | np.ndarray
    ):
        if self._word_int_resolver is None:
            return tensor_to_boxes(words)
        return resolve_intersected_objects(
            words, word_confs, 0.1, self._word_int_resolver
        )

    def save_prediction_results(
        self, 
        save_dir: str, 
        image: np.ndarray, 
        lines: List[Obb], 
        words: List[Obb],
        word_images: List[np.ndarray]
    ):
        if os.path.exists(save_dir):
            print(f"The {save_dir} already exists. Do you want to override it? (Y/N) ")
            decision = input(">>> ").lower()
            if decision == "y":
                shutil.rmtree(save_dir)
            else:
                return

        os.makedirs(save_dir)
        lines_dir = os.path.join(save_dir, "lines")
        os.makedirs(lines_dir)

        words_dir = os.path.join(save_dir, "words-detection")
        os.makedirs(words_dir)

        lines_image = plot_obbs_on_image(image.copy(), lines, (255, 0, 0))
        cv2.imwrite(os.path.join(lines_dir, "line_predictions.png"), lines_image)

        for i, line in enumerate(lines):
            line_image = crop_line_from_image(image, line, rotate=self._rotate_lines)
            if line_image.size == 0:
                continue
            line_image = plot_obbs_on_image(line_image, words[i], (255, 0, 0))
            cv2.imwrite(os.path.join(words_dir, f"{i}.png"), line_image)

        cropped_words_dir = os.path.join(save_dir, "cropped-words")
        os.makedirs(cropped_words_dir)

        for i, cropped_word in enumerate(word_images):
            cv2.imwrite(os.path.join(cropped_words_dir, f"{i}.png"), cropped_word)

    def crop_words(self, line_image: np.ndarray, words: List[Obb]) -> List[np.ndarray]:
        words = map(
            lambda word: obb_to_image_coords(
                line_image.shape[1], line_image.shape[0], word
            ),
            words,
        )
        words: list[Bbox] = list(map(obb_to_bbox, words))
        word_images = []
        for word in words:
            cx, cy, w, h = list(map(round, word))
            word_image = line_image[
                cy - h // 2 : cy + h // 2, cx - w // 2: cx + w // 2, :
            ]
            word_images.append(word_image)

        return word_images

    def predict(
        self, image: str | np.ndarray, save_dir: str | None = None
    ) -> np.ndarray:
        line_results = self._line_model.predict([image], conf=self._line_conf)

        lines = line_results[0].obb.xyxyxyxyn
        line_confs = line_results[0].obb.conf

        if isinstance(image, str):
            image = cv2.imread(image)

        lines = self._resolve_line_intersections(lines, line_confs)
        lines = extend_lines_to_corners(lines)
        lines.sort(key=lambda line: obb_center(line)[1])

        words_predictions = []
        word_images = []
        for line in lines:
            line_image = crop_line_from_image(image, line, rotate=self._rotate_lines)

            if line_image.size == 0:
                print("[WARNING] Failed to crop a line. Its size is zero!")
                words_predictions.append([])
                continue

            word_results = self._word_model.predict([line_image], conf=self._word_conf)

            line_words = word_results[0].boxes.xyxyn
            word_confs = word_results[0].boxes.conf

            line_words = self._resolve_word_intersections(line_words, word_confs)
            line_words.sort(key=lambda word: obb_center(word)[0])
            words_predictions.append(line_words)
            word_images += self.crop_words(line_image, line_words)

        if save_dir is not None:
            self.save_prediction_results(save_dir, image, lines, words_predictions, word_images)

        return word_images


def build_line_word_pipeline(config: dict[str, any]):
    """
    Builds LineWordPipeline based on config file
    Args:
        config (dict[str, any]) - parsed yaml file
    Returns:
        line_word_pipeline (LineWordPipeline)
    """
    line_int_resolver = build_resolver_by_name(
        config["line_detection"].get("intersection_resolver")
    )
    word_int_resolver = build_resolver_by_name(
        config["word_detection"].get("intersection_resolver")
    )

    return LineWordPipeline(
        config["line_detection"]["model_path"],
        config["word_detection"]["model_path"],
        config["line_detection"]["min_conf"],
        config["word_detection"]["min_conf"],
        line_int_resolver,
        word_int_resolver,
        config["line_detection"]["rotate_lines"],
    )
