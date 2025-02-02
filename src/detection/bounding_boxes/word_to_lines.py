from skimage.draw import polygon2mask
from typing import Optional
from enum import Enum, auto
import numpy as np
import random
import shutil
import cv2
import os

from .shapes import Bbox, Obb
from .obb_helper import line_angle
from . import shapes as sh


class CropMaskColor(Enum):
    WHITE = auto()
    BLACK = auto()
    MEAN = auto()


def obb_to_nparray(obb: Obb) -> np.ndarray:
    points = []
    for i in range(0, len(obb), 2):
        points.append([obb[i], obb[i + 1]])

    return np.array(points)


def obb_to_bbox(obb: Obb) -> Bbox:
    points = obb_to_nparray(obb)
    x, y, w, h = cv2.boundingRect(points)
    x = (x + w) / 2
    y = (y + h) / 2

    return Bbox(x, y, w, h)


def obb_to_rec(obb: Obb) -> list:
    """Returns x, y, w, h"""
    x = min(obb.x1, obb.x2, obb.x3, obb.x4)
    y = min(obb.y1, obb.y2, obb.y3, obb.y4)
    w = max(obb.x1, obb.x2, obb.x3, obb.x4) - x
    h = max(obb.y1, obb.y2, obb.y3, obb.y4) - y

    return [x, y, w, h]


def mean_image_color(image: np.ndarray) -> np.ndarray:
    r_mean = np.mean(image[:, :, 0])
    g_mean = np.mean(image[:, :, 1])
    b_mean = np.mean(image[:, :, 2])

    return np.array([r_mean, g_mean, b_mean])


def crop_line_from_image(
    image: np.ndarray,
    line: Obb,
    rotate: bool = True,
    mask_image: bool = True,
    mask_color: CropMaskColor = CropMaskColor.BLACK,
) -> np.ndarray:
    line = sh.obb_to_image_coords(image.shape[1], image.shape[0], line)
    np_line = obb_to_nparray(line)
    for i in range(np_line.shape[0]):
        np_line[i][0], np_line[i][1] = np_line[i][1], np_line[i][0]

    line_mask = polygon2mask(image.shape[:2], np_line)

    line_mask = np.stack([line_mask, line_mask, line_mask], axis=-1)
    x, y, w, h = obb_to_rec(line)

    cropped = image[y : y + h, x : x + w].copy()
    if any(dim == 0 for dim in cropped.shape):
        return cropped

    if mask_image:
        cropped_mask = line_mask[y : y + h, x : x + w].copy()
        cropped = cropped * cropped_mask

    if rotate:
        rotation_matrix = cv2.getRotationMatrix2D(
            (cropped.shape[1] // 2, cropped.shape[0] // 2), line_angle(line), scale=1
        )
        cropped = cv2.warpAffine(
            cropped,
            rotation_matrix,
            (cropped.shape[1], cropped.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
        )

    if mask_color == CropMaskColor.WHITE:
        cropped = np.vectorize(lambda x: 255 if x == 0 else x)(cropped)
    elif mask_color == CropMaskColor.MEAN:
        mean_color = mean_image_color(image)
        mask = np.all(cropped == 0, axis=-1)
        cropped[mask] = mean_color

    return cropped


def bbox_centers_to_numpy(words: list[Bbox]) -> np.ndarray:
    centers = []
    for word in words:
        centers.append([word.cx, word.cy])

    return np.array(centers)


def rotate_words_in_line(
    rotation_angle: float, img_shape: tuple[int, int], words: list[Bbox]
) -> list[Bbox]:
    """
    Rotate words' centers around to compensate line rotation.
    Args:
        rotation_angle (float): Line angle in degrees.
        img_shape (tuple[int, int]): Shape of cropped line image. Used to define center and make rotation accurate.
        words (list[Bbox]): List of bboxes containing words' coordinates belonging to rotated line.
    Returns:
        list[Bbox]: List of bboxes with coordinates of rotated words.
    """
    img_shape = np.array(img_shape)
    rotation_angle = np.radians(rotation_angle)
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )

    rotation_center = img_shape / 2

    centers = bbox_centers_to_numpy(words) * img_shape
    centers -= rotation_center
    rotated_centers = centers @ rotation_matrix.T  # (A @ B).T = B.T @ A.T
    rotated_centers += rotation_center
    rotated_centers /= img_shape

    rotated_words = []
    for i in range(rotated_centers.shape[0]):
        center = rotated_centers[i]
        bbox = words[i]
        rotated_bbox = sh.Bbox(center[0], center[1], bbox.w, bbox.h)
        cropped_bbox = crop_bbox(rotated_bbox)
        rotated_words.append(cropped_bbox)

    return rotated_words


def crop_bbox(bbox: Bbox) -> Bbox:
    x1 = bbox.cx - bbox.w / 2
    y1 = bbox.cy - bbox.h / 2
    x2 = bbox.cx + bbox.w / 2
    y2 = bbox.cy + bbox.h / 2

    rec = [x1, y1, x2, y2]
    for i in range(4):
        if rec[i] > 1:
            rec[i] = 1
        elif rec[i] < 0:
            rec[i] = 0

    cx = (rec[0] + rec[2]) / 2
    cy = (rec[1] + rec[3]) / 2
    w = rec[2] - rec[0]
    h = rec[3] - rec[1]

    return Bbox(cx, cy, w, h)


def bbox_to_line_space(bbox: Bbox, line: list, crop=True) -> Bbox:
    """
    Transforms the bounding box coordinates to line space according to the provided line.

    Args:
        bbox (Bbox): The bounding box coordinates.
        line (list): The line in form of rectangle (x, y, w, h).

    Returns:
        Bbox: The transformed bounding box in line space.
    """
    bbox_center = np.array([bbox.cx, bbox.cy])
    bbox_size = np.array([bbox.w, bbox.h])
    line_point = np.array([line[0], line[1]])
    line_size = np.array([line[2], line[3]])

    line_bbox_center = (bbox_center - line_point) / line_size
    line_bbox_size = bbox_size / line_size

    if crop:
        return crop_bbox(Bbox(*line_bbox_center, *line_bbox_size))

    return Bbox(*line_bbox_center, *line_bbox_size)


def words_to_line_space(words: list[Bbox], line: Obb) -> list[Bbox]:
    line_words = []
    line_rec = obb_to_rec(line)
    for word in words:
        word_in_line = bbox_to_line_space(word, line_rec)
        line_words.append(word_in_line)

    return line_words


def write_bboxes_to_file(bboxes: list[Bbox], filename: str, class_name: str) -> None:
    with open(filename, "w") as file:
        for bbox in bboxes:
            line = class_name + " " + " ".join(map(str, bbox)) + "\n"
            file.write(line)


def create_lines_dataset_from_image(
    image_path: str,
    lines: list[Obb],
    words: list[Bbox],
    output_dir: str,
    rotate: bool,
    mask_image: bool,
    mask_color: CropMaskColor,
    word_class: str = "0",
) -> None:
    if not os.path.exists(image_path):
        raise ValueError(f"There is no such an image {image_path}")

    image_name = os.path.basename(image_path)

    image = cv2.imread(image_path)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    line_to_words = sh.map_words_to_lines(words, lines, image)
    for line_idx in line_to_words:
        line = lines[line_idx]

        line_image = crop_line_from_image(image, line, rotate, mask_image, mask_color)
        line_image_path = os.path.join(output_dir, f"{image_name[:-4]}_{line_idx}.jpg")

        try:
            cv2.imwrite(line_image_path, line_image)
        except:
            print(f"\t[WARNING] Failed to save line image with index {line_idx}")

        line_words = [words[idx] for idx in line_to_words[line_idx]]
        line_words = words_to_line_space(line_words, line)
        if rotate and len(line_words):
            rotation_angle = -line_angle(line)
            line_words = rotate_words_in_line(
                rotation_angle, (line_image.shape[1], line_image.shape[0]), line_words
            )

        write_bboxes_to_file(
            line_words,
            os.path.join(output_dir, f"{image_name[:-4]}_{line_idx}.txt"),
            word_class,
        )


def find_line_label(word_label_path: str, line_labels: list[str]) -> Optional[str]:
    word_label = os.path.basename(word_label_path)
    for line_label in line_labels:
        if word_label in line_label:
            return line_label


def create_dataset_of_lines(
    words_dataset_path: str,
    words_ids: list,
    lines_dataset_path: str,
    lines_ids: list,
    output_dataset_path: str,
    rotate: bool,
    mask_image: bool,
    mask_color: CropMaskColor,
):
    os.makedirs(output_dataset_path, exist_ok=True)
    word_label_paths: list[str] = []
    line_label_paths: list[str] = []

    word_subdirs = [
        subdir for subdir in os.scandir(words_dataset_path) if subdir.is_dir()
    ]
    for word_subdir in word_subdirs:
        for file in os.listdir(word_subdir.path):
            if file.endswith(".txt"):
                word_label_paths.append(os.path.join(word_subdir.path, file))

    line_subdirs = [
        subdir for subdir in os.scandir(lines_dataset_path) if subdir.is_dir()
    ]
    for line_subdir in line_subdirs:
        for file in os.listdir(line_subdir.path):
            if file.endswith(".txt"):
                line_label_paths.append(os.path.join(line_subdir.path, file))

    for word_label in word_label_paths:
        image_path = word_label.split(".")[0] + ".jpg"
        word_bboxes = sh.read_shapes(word_label, sh.to_bbox, words_ids)
        line_label = find_line_label(word_label, line_label_paths)
        if line_label is None:
            print(
                f"[WARNING] - {os.path.basename(word_label)} was not found in line labels"
            )
            continue

        print(f"[INFO] processing {os.path.basename(word_label).split('.')[0]}")

        line_obbs = sh.read_shapes(line_label, sh.to_obb, lines_ids)

        create_lines_dataset_from_image(
            image_path,
            line_obbs,
            word_bboxes,
            output_dataset_path,
            rotate,
            mask_image,
            mask_color,
        )


def split_dataset(dataset_path: str, output_path: str):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(os.path.join(output_path, "train")):
        os.mkdir(os.path.join(output_path, "train"))

    if not os.path.exists(os.path.join(output_path, "validation")):
        os.mkdir(os.path.join(output_path, "validation"))

    for file in os.listdir(dataset_path):
        if file.endswith(".jpg"):
            if random.random() < 0.8:
                shutil.move(
                    os.path.join(dataset_path, file),
                    os.path.join(output_path, "train", file),
                )
                shutil.move(
                    os.path.join(dataset_path, file[:-4] + ".txt"),
                    os.path.join(output_path, "train", file[:-4] + ".txt"),
                )
            else:
                shutil.move(
                    os.path.join(dataset_path, file),
                    os.path.join(output_path, "validation", file),
                )
                shutil.move(
                    os.path.join(dataset_path, file[:-4] + ".txt"),
                    os.path.join(output_path, "validation", file[:-4] + ".txt"),
                )
