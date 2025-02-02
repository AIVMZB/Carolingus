import torch
from typing import List
from abc import ABC, abstractmethod

import detection.bounding_boxes.shapes as sh
from detection.bounding_boxes.obb_helper import (
    find_right_bottom,
    find_left_bottom,
    find_right_top,
    find_left_top
)


def find_intersecting_objects(
    boxes: List[sh.Obb], threshold: float = 0.4
) -> List[tuple[int, int]]:
    """
    Finds interseceted boxes by IoU
    Args:
        boxes (List[Bbox, Obb]) - list of boxes
        threshold (float) - IoU threshold
    Returns
        intersecting_objects (List[tuple[int, int]]) - List of pairs containing indexes of intersected objects
    """
    intersecting_objects = []

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if sh.IoU(boxes[i], boxes[j]) >= threshold:
                intersecting_objects.append((i, j))

    return intersecting_objects


def tensor_to_boxes(boxes: torch.Tensor) -> List[sh.Obb]:
    shapes = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        shapes.append(sh.yolo_result_to_obb(box))

    return shapes


class IntersectionResolver(ABC):
    @abstractmethod
    def __call__(
        self, box_1: sh.Obb, conf_1: float, box_2: sh.Obb, conf_2: float
    ) -> sh.Obb:
        pass


class ByConfidenceResolver(IntersectionResolver):
    def __call__(
        self, box_1: sh.Obb, conf_1: float, box_2: sh.Obb, conf_2: float
    ) -> sh.Obb:
        return box_1 if conf_1 > conf_2 else box_2


class UnionResolver(IntersectionResolver):
    def __call__(
        self, box_1: sh.Obb, conf_1: float, box_2: sh.Obb, conf_2: float
    ) -> sh.Obb:
        x_list = [box_1[i] for i in range(0, 8, 2)] + [box_2[i] for i in range(0, 8, 2)]
        y_list = [box_1[i] for i in range(1, 8, 2)] + [box_2[i] for i in range(1, 8, 2)]

        max_x = max(x_list)
        min_x = min(x_list)
        max_y = max(y_list)
        min_y = min(y_list)

        return sh.Obb(min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y)


def build_resolver_by_name(resolver_name: str | None) -> IntersectionResolver | None:
    """
    Build specific IntersectionResolver object based on name. If needed resolver is not found, returns None
    Args:
        resolver_name (str | None) - name of an intersection resolver
    Returns:
        resolver (IntersectionResolver | None)
    """
    if resolver_name == "ByConfResolver":
        from detection.intersect_resolver import ByConfidenceResolver

        return ByConfidenceResolver()
    if resolver_name == "MeanResolver":
        from detection.intersect_resolver import MeanResolver

        return MeanResolver()
    if resolver_name == "UnionResolver":
        from detection.intersect_resolver import UnionResolver

        return UnionResolver()
    else:
        raise ValueError(
            "Choose one the available resolvers: ByConfResolver, MeanResolver, UnionResolver"
        )


class MeanResolver(IntersectionResolver):
    def __call__(
        self, box_1: sh.Obb, conf_1: float, box_2: sh.Obb, conf_2: float
    ) -> sh.Obb:
        box_1 = sh.Obb(
            *find_left_top(box_1),
            *find_right_top(box_1),
            *find_right_bottom(box_1),
            *find_left_bottom(box_1)
        )

        box_2 = sh.Obb(
            *find_left_top(box_2),
            *find_right_top(box_2),
            *find_right_bottom(box_2),
            *find_left_bottom(box_2)
        )

        result_box = [0] * 8
        for i in range(8):
            result_box[i] = (box_1[i] + box_2[i]) / 2

        return sh.Obb(*result_box)


def resolve_intersected_objects(
    boxes: torch.Tensor,
    confs: torch.Tensor,
    threshold: float,
    resolver: IntersectionResolver,
) -> List[sh.Obb]:
    boxes = tensor_to_boxes(boxes)
    confs = confs.to("cpu")
    intersected_boxes = find_intersecting_objects(boxes, threshold)

    for pair in intersected_boxes:
        i = pair[0]
        j = pair[1]

        if boxes[i] is not None and boxes[j] is not None:  ## TODO: Fix it
            chosen_box = resolver(boxes[i], confs[i], boxes[j], confs[j])
            boxes[i] = chosen_box
            boxes[j] = None

    return list(filter(lambda x: x != None, boxes))
