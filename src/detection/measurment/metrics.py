from typing import List, Union

from detection.bounding_boxes import Bbox, Obb, IoU, to_obb


IOU_THRESHOLD = 0.5


def get_tp_fp(
    predictions: List[Union[Bbox, Obb]],
    groundtruths: List[Union[Bbox, Obb]],
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[float, float]:
    predictions = list(map(to_obb, predictions))
    groundtruths = list(map(to_obb, groundtruths))

    matched_groundtruths = set()
    true_positives = 0
    false_positives = 0

    for prediction in predictions:
        found = False
        for idx, groundtruth in enumerate(groundtruths):
            if (
                idx not in matched_groundtruths
                and IoU(prediction, groundtruth) > iou_threshold
            ):
                true_positives += 1
                matched_groundtruths.add(idx)
                found = True
                break

        if not found:
            false_positives += 1

    return true_positives, false_positives


def get_tp_fn(
    predictions: List[Union[Bbox, Obb]],
    groundtruths: List[Union[Bbox, Obb]],
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[float, float]:
    predictions = list(map(to_obb, predictions))
    groundtruths = list(map(to_obb, groundtruths))

    matched_groundtruths = set()
    true_positives = 0

    for prediction in predictions:
        for idx, groundtruth in enumerate(groundtruths):
            if (
                idx not in matched_groundtruths
                and IoU(prediction, groundtruth) > iou_threshold
            ):
                true_positives += 1
                matched_groundtruths.add(idx)
                break

    false_negatives = len(groundtruths) - len(matched_groundtruths)

    return true_positives, false_negatives


def detection_precision(
    predictions: List[Union[Bbox, Obb]],
    groundtruths: List[Union[Bbox, Obb]],
    iou_threshold: float = IOU_THRESHOLD,
):
    true_positives, false_positives = get_tp_fp(
        predictions, groundtruths, iou_threshold
    )

    return (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )


def detection_recall(
    predictions: List[Union[Bbox, Obb]],
    groundtruths: List[Union[Bbox, Obb]],
    iou_threshold: float = IOU_THRESHOLD,
):
    true_positives, false_negatives = get_tp_fn(
        predictions, groundtruths, iou_threshold
    )

    return (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )


def united_detection_precision(
    predictions_set: List[List[Union[Bbox, Obb]]],
    groundtruths_set: List[List[Union[Bbox, Obb]]],
    iou_threshold: float = IOU_THRESHOLD,
):
    assert len(predictions_set) == len(
        groundtruths_set
    ), "The predictions_set and groundtruths_set must be of the same length"
    true_positives = 0
    false_positives = 0

    for predictions, groundtruths in zip(predictions_set, groundtruths_set):
        tp, fp = get_tp_fp(predictions, groundtruths, iou_threshold)
        true_positives += tp
        false_positives += fp

    return (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )


def united_detection_recall(
    predictions_set: List[List[Union[Bbox, Obb]]],
    groundtruths_set: List[List[Union[Bbox, Obb]]],
    iou_threshold: float = IOU_THRESHOLD,
):
    assert len(predictions_set) == len(
        groundtruths_set
    ), "The predictions_set and groundtruths_set must be of the same length"
    true_positives = 0
    false_negatives = 0

    for predictions, groundtruths in zip(predictions_set, groundtruths_set):
        tp, fn = get_tp_fn(predictions, groundtruths, iou_threshold)
        true_positives += tp
        false_negatives += fn

    return (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
