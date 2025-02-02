from detection.models import WordDetector
import numpy as np


def measure_inference(
    model: WordDetector, image: str | np.ndarray, gt: str | np.ndarray
): ...
