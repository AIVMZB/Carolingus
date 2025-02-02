from abc import ABC, abstractmethod
from typing import List
import numpy as np


class WordDetector(ABC):
    @abstractmethod
    def predict(self, image: str | np.ndarray, save_dir: str | None = None) -> List[np.ndarray]:
        """
        Returns sequence of detected words in right order
        Args:
            image (str | numpy array): Image path or numpy array
            save_dir (str | None): Directory path to save results
        Returns:
            List[Bbox | Obb]: detected words in right order
        """
        raise NotImplementedError()
