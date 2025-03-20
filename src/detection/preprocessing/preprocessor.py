import os
import numpy as np
import pickle
import cv2
from abc import ABC, abstractmethod


class PreprocessStep(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray: ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class Identity(PreprocessStep):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image


class TextureRemove(PreprocessStep):
    @staticmethod
    def _extract_features(image: np.ndarray) -> np.ndarray:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        features = np.sqrt(sobelx**2 + sobely**2)

        return features

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image - self._extract_features(image)


class GrayScale(PreprocessStep):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


class EdgeDetection(PreprocessStep):
    def __init__(self, t_lower: int = 50, t_upper: int = 100):
        self.t_lower = t_lower
        self.t_upper = t_upper

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.Canny(image, self.t_lower, self.t_upper)


class ImagePreprocessor:
    def __init__(self, name: str = "", methods: list = [Identity()]) -> None:
        """
        Args:
            methods - list of preprocessing functions of type f(np.ndarray) -> np.ndarray
        """
        self._methods = methods
        self._name = name

    def process(self, img: np.ndarray) -> np.ndarray:
        for method in self._methods:
            img = method(img)
        return img

    def save(self, preprocessors_path: str, rewrite_if_exists: bool = False) -> str:
        save_path = os.path.join(preprocessors_path, self._name)
        try:
            os.makedirs(save_path, exist_ok=rewrite_if_exists)
        except OSError:
            raise ValueError(
                f"Preprocessor with name {self._name} already exists. Change the name or set `rewrite_if_exists = True`"
            )

        for method in self._methods:
            method_name = method.__class__.__name__ + ".pkl"
            with open(os.path.join(save_path, method_name), "wb") as f:
                pickle.dump(method, f)

        with open(os.path.join(save_path, "methods.txt"), "w") as f:
            for method in self._methods:
                f.write(method.__class__.__name__ + "\n")

    @staticmethod
    def load(path: str):
        name = os.path.basename(path)
        method_names = []
        with open(os.path.join(path, "methods.txt"), "r") as f:
            for line in f:
                method_names.append(line[:-1])

        methods = []
        for method_name in method_names:
            with open(os.path.join(path, method_name + ".pkl"), "rb") as f:
                methods.append(pickle.load(f))

        return ImagePreprocessor(name, methods)

    def __repr__(self) -> str:
        object_data = f'( Image preprocessor "{self._name}" | '
        object_data += "Methods: "
        object_data += " -> ".join(str(method) for method in self._methods) + " )"
        return object_data
