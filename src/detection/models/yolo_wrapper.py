from ultralytics import YOLO
from typing import Union, Any
import numpy as np
import yaml
import cv2
import torch
import os


class YoloWrapper:
    def __init__(
        self,
        model_path: str,
        device: Union[torch.device, str],
    ):
        """
        Creates instance of YoloWrapper
        Args:
            model_path (str): path to checkpoint of model weights
            device (torch.device | str): name of torch device or torch device itself
            preprocessor (ImagePreprocessor | str | None): instance of ImagePreprocessor or path to its folder. Optional
        """
        if isinstance(device, torch.device):
            self._device = device
        elif isinstance(device, str):
            self._device = torch.device(device)

        self._model = YOLO(model_path).to(self._device)

    def train(
        self, config: Union[str, dict[str, Any]]
    ) -> None:
        if isinstance(config, str):
            config = yaml.safe_load(open(config))

        self._model.train(**config)

    def inference_image(
        self,
        image_path: str,
        prediction_dir: str,
        min_conf: float = 0.5,
        show_plot: bool = True,
        save_boxex_file: str | None = None,
    ):
        """
        Inferences model work in image
        Args:
            image_path (str): Path to image
            prediction_dir (str): Path to directory where to save results
            min_conf (float): Minimal confidence of prediction. Defaults to 0.5
            show_plot (bool): If set to True, shows plot of model's prediction
            save_boxes_file (str | None): Set path to txt file to save predicted bounding boxes. Defauls to None
        """
        image = cv2.imread(image_path)

        result = self._model.predict([image], conf=min_conf)[0]

        if not os.path.exists(prediction_dir):
            os.mkdir(prediction_dir)

        result.plot(
            labels=True,
            probs=False,
            show=show_plot,
            save=True,
            line_width=2,
            filename=os.path.join(prediction_dir, os.path.basename(image_path)),
        )

        if save_boxex_file is not None:
            boxes = result.obb.xyxyxyxy.cpu().numpy().reshape(-1, 8)
            print(boxes.shape)
            np.savetxt(save_boxex_file, boxes, delimiter=",")

        return result

    @property
    def model(self) -> YOLO:
        return self._model

    @model.setter
    def model(self, weiths: str):
        self._model = YOLO(weiths).to(self._device)
