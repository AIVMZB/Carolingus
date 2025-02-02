from detection.preprocessing.preprocessor import ImagePreprocessor

from ultralytics import YOLO
from typing import Union
import numpy as np
import cv2
import torch
import os


class YoloWrapper:
    TRAIN_KWARGS = dict(
        batch=2,
        box=10,
        cls=0.2,
        dfl=0.7,
        workers=1,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        translate=0.1,
        scale=0.1,
        fliplr=0.0,
        mosaic=0.0,
        erasing=0.0,
        crop_fraction=0.1,
    )

    def __init__(
        self,
        model_path: str,
        device: Union[torch.device, str],
        preprocessor: Union[ImagePreprocessor, str, None] = None,
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

        if isinstance(preprocessor, ImagePreprocessor):
            self._preprocessor = preprocessor
        elif isinstance(preprocessor, str):
            assert os.path.exists(
                preprocessor
            ), "Provide preprocessor object or its valid path"
            self._preprocessor = ImagePreprocessor.load(preprocessor)
        else:
            self._preprocessor = ImagePreprocessor()

    def train(
        self, data_file: str, epochs: int, img_size: int, angle_aug: float = 0
    ) -> None:
        """
        Trains model using given data
        Args:
            data_file (str): path to yaml file of dataset
            epochs (int): number of epochs to train
            img_size (int): image size
            angle_aug (float): Value for applying rotation augmentation by given angle.
                Useful to train line detection model with angle_aug=3. Defaults to 0.
        """
        self._model.train(
            data=data_file,
            epochs=epochs,
            imgsz=img_size,
            device=self._device,
            degrees=angle_aug,
            **YoloWrapper.TRAIN_KWARGS,
        )

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
        image = self._preprocessor.process(image)

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
