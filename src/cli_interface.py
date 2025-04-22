import os
import click
import torch
import yaml
from detection.models.yolo_wrapper import YoloWrapper
from detection.models.yolo_pipeline import build_line_word_pipeline


@click.group()
def main():
    ...


@main.command()
@click.option("-m", "--model", type=str, default="yolov8m.pt", help="Model name or path to weights path")
@click.option("-c", "--config", type=str, help="Path to .yaml file with training configurations")
def train(model: str, config: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_wrapper = YoloWrapper(model, device)
    yolo_wrapper.train(config)


@main.command()
@click.option("-m", "--model", type=str, default="yolov8m.pt", help="Model name or path to weights path")
@click.option("-i", "--image", type=str, help="Path to image")
@click.option("-p", "--prediction-dir", type=str, default="./predictions", help="Path to prediction dir.")
@click.option("-mc", "--min-conf", type=float, default=0.5, help="Minimum confidence value.")
@click.option("--show", is_flag=True, default=False, help="Set this flag to show the detection results")
def detect(
    model: str,
    image: str,
    prediction_dir: str = "./predictions",
    min_conf: float = 0.5,
    show: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_wrapper = YoloWrapper(model, device)
    yolo_wrapper.inference_image(
        image,
        prediction_dir,
        min_conf,
        show,
        os.path.join(prediction_dir, "boxes.txt")
    )


@main.command()
@click.option("-i", "--image", type=str, help="Image of a document.")
@click.option("-c", "--config", type=str, default="../config/yolo_inference.yaml", help="Processing configurations file.")
@click.option("-s", "--save-dir", type=str, default="../predictions", help="Path, where to save results.")
def pipeline_detect(image: str, config: str, save_dir: str):
    config = yaml.safe_load(open(config))
    pipeline = build_line_word_pipeline(config)
    pipeline.predict(image, save_dir)


if __name__ == "__main__":
    main()
