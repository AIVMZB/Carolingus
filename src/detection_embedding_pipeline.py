from detection.models.yolo_pipeline import build_line_word_pipeline
from recognition.syntax_embedding.inference import SyntaxEncoderInference
from argparse import ArgumentParser
from typing import Union
import yaml
import cv2
import os


class DetectionEmbeddingPipeline:
    def __init__(self, config: Union[str, dict]):
        if isinstance(config, str):
            config = yaml.safe_load(open(config))
        self.config = config

        self.detector = build_line_word_pipeline(self.config["detection_config"])
        self.embedding_model = SyntaxEncoderInference(
            self.config["embedding_model_weights"], self.config["embeddings"]
        )
        self.prediction_dir = self.config["prediction_dir"]

    def transcribe(self, document: str, max_words: int = 1):
        word_images = self.detector.predict(document, save_dir="predictions")

        # TODO: creation of prediction dir must be in this class too
        rec_dir = os.path.join(self.prediction_dir, "recognitions")
        os.makedirs(rec_dir)

        for i, word_image in enumerate(word_images):
            words = self.embedding_model.inference(word_image, max_words)
            name = f"{i}-{'_'.join(words)}.png"

            cv2.imwrite(os.path.join(rec_dir, name), word_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--document", type=str, help="document to be transcribed")
    parser.add_argument(
        "-c",
        "--config",
        default="config/detection-embedding-pipeline-config.yaml",
        help="pipeline config",
    )
    args = parser.parse_args()

    pipeline = DetectionEmbeddingPipeline(args.config)
    pipeline.transcribe(args.document, max_words=2)
