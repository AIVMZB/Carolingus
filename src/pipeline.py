from detection.models import build_line_word_pipeline
from detection.models.yolo_pipeline import LineWordPipelineResult
from recognition.classification.inference import WordClassifier
from recognition.syntax_embedding.inference import SyntaxEncoderInference
from typing import NamedTuple
from pydantic import BaseModel
import yaml


class PipelineConfig(BaseModel):
    line_word_pipeline_config: str
    syntax_encoder_weights: str
    embeddings: str
    classifier_weights: str
    labels: str

    @staticmethod
    def load(config: str) -> "PipelineConfig":
        return PipelineConfig.model_validate(yaml.safe_load(open(config)))


class FullResult(NamedTuple):
    detection_results: LineWordPipelineResult
    words: list[str]


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self._config = config
        self._line_word_pipeline = build_line_word_pipeline(
            yaml.safe_load(open(self._config.line_word_pipeline_config))
        )
        self._syntax_encoder = SyntaxEncoderInference(
            weights=self._config.syntax_encoder_weights,
            embeddings=self._config.embeddings,
        )
        self._word_classifier = WordClassifier(
            self._config.classifier_weights, self._config.labels
        )
        self._softmax_threshold = 0.9

    def process_document(self, image: str) -> FullResult:
        detection_results = self._line_word_pipeline.predict(image)

        cropped_words = detection_results.word_images

        words = []

        for cropped_word in cropped_words:
            word, prob = self._word_classifier.classify(cropped_word)
            if prob < self._softmax_threshold:
                word = self._syntax_encoder.inference(cropped_word, max_words=1)[0]

            words.append(word)

        return FullResult(
            detection_results=detection_results,
            words=words,
        )
