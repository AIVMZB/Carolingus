import cv2
import numpy as np
from PIL import Image
import streamlit as st
from pipeline import Pipeline, PipelineConfig
from .detection_results import show_detection_results
from .recognition_results import show_recognition_results


def show_page():
    pipeline = Pipeline(PipelineConfig.load("config/full_pipeline.yaml"))

    st.title("Вітаю у вашому ШІ асистенті!")

    image = st.file_uploader(
        "Завантажте зображення документу", type=["png", "jpeg", "jpg"]
    )
    if image is not None:
        image = np.asarray(Image.open(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pipeline.process_document(image)

        show_detection_results(image, results)
        show_recognition_results(results)
