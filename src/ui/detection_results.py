import cv2
import numpy as np
import streamlit as st
from pipeline import FullResult
from detection.bounding_boxes.plotter import plot_obbs_on_image


def show_detection_results(image: np.ndarray, results: FullResult):
    detection_results = results.detection_results

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_with_lines = plot_obbs_on_image(image.copy(), detection_results.lines)

    st.write("Виявлені рядки:")
    st.image(image_with_lines)

    st.divider()

    st.write("Виявлені слова:")

    for i, line_image in enumerate(detection_results.line_images):
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
        line_image = plot_obbs_on_image(line_image.copy(), detection_results.words[i])
        st.image(line_image)
