import cv2
import yaml
import numpy as np
import streamlit as st
from PIL import Image
from detection.models.yolo_pipeline import build_line_word_pipeline
from detection.bounding_boxes.plotter import plot_obbs_on_image


def ui():
    pipeline = build_line_word_pipeline(
        yaml.safe_load(open("../config/yolo_inference.yaml"))
    )

    st.title("Вітаю у вашому ШІ асистенті!")

    image = st.file_uploader(
        "Завантажте зображення документу", type=["png", "jpeg", "jpg"]
    )
    if image is not None:
        image = np.asarray(Image.open(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pipeline.predict(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_with_lines = plot_obbs_on_image(image.copy(), results.lines)

        st.write("Виявлені рядки:")
        st.image(image_with_lines)

        st.divider()

        st.write("Виявлені слова:")

        for i, line_image in enumerate(results.line_images):
            line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
            line_image = plot_obbs_on_image(line_image.copy(), results.words[i])
            st.image(line_image)


if __name__ == "__main__":
    ui()
