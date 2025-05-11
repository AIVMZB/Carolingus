import cv2
import streamlit as st
from pipeline import FullResult


IMG_HEIGHT = 100


def show_recognition_results(full_result: FullResult):
    columns = st.columns(3)

    for i, word in enumerate(full_result.words):
        word_image = full_result.detection_results.word_images[i]

        column_idx = i % len(columns)

        column = columns[column_idx]
        image = cv2.cvtColor(word_image, cv2.COLOR_BGR2RGB)

        ratio = IMG_HEIGHT / image.shape[0]

        with column:
            st.markdown(
                f"""
<div style="text-align: left; white-space: nowrap; overflow: hidden;
             text-overflow: ellipsis; width: 100px; margin-top: 5px;">
    <span style="font-size: 14px;">{word}</span>
</div>
""",
                unsafe_allow_html=True,
            )
            st.image(image, width=int(image.shape[1] * ratio))
