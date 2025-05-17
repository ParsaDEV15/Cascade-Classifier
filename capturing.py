import streamlit as st
import cv2
import os
from datetime import datetime
import numpy as np


SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)


def capture_image():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        stframe.image(frame, channels="BGR")

        if st.button("Capture"):
            label = st.text_input("Label:", value="cat")
            if label:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                label_dir = os.path.join(SAVE_DIR, label)
                os.makedirs(label_dir, exist_ok=True)
                filename = os.path.join(label_dir, f"{label}_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                st.success(f"Saved {filename}")
                break

    cap.release()

st.title("Live Object Capture for Training")
capture_image()