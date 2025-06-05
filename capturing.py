import streamlit as st
import cv2
import os
from datetime import datetime

SAVE_DIR = "data"
NUM_IMAGES = 10


os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    st.title("Take 10 Photos")

    if 'images' not in st.session_state:
        st.session_state.images = []
    if 'capturing_done' not in st.session_state:
        st.session_state.capturing_done = False

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    if not st.session_state.capturing_done:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                stframe.image(frame, channels="BGR", caption=f"Captured: {len(st.session_state.images)}/{NUM_IMAGES}")

                if st.button("Capture Photo"):
                    st.session_state.images.append(frame.copy())
                    st.success(f"Captured image {len(st.session_state.images)}")

                if len(st.session_state.images) == NUM_IMAGES:
                    st.session_state.capturing_done = True
                    cap.release()
                    st.success("Captured all 10 images. Please enter a label to save them.")
            else:
                st.error("Camera not working.")
        else:
            st.error("Cannot access camera.")
    else:
        label = st.text_input("Enter a label for the captured images:")
        if label and st.button("Save Images"):
            label_dir = os.path.join(SAVE_DIR, label)
            os.makedirs(label_dir, exist_ok=True)

            for i, img in enumerate(st.session_state.images):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(label_dir, f"{label}_{timestamp}_{i+1}.jpg")
                cv2.imwrite(filename, img)

            st.success(f"Saved 10 images to '{label_dir}'")
            st.session_state.images = []
            st.session_state.capturing_done = False

main()