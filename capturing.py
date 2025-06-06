import streamlit as st
import cv2
import os
from datetime import datetime
from ModelsCode.YOLODetector import ObjectExtractor
from ModelsCode.CascadeClassifier import CascadeTrainer

SAVE_DIR = "data"
POS_DIR = 'data/Positive'
NEG_DIR = 'data/Negative'
NEW_DIR = 'Dataset'
NUM_IMAGES = 10

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)
os.makedirs(NEW_DIR, exist_ok=True)

def main():
    st.title("Capture Images for Cascade Training")

    # Init session state
    if 'images' not in st.session_state:
        st.session_state.images = []
    if 'phase' not in st.session_state:
        st.session_state.phase = 'positive'
    if 'capturing_done' not in st.session_state:
        st.session_state.capturing_done = False
    if 'label' not in st.session_state:
        st.session_state.label = ''

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    # Capture loop
    if not st.session_state.capturing_done:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                stframe.image(frame, channels="BGR", caption=f"{st.session_state.phase.title()} Image {len(st.session_state.images)}/{NUM_IMAGES}")

                if st.button("Capture Photo"):
                    st.session_state.images.append(frame.copy())
                    st.success(f"Captured image {len(st.session_state.images)}")

                if len(st.session_state.images) == NUM_IMAGES:
                    st.session_state.capturing_done = True
                    cap.release()
                    st.success(f"Captured all 10 {st.session_state.phase} images.")
            else:
                st.error("Camera not working.")
        else:
            st.error("Cannot access camera.")

    if st.session_state.capturing_done:
        if st.session_state.phase == 'positive':
            label = st.text_input("Enter a label for positive images:", value=st.session_state.label)
            if label:
                st.session_state.label = label
                if st.button("Save Positive Images"):
                    label_dir = os.path.join(POS_DIR, label)
                    os.makedirs(label_dir, exist_ok=True)
                    for i, img in enumerate(st.session_state.images):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(label_dir, f"{label}_{timestamp}_{i+1}.jpg")
                        cv2.imwrite(filename, img)
                    st.success("Saved positive images.")
                    st.session_state.images = []
                    st.session_state.phase = 'negative'
                    st.session_state.capturing_done = False

        elif st.session_state.phase == 'negative':
            if st.button("Save Negative Images"):
                label_dir = os.path.join(NEG_DIR, st.session_state.label)
                os.makedirs(label_dir, exist_ok=True)
                for i, img in enumerate(st.session_state.images):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(label_dir, f"neg_{timestamp}_{i+1}.jpg")
                    cv2.imwrite(filename, img)
                st.success("Saved negative images.")
                st.session_state.images = []
                st.session_state.phase = 'done'
                st.session_state.capturing_done = False

    # Train
    if st.session_state.phase == 'done':
        if st.button("Train Model"):
            detector = ObjectExtractor(POS_DIR, NEG_DIR, NEW_DIR)
            detector.extract()
            trainer = CascadeTrainer(NEW_DIR, st.session_state.label)
            trainer.train()
            st.success("Model training completed.")
            st.session_state.phase = 'positive'

main()
