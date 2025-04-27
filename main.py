import streamlit as st
import cv2
import os
import tempfile
import mediapipe as mp
import numpy as np
from PIL import Image

# Define processing function first
def process_img(img, face_detection):

    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Perform face detection
    out = face_detection.process(img_rgb)

    # If faces are detected, blur them
    if out.detections is not None:
        for detection in out.detections:
            # Get the bounding box of each detected face
            bbox = detection.location_data.relative_bounding_box
            x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            # Scale bounding box to actual image dimensions
            x = int(x * W)
            y = int(y * H)
            w = int(w * W)
            h = int(h * H)
            # Blur the detected face region
            img[y: y + h, x: x + w] = cv2.blur(img[y: y + h, x: x + w], (100, 100))
    return img

st.title('Face Blurring App')

# Sidebar to choose input source
st.sidebar.title("Select Input Source")
option = st.sidebar.radio("Choose an input source:", ("Upload Image", "Upload Video", "Use Webcam"))

mp_face_detection = mp.solutions.face_detection

# Initialize face detection model
with mp_face_detection.FaceDetection(0, 0.7) as face_detection:
    # Handling Image Input
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:

            img = Image.open(uploaded_file)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = process_img(img, face_detection)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, caption="Blurred Face Image", use_container_width=True)

    # Handling Video Input
    elif option == "Upload Video":
        uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()

            # Process each frame of the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = process_img(frame, face_detection)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels="RGB")
            cap.release()
            tfile.close()
            os.unlink(tfile.name)

    # Handling Webcam Input
    elif option == "Use Webcam":
        run = st.checkbox('Start Webcam')

        # Create a window to show webcam frames
        FRAME_WINDOW = st.image([])
        webcam = cv2.VideoCapture(0)

        # Process webcam frames if checkbox is selected
        while run:
            ret, frame = webcam.read()
            if not ret:
                st.error("Failed to capture from webcam.")
                break
            frame = process_img(frame, face_detection)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
        else:
            webcam.release()
