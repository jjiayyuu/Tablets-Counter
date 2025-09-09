import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
from ultralytics import YOLO

# Configure Streamlit page
st.set_page_config(page_title="Tablet Counter", layout="wide")

@st.cache_resource
def load_model():
    """Load YOLO model (cached for performance)"""
    try:
        model = YOLO("best50.pt")  # put best.pt in same folder
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def model_count_tablets(image, model):
    """Count tablets using YOLO model"""
    if model is None:
        return 0
    try:
        img_array = np.array(image)
        results = model(img_array)

        tablet_count = 0
        for result in results:
            if result.boxes is not None:
                tablet_count += len(result.boxes)
        return tablet_count
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        return 0

# ---------------- STREAMLIT UI ----------------
st.title("Tablet Counter")
st.write("Upload an image OR use your camera for live scan")

# Load model
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.error("Could not load model. Please check the path and try again.")
    st.stop()
else:
    st.success("Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Camera live scan option
use_camera = st.checkbox("Use Camera for Live Scan")

# ---------- File Upload Mode ----------
if uploaded_file is not None and not use_camera:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Count Tablets", type="primary"):
        with st.spinner("Analyzing image..."):
            count = model_count_tablets(image, model)

        if count > 0:
            st.success(f"Number of tablets detected: **{count}**")
        else:
            st.warning("No tablets detected in the image")

# ---------- Live Camera Mode ----------
elif use_camera:
    st.info("Starting camera... hold still for 5 seconds to scan")

    run = st.button("Start Live Scan")
    if run:
        cap = cv2.VideoCapture(0)
        prev_frame = None
        still_start = None
        result_image = None

        frame_placeholder = st.empty()
        result_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # Detect motion by frame difference
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is None:
                prev_frame = gray
                continue

            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_level = np.sum(thresh)

            if motion_level < 1e6:  # low movement threshold
                if still_start is None:
                    still_start = time.time()
                elif time.time() - still_start >= 5:  # 5 seconds stable
                    result_image = frame_rgb
                    break
            else:
                still_start = None

            prev_frame = gray

        cap.release()

        if result_image is not None:
            st.image(result_image, caption="Captured Frame", use_column_width=True)
            with st.spinner("Analyzing captured frame..."):
                image = Image.fromarray(result_image)
                count = model_count_tablets(image, model)

            if count > 0:
                result_placeholder.success(f"Number of tablets detected: **{count}**")
            else:
                result_placeholder.warning("No tablets detected in the captured frame")
