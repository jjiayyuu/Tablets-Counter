import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import time

st.set_page_config(page_title="Tablet Counter", layout="wide")

@st.cache_resource
def load_model():
    try:
        model = YOLO("best50.pt")  # make sure best.pt is in same folder
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def model_count_tablets(image, model):
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

# ---------------- UI ----------------
st.title("Tablet Counter")
st.write("Upload an image OR use your camera for a live scan")

# Load model
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Camera option
use_camera = st.checkbox("Use Camera for Live Scan")

if uploaded_file is not None and not use_camera:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Count Tablets", type="primary"):
        count = model_count_tablets(image, model)
        st.success(f"Number of tablets detected: {count}")

elif use_camera:
    st.info("Click 'Take Photo' twice. If the two photos are similar (no movement), detection will run.")

    img1 = st.camera_input("Take first photo")
    img2 = st.camera_input("Take second photo (after 5s pause)")

    if img1 and img2:
        # Compare two images
        im1 = np.array(Image.open(img1).convert("L"))
        im2 = np.array(Image.open(img2).convert("L"))

        diff = np.mean(np.abs(im1.astype("float") - im2.astype("float")))

        if diff < 5:  # images almost the same → no movement
            st.success("No movement detected, running tablet count...")
            image = Image.open(img2)
            count = model_count_tablets(image, model)
            st.success(f"Number of tablets detected: {count}")
        else:
            st.warning("Movement detected between snapshots. Please hold still and try again.")
