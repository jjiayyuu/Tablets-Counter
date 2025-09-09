import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from PIL import ImageDraw

st.set_page_config(page_title="Tablet Counter", layout="wide")

@st.cache_resource
def load_model():
    """Load YOLO model (cached for performance)"""
    try:
        model = YOLO("best50.pt")  # make sure best.pt is in same folder
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def model_count_tablets(image, model):
    """Run YOLO detection and count tablets"""
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

def model_count_tablets_with_boxes(image, model):
    """Run YOLO detection, count tablets, and draw bounding boxes"""
    if model is None:
        return 0, image

    try:
        img_array = np.array(image)
        results = model(img_array)

        tablet_count = 0
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        for result in results:
            if result.boxes is not None:
                tablet_count += len(result.boxes)
                for box in result.boxes.xyxy:  # xyxy = [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, box)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        return tablet_count, draw_image
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        return 0, image

# ---------------- STREAMLIT UI ----------------
st.title("Tablet Counter")
st.write("Upload an image OR use your camera to count tablets")

# Load model
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Camera option
use_camera = st.checkbox("Use Camera")

# ---------- File Upload Mode ----------
if uploaded_file is not None and not use_camera:
    image = Image.open(uploaded_file)
    st.image(boxed_image, caption=f"Detected Tablets: {count}", use_container_width=True)

    if st.button("Count Tablets from File", type="primary"):
        count, boxed_image = model_count_tablets_with_boxes(image, model)
        if count > 0:
            st.success(f"Number of tablets detected: {count}")
        else:
            st.warning("No tablets detected")

# ---------- Camera Mode ----------
elif use_camera:
    camera_file = st.camera_input("Take a photo with your camera")

    if camera_file is not None:
        image = Image.open(camera_file)
        st.image(boxed_image, caption=f"Detected Tablets: {count}", use_container_width=True)

        if st.button("Count Tablets from Camera", type="primary"):
            count, boxed_image = model_count_tablets_with_boxes(image, model)
            if count > 0:
                st.success(f"Number of tablets detected: {count}")
            else:
                st.warning("No tablets detected")
