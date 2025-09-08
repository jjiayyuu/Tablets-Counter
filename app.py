import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Configure Streamlit page
st.set_page_config(page_title="Tablet Counter", layout="wide")

@st.cache_resource
def load_model():
    """Load YOLO model (cached for performance)"""
    try:
        # Load your trained YOLO model
        model = YOLO('/content/drive/MyDrive/Model (GPU)/runs/detect/train/weights/best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def model_count_tablets(image, model):
    """Count tablets using YOLO model"""
    if model is None:
        return 0
    
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run YOLO inference
        results = model(img_array)
        
        # Count detections
        tablet_count = 0
        for result in results:
            if result.boxes is not None:
                tablet_count += len(result.boxes)
        
        return tablet_count
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        return 0

st.title("Tablet Counter")
st.write("Upload an image to count the number of tablets using YOLO")

# Load model
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.error("Could not load model. Please check the path and try again.")
    st.stop()
else:
    st.success("Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing tablets to count"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.write("### Analysis")

        if st.button("Count Tablets", type="primary"):
            with st.spinner("Analyzing image..."):
                count = model_count_tablets(image, model)

            if count > 0:
                st.success(f"Number of tablets detected: **{count}**")
            else:
                st.warning("No tablets detected in the image")

            st.info("Tip: For best results, ensure good lighting and clear tablet visibility")
