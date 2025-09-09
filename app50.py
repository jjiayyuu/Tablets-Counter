import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
import cv2
import time

st.set_page_config(page_title="Tablet Counter", layout="wide")

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    try:
        model = YOLO("best50.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ==================== Count Image ====================
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

# ==================== Live Webcam Detection ====================
def live_pill_detection_streamlit(model, confidence=0.45):
    """Live pill detection streamed inside Streamlit"""
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence, verbose=False)
        pill_count = len(results[0].boxes) if results[0].boxes is not None else 0

        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{i+1}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Pills: {pill_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


        # Convert BGR to RGB for StreamLit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        # Stop live feed when user clicks 'STOP' button
        if st.session_state.get("stop_live", False):
            break

        time.sleep(0.03)
        
    cap.release()   

# ---------------- STREAMLIT UI ----------------
st.title("💊 Tablet Counter")
st.write("1. Upload an image 🖼️")
st.write("2. Use camera 📸 to take an image")
st.write("3. Live webcam 🎥 detection")

# Load model
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.stop()

# Mode Selection
mode = st.radio("Select Mode: ", ["Upload Image", "Camera Snapshot", "Live Webcam"])

# ========== Upload Image ==========
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if st.button("Count Tablets from File", type="primary"):
            count, boxed_image = model_count_tablets_with_boxes(image, model)
            st.image(boxed_image, caption=f"Detected Tablets: {count}", use_container_width=True)
            if count > 0:
                st.success(f"Number of tablets detected: {count}")
            else:
                st.warning("No tablets detected.")

# ========== Camera Snapshot ==========
elif mode == "Camera Snapshot":
    camera_file = st.camera_input("Take a photo with your camera")
    if camera_file is not None:
        image = Image.open(camera_file)
        if st.button("Count Tablets from Camera", type="primary"):
            count, boxed_image = model_count_tablets_with_boxes(image, model)
            st.image(boxed_image, caption=f"Detected Tablets: {count}", use_container_width=True)
            if count > 0:
                st.success(f"Number of tablets detected: {count}")
            else:
                st.warning("No tablets detected.")

# ========== Live Webcam ==========
elif mode == "Live Webcam":
    if st.button("Start Live Detection", type="primary"):
        st.session_state["stop_live"] = False
        live_pill_detection_streamlit(model)

    if st.button("Stop Live Detection", type="secondary"):
        st.session_state["stop_live"] = True
