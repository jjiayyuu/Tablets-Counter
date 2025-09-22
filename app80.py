import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
import cv2
import time
from datetime import datetime
import io

st.set_page_config(page_title="Tablet Counter", layout="wide")

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    try:
        model = YOLO("best80.pt")  # Make sure best50.pt is in the same folder
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
    """Enhanced live pill detection with auto-detection and snapshot capability"""
    
    # Initialize session state variables
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = True
    if 'current_frame_with_boxes' not in st.session_state:
        st.session_state.current_frame_with_boxes = None
    if 'current_tablet_count' not in st.session_state:
        st.session_state.current_tablet_count = 0
    if 'snapshots' not in st.session_state:
        st.session_state.snapshots = []
    
    # Create columns for buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔴 Stop Detection", type="secondary"):
            st.session_state.detection_active = False
            st.rerun()
    
    with col2:
        snapshot_button = st.button("📸 Take Snapshot", type="primary", 
                                   disabled=st.session_state.current_frame_with_boxes is None)
    
    with col3:
        if st.button("🗑️ Clear Snapshots", type="secondary"):
            st.session_state.snapshots = []
            st.rerun()
    
    # Display current detection count
    count_placeholder = st.empty()
    
    # Main video display
    video_placeholder = st.empty()
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam. Please check if your camera is available and not being used by another application.")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        detection_interval = 10  # Detect every 10 frames for better performance
        
        while st.session_state.detection_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break
            
            frame_count += 1
            
            # Run detection every few frames to improve performance
            if frame_count % detection_interval == 0:
                try:
                    # Run YOLO detection
                    results = model(frame, conf=confidence, verbose=False)
                    
                    # Create a copy of frame for drawing
                    display_frame = frame.copy()
                    tablet_count = 0
                    
                    if results[0].boxes is not None:
                        tablet_count = len(results[0].boxes)
                        
                        # Draw bounding boxes and labels
                        for i, box in enumerate(results[0].boxes.xyxy):
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Draw bounding box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw tablet number
                            label = f"Tablet {i+1}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(display_frame, (x1, y1-label_size[1]-10), 
                                        (x1+label_size[0], y1), (0, 255, 0), -1)
                            cv2.putText(display_frame, label, (x1, y1-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Add count display on frame
                    count_text = f"Tablets Detected: {tablet_count}"
                    cv2.rectangle(display_frame, (10, 10), (350, 50), (0, 0, 0), -1)
                    cv2.putText(display_frame, count_text, (20, 35),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Update session state
                    st.session_state.current_tablet_count = tablet_count
                    st.session_state.current_frame_with_boxes = display_frame.copy()
                    
                except Exception as e:
                    st.error(f"Detection error: {str(e)}")
                    display_frame = frame.copy()
            else:
                # Use the last detection result
                display_frame = st.session_state.current_frame_with_boxes if st.session_state.current_frame_with_boxes is not None else frame
            
            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update count display
            count_placeholder.metric("Current Detection", f"{st.session_state.current_tablet_count} tablets")
            
            # Handle snapshot button
            if snapshot_button and st.session_state.current_frame_with_boxes is not None:
                # Convert frame to PIL Image for saving
                snapshot_rgb = cv2.cvtColor(st.session_state.current_frame_with_boxes, cv2.COLOR_BGR2RGB)
                snapshot_pil = Image.fromarray(snapshot_rgb)
                
                # Create snapshot data
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                snapshot_data = {
                    'image': snapshot_pil,
                    'count': st.session_state.current_tablet_count,
                    'timestamp': timestamp
                }
                
                st.session_state.snapshots.append(snapshot_data)
                st.success(f"Snapshot saved! Detected {st.session_state.current_tablet_count} tablets at {timestamp}")
                st.rerun()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.03)
    
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()

# ==================== Display Snapshots ====================
def display_snapshots():
    """Display saved snapshots"""
    if st.session_state.get('snapshots'):
        st.subheader("📷 Saved Snapshots")
        
        for i, snapshot in enumerate(reversed(st.session_state.snapshots)):
            with st.expander(f"Snapshot {len(st.session_state.snapshots)-i} - {snapshot['count']} tablets - {snapshot['timestamp']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.image(snapshot['image'], caption=f"Detected: {snapshot['count']} tablets", use_container_width=True)
                
                with col2:
                    st.metric("Tablets", snapshot['count'])
                    st.text(f"Time: {snapshot['timestamp']}")
                    
                    # Download button for individual snapshot
                    img_buffer = io.BytesIO()
                    snapshot['image'].save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Image",
                        data=img_buffer.getvalue(),
                        file_name=f"tablet_snapshot_{len(st.session_state.snapshots)-i}_{snapshot['timestamp'].replace(':', '-')}.png",
                        mime="image/png"
                    )

# ---------------- STREAMLIT UI ----------------
st.title("💊 Tablet Counter (80)")
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


