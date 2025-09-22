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
def live_detection_interface(model):
    """Live detection interface using Streamlit's camera input with auto-refresh"""
    
    # Initialize session state
    if 'snapshots' not in st.session_state:
        st.session_state.snapshots = []
    if 'live_detection_active' not in st.session_state:
        st.session_state.live_detection_active = False
    if 'detection_interval' not in st.session_state:
        st.session_state.detection_interval = 2  # seconds
    
    st.subheader("🎥 Live Tablet Detection")
    st.info("📌 **How it works**: Enable live detection, then the camera will automatically refresh and detect tablets at set intervals. Click 'Capture & Count' when you see tablets to save a snapshot with bounding boxes.")
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🟢 Start Live Detection", type="primary"):
            st.session_state.live_detection_active = True
            st.rerun()
    
    with col2:
        if st.button("🔴 Stop Live Detection", type="secondary"):
            st.session_state.live_detection_active = False
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Snapshots"):
            st.session_state.snapshots = []
            st.rerun()
    
    # Settings
    st.session_state.detection_interval = st.slider(
        "Auto-refresh interval (seconds)", 
        min_value=1, max_value=10, 
        value=st.session_state.detection_interval
    )
    
    # Live detection area
    if st.session_state.live_detection_active:
        st.success("🔴 **LIVE DETECTION ACTIVE** - Camera will refresh automatically")
        
        # Auto-refresh camera input
        camera_key = f"live_camera_{int(time.time() // st.session_state.detection_interval)}"
        
        camera_image = st.camera_input(
            "📷 Live Camera Feed (Auto-refreshing)", 
            key=camera_key,
            help="Position tablets in view and wait for auto-refresh"
        )
        
        if camera_image is not None:
            # Process the image
            image = Image.open(camera_image)
            
            # Automatic detection
            with st.spinner("🔍 Detecting tablets..."):
                count, boxed_image = model_count_tablets_with_boxes(image, model)
            
            # Display results
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.image(boxed_image, caption=f"🎯 Live Detection: {count} tablets found", use_container_width=True)
            
            with col2:
                st.metric("Tablets Detected", count)
                
                # Capture button
                if st.button("📸 Capture & Save This Detection", type="primary"):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    snapshot_data = {
                        'image': boxed_image,
                        'original_image': image,
                        'count': count,
                        'timestamp': timestamp
                    }
                    st.session_state.snapshots.append(snapshot_data)
                    st.success(f"✅ Snapshot saved! {count} tablets detected at {timestamp}")
                    st.rerun()
                
                # Status indicator
                if count > 0:
                    st.success(f"✅ {count} tablets detected!")
                else:
                    st.info("👀 Move tablets into view...")
        
        # Auto-refresh mechanism
        time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        st.rerun()  # This will refresh the camera input
    
    else:
        st.info("Click 'Start Live Detection' to begin automatic tablet detection")
        
        # Manual camera input when live detection is off
        manual_camera = st.camera_input("📷 Manual Camera (Take photo manually)")
        
        if manual_camera is not None:
            image = Image.open(manual_camera)
            
            if st.button("🔍 Count Tablets in This Photo", type="primary"):
                count, boxed_image = model_count_tablets_with_boxes(image, model)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(boxed_image, caption=f"Detection Result: {count} tablets", use_container_width=True)
                with col2:
                    st.metric("Tablets Found", count)
                    
                    if st.button("💾 Save This Result"):
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        snapshot_data = {
                            'image': boxed_image,
                            'original_image': image,
                            'count': count,
                            'timestamp': timestamp
                        }
                        st.session_state.snapshots.append(snapshot_data)
                        st.success("Snapshot saved!")
                        st.rerun()

# ==================== Display Snapshots ====================
def display_snapshots():
    """Display saved snapshots with enhanced features"""
    if st.session_state.get('snapshots'):
        st.subheader("📷 Saved Detection Results")
        st.write(f"Total snapshots: {len(st.session_state.snapshots)}")
        
        # Summary statistics
        if st.session_state.snapshots:
            total_tablets = sum(snap['count'] for snap in st.session_state.snapshots)
            avg_tablets = total_tablets / len(st.session_state.snapshots)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Snapshots", len(st.session_state.snapshots))
            col2.metric("Total Tablets Counted", total_tablets)
            col3.metric("Average per Snapshot", f"{avg_tablets:.1f}")
        
        # Display snapshots
        for i, snapshot in enumerate(reversed(st.session_state.snapshots)):
            snapshot_num = len(st.session_state.snapshots) - i
            
            with st.expander(f"📸 Snapshot #{snapshot_num} - {snapshot['count']} tablets - {snapshot['timestamp']}", expanded=i==0):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Show both original and detection result
                    tab1, tab2 = st.tabs(["🎯 With Detection", "📷 Original"])
                    
                    with tab1:
                        st.image(snapshot['image'], caption=f"Detected: {snapshot['count']} tablets", use_container_width=True)
                    
                    with tab2:
                        st.image(snapshot['original_image'], caption="Original image", use_container_width=True)
                
                with col2:
                    st.metric("Tablets Detected", snapshot['count'])
                    st.text(f"📅 {snapshot['timestamp']}")
                    
                    # Download buttons
                    for img_type, img_key, label in [
                        ("detection", 'image', "🎯 Download with Boxes"),
                        ("original", 'original_image', "📷 Download Original")
                    ]:
                        img_buffer = io.BytesIO()
                        snapshot[img_key].save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        
                        st.download_button(
                            label=label,
                            data=img_buffer.getvalue(),
                            file_name=f"tablet_{img_type}_{snapshot_num}_{snapshot['timestamp'].replace(':', '-').replace(' ', '_')}.png",
                            mime="image/png",
                            key=f"download_{img_type}_{i}"
                        )
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{i}"):
                        st.session_state.snapshots.remove(snapshot)
                        st.rerun()

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



