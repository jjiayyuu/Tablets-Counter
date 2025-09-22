import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import io

st.set_page_config(page_title="Tablet Counter", layout="wide")

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    try:
        model = YOLO("best80.pt")  # Make sure best80.pt is in the same folder
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
        results = model(img_array, conf=0.45, verbose=False)
        tablet_count = 0
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        for result in results:
            if result.boxes is not None:
                tablet_count += len(result.boxes)
                for i, box in enumerate(result.boxes.xyxy):  # xyxy = [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, box)
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    # Add tablet number
                    draw.text((x1, y1-20), f"Tablet {i+1}", fill="red")
        
        return tablet_count, draw_image
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        return 0, image

# ==================== Live Detection with Streamlit Camera ====================
def live_detection_interface(model):
    """Live detection interface with continuous monitoring"""
    
    # Initialize session state
    if 'snapshots' not in st.session_state:
        st.session_state.snapshots = []
    if 'live_mode_active' not in st.session_state:
        st.session_state.live_mode_active = False
    if 'detection_counter' not in st.session_state:
        st.session_state.detection_counter = 0
    if 'last_detection_time' not in st.session_state:
        st.session_state.last_detection_time = 0
    
    st.subheader("🎥 Live Tablet Detection")
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🟢 Start Live Mode", type="primary"):
            st.session_state.live_mode_active = True
            st.session_state.detection_counter = 0
            st.rerun()
    
    with col2:
        if st.button("🔴 Stop Live Mode", type="secondary"):
            st.session_state.live_mode_active = False
            st.rerun()
    
    with col3:
        refresh_rate = st.selectbox("📊 Refresh Rate", 
                                   options=[0.5, 1, 2, 3], 
                                   index=1, 
                                   format_func=lambda x: f"{x}s")
    
    with col4:
        confidence = st.slider("🎯 Confidence", 0.1, 1.0, 0.45, 0.05)
    
    # Status indicator
    if st.session_state.live_mode_active:
        st.success("🔴 **LIVE DETECTION ACTIVE** - Camera monitoring tablets continuously!")
        
        # Create a unique key that changes over time for auto-refresh
        current_time = time.time()
        if current_time - st.session_state.last_detection_time >= refresh_rate:
            st.session_state.detection_counter += 1
            st.session_state.last_detection_time = current_time
        
        camera_key = f"live_camera_{st.session_state.detection_counter}"
        
        # Auto-refreshing camera with detection
        camera_image = st.camera_input(
            f"🔄 Live Feed (Frame #{st.session_state.detection_counter}) - Auto-refreshing every {refresh_rate}s",
            key=camera_key,
            help="Keep tablets in view - detection happens automatically!"
        )
        
        if camera_image is not None:
            # Process immediately
            image = Image.open(camera_image)
            
            # Run detection
            img_array = np.array(image)
            results = model(img_array, conf=confidence, verbose=False)
            
            tablet_count = 0
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            # Draw bounding boxes with better visibility
            for result in results:
                if result.boxes is not None:
                    tablet_count += len(result.boxes)
                    for i, box in enumerate(result.boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        # Draw thick bounding box
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
                        # Draw tablet number with background
                        text = f"#{i+1}"
                        # Create background for text
                        bbox = draw.textbbox((x1, y1-30), text)
                        draw.rectangle([bbox[0]-5, bbox[1]-3, bbox[2]+5, bbox[3]+3], fill="red")
                        draw.text((x1, y1-30), text, fill="white", stroke_width=2, stroke_fill="black")
            
            # Add detection info overlay on image
            draw.rectangle([10, 10, 300, 80], fill="black", outline="red", width=2)
            draw.text((20, 20), f"LIVE DETECTION", fill="red", stroke_width=1, stroke_fill="white")
            draw.text((20, 40), f"Tablets Found: {tablet_count}", fill="white")
            draw.text((20, 60), f"Confidence: {confidence}", fill="white")
            
            # Display results in columns
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.image(draw_image, 
                        caption=f"🔴 LIVE: {tablet_count} tablets detected (Frame #{st.session_state.detection_counter})", 
                        use_container_width=True)
            
            with col2:
                # Real-time metrics
                st.metric("🎯 Live Count", tablet_count)
                st.metric("📊 Frame #", st.session_state.detection_counter)
                
                # Live status
                if tablet_count > 0:
                    st.success(f"✅ {tablet_count} detected!")
                    
                    # Quick save button
                    if st.button("💾 Quick Save", type="primary", key=f"save_{st.session_state.detection_counter}"):
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        snapshot_data = {
                            'image': draw_image,
                            'original_image': image,
                            'count': tablet_count,
                            'timestamp': timestamp
                        }
                        st.session_state.snapshots.append(snapshot_data)
                        st.success("✅ Saved!")
                        st.balloons()
                else:
                    st.info("👀 Scanning...")
                
                # Show detection history
                if len(st.session_state.snapshots) > 0:
                    st.info(f"📸 {len(st.session_state.snapshots)} snapshots saved")
        
        # Auto-refresh mechanism
        time.sleep(0.1)
        st.rerun()  # This creates the continuous detection
        
    else:
        st.info("Click '🟢 Start Live Mode' to begin continuous tablet detection")
        st.markdown("### 📸 Manual Detection Mode")
        
        # Manual mode
        manual_camera = st.camera_input("Take a single photo for detection")
        
        if manual_camera is not None:
            image = Image.open(manual_camera)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔍 Analyze Photo", type="primary"):
                    # Process the image
                    img_array = np.array(image)
                    results = model(img_array, conf=confidence, verbose=False)
                    
                    tablet_count = 0
                    draw_image = image.copy()
                    draw = ImageDraw.Draw(draw_image)
                    
                    for result in results:
                        if result.boxes is not None:
                            tablet_count += len(result.boxes)
                            for i, box in enumerate(result.boxes.xyxy):
                                x1, y1, x2, y2 = map(int, box)
                                draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
                                text = f"Tablet {i+1}"
                                bbox = draw.textbbox((x1, y1-25), text)
                                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill="lime")
                                draw.text((x1, y1-25), text, fill="black")
                    
                    # Display results
                    st.image(draw_image, caption=f"Found {tablet_count} tablets", use_container_width=True)
                    
                    if tablet_count > 0:
                        st.success(f"✅ Detected {tablet_count} tablet(s)!")
                        
                        with col2:
                            if st.button("💾 Save Result", type="primary"):
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                snapshot_data = {
                                    'image': draw_image,
                                    'original_image': image,
                                    'count': tablet_count,
                                    'timestamp': timestamp
                                }
                                st.session_state.snapshots.append(snapshot_data)
                                st.success("Saved!")
                    else:
                        st.warning("No tablets detected")
    
    # Clear snapshots option
    if len(st.session_state.snapshots) > 0:
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📊 Total snapshots saved: {len(st.session_state.snapshots)}")
        with col2:
            if st.button("🗑️ Clear All Snapshots"):
                st.session_state.snapshots = []
                st.rerun()
    
    # Instructions
    with st.expander("💡 How Live Detection Works"):
        st.markdown("""
        ### 🎥 **Live Mode Features:**
        - **Continuous Detection**: Camera automatically takes photos at set intervals
        - **Real-time Analysis**: Each frame is immediately analyzed for tablets
        - **Visual Feedback**: Red bounding boxes and count overlay on live feed  
        - **Quick Save**: Save detections instantly when tablets are found
        - **Adjustable Speed**: Change refresh rate from 0.5s to 3s
        
        ### 📋 **Tips for Best Results:**
        1. **🔆 Good Lighting**: Ensure tablets are well-lit
        2. **📐 Flat Surface**: Keep tablets on a flat, contrasting background
        3. **📏 Steady Camera**: Hold device steady during detection
        4. **🎯 Clear View**: Make sure all tablets are fully visible
        5. **⚙️ Adjust Settings**: Lower confidence for more sensitive detection
        
        ### ⚠️ **Note**: 
        Live mode uses continuous camera refresh which may consume more battery on mobile devices.
        """)

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
st.markdown("""
### Features:
1. 🖼️ **Upload Image** - Upload and analyze tablet images
2. 📸 **Camera Snapshot** - Take a single photo and analyze
3. 🎥 **Live Detection** - Real-time detection with auto-refresh camera
""")

# Load model
with st.spinner("🔄 Loading AI model..."):
    model = load_model()

if model is None:
    st.error("❌ Failed to load the detection model. Please check if 'best80.pt' file exists.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Mode Selection
mode = st.selectbox(
    "🎯 Select Detection Mode:", 
    ["Upload Image", "Camera Snapshot", "Live Detection"],
    help="Choose how you want to provide images for tablet counting"
)

# ========== Upload Image ==========
if mode == "Upload Image":
    st.subheader("🖼️ Upload Image Analysis")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing tablets for analysis"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Show preview
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Count Tablets", type="primary"):
            with st.spinner("Analyzing image..."):
                count, boxed_image = model_count_tablets_with_boxes(image, model)
            
            st.image(boxed_image, caption=f"Detection Result: {count} tablets found", use_container_width=True)
            
            if count > 0:
                st.success(f"✅ Found {count} tablet(s) in the image!")
            else:
                st.warning("⚠️ No tablets detected. Try adjusting lighting or camera angle.")

# ========== Camera Snapshot ==========
elif mode == "Camera Snapshot":
    st.subheader("📸 Camera Snapshot Analysis")
    
    camera_file = st.camera_input(
        "Take a photo of tablets",
        help="Position tablets clearly in the camera view and take a photo"
    )
    
    if camera_file is not None:
        image = Image.open(camera_file)
        
        if st.button("🔍 Analyze This Photo", type="primary"):
            with st.spinner("Analyzing photo..."):
                count, boxed_image = model_count_tablets_with_boxes(image, model)
            
            st.image(boxed_image, caption=f"Analysis Result: {count} tablets detected", use_container_width=True)
            
            if count > 0:
                st.success(f"✅ Detected {count} tablet(s)!")
            else:
                st.warning("⚠️ No tablets detected.")

# ========== Live Detection ==========
elif mode == "Live Detection":
    live_detection_interface(model)
    
    # Display snapshots below live detection
    st.divider()
    display_snapshots()

# Footer
st.divider()
st.markdown("---")
st.markdown("💡 **Tips**: Ensure good lighting and clear view of tablets for best detection results.")
