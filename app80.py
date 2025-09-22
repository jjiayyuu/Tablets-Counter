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
    """Stable live detection interface using Streamlit's camera input"""
    
    # Initialize session state
    if 'snapshots' not in st.session_state:
        st.session_state.snapshots = []
    if 'current_detection' not in st.session_state:
        st.session_state.current_detection = None
    if 'last_image' not in st.session_state:
        st.session_state.last_image = None
    
    st.subheader("🎥 Live Tablet Detection")
    st.info("📌 **How it works**: Take photos with the camera below. Each photo will be automatically analyzed for tablets and show bounding boxes.")
    
    # Control panel
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Snapshots"):
            st.session_state.snapshots = []
            st.rerun()
    
    with col2:
        # Confidence threshold
        confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.45, 0.05)
    
    # Main camera interface - STABLE, no auto-refresh
    st.markdown("### 📷 Camera")
    camera_image = st.camera_input(
        "Take photos to detect tablets",
        help="Position tablets clearly in view and click the camera button to take a photo",
        key="main_camera"
    )
    
    # Process camera image when available
    if camera_image is not None:
        # Only process if it's a new image
        if st.session_state.last_image != camera_image:
            st.session_state.last_image = camera_image
            
            # Process the image
            image = Image.open(camera_image)
            
            # Run detection with custom confidence
            with st.spinner("🔍 Analyzing image for tablets..."):
                img_array = np.array(image)
                results = model(img_array, conf=confidence, verbose=False)
                
                tablet_count = 0
                draw_image = image.copy()
                draw = ImageDraw.Draw(draw_image)
                
                # Draw bounding boxes
                for result in results:
                    if result.boxes is not None:
                        tablet_count += len(result.boxes)
                        for i, box in enumerate(result.boxes.xyxy):
                            x1, y1, x2, y2 = map(int, box)
                            # Draw bounding box
                            draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
                            # Draw tablet number with background
                            text = f"Tablet {i+1}"
                            # Get text size for background rectangle
                            bbox = draw.textbbox((x1, y1-25), text)
                            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill="lime")
                            draw.text((x1, y1-25), text, fill="black")
                
                # Store current detection
                st.session_state.current_detection = {
                    'original': image,
                    'processed': draw_image,
                    'count': tablet_count
                }
    
    # Display current detection results
    if st.session_state.current_detection:
        st.markdown("### 🎯 Detection Results")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Show tabs for original and processed image
            tab1, tab2 = st.tabs(["🎯 With Detection Boxes", "📷 Original Image"])
            
            with tab1:
                st.image(
                    st.session_state.current_detection['processed'], 
                    caption=f"Detected: {st.session_state.current_detection['count']} tablets",
                    use_container_width=True
                )
            
            with tab2:
                st.image(
                    st.session_state.current_detection['original'],
                    caption="Original camera image",
                    use_container_width=True
                )
        
        with col2:
            # Detection metrics
            count = st.session_state.current_detection['count']
            st.metric("Tablets Found", count)
            
            # Status message
            if count > 0:
                st.success(f"✅ {count} tablet(s) detected!")
            else:
                st.info("👀 No tablets found")
            
            # Save button
            if st.button("💾 Save This Detection", type="primary", disabled=count==0):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                snapshot_data = {
                    'image': st.session_state.current_detection['processed'],
                    'original_image': st.session_state.current_detection['original'],
                    'count': count,
                    'timestamp': timestamp
                }
                st.session_state.snapshots.append(snapshot_data)
                st.success(f"✅ Snapshot saved! {count} tablets at {timestamp}")
                st.balloons()  # Celebration effect
    
    else:
        st.info("📸 Take a photo above to start detecting tablets!")
        
    # Instructions
    with st.expander("💡 Tips for Better Detection"):
        st.markdown("""
        **For best results:**
        - 🔆 Ensure good lighting
        - 📐 Keep tablets flat and separated
        - 📏 Keep camera steady and at appropriate distance
        - 🎯 Make sure tablets are fully visible in frame
        - 🔄 Adjust confidence slider if needed (lower = more sensitive)
        
        **The system will:**
        - ✅ Automatically detect tablets in each photo
        - 📦 Draw green boxes around found tablets
        - 🔢 Number each detected tablet
        - 💾 Allow you to save successful detections
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
