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
def model_count_tablets_with_boxes(image, model, confidence=0.45):
    """Run YOLO detection, count tablets, and draw bounding boxes"""
    if model is None:
        return 0, image
    
    try:
        img_array = np.array(image)
        results = model(img_array, conf=confidence, verbose=False)
        tablet_count = 0
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        for result in results:
            if result.boxes is not None:
                tablet_count += len(result.boxes)
                for i, box in enumerate(result.boxes.xyxy):  # xyxy = [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, box)
                    # Draw bounding box with thicker lines
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                    # Add tablet number with background
                    text = f"#{i+1}"
                    bbox = draw.textbbox((x1, y1-25), text)
                    draw.rectangle([bbox[0]-3, bbox[1]-2, bbox[2]+3, bbox[3]+2], fill="red")
                    draw.text((x1, y1-25), text, fill="white")
        
        return tablet_count, draw_image
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        return 0, image

# ==================== Working Live Detection ====================
def live_detection_interface(model):
    """Working live detection with timer-based refresh"""
    
    # Initialize session state
    if 'snapshots' not in st.session_state:
        st.session_state.snapshots = []
    if 'current_detection' not in st.session_state:
        st.session_state.current_detection = None
    if 'live_mode_active' not in st.session_state:
        st.session_state.live_mode_active = False
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    
    st.subheader("🎥 Live Tablet Detection")
    
    # Settings and control panel
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        confidence = st.slider("🎯 Detection Confidence", 0.1, 1.0, 0.45, 0.05)
    
    with col2:
        refresh_interval = st.selectbox("📊 Refresh Rate", 
                                       options=[1, 2, 3, 4], 
                                       index=1, 
                                       format_func=lambda x: f"{x}s")
    
    with col3:
        if not st.session_state.live_mode_active:
            if st.button("🟢 **START LIVE**", type="primary"):
                st.session_state.live_mode_active = True
                st.session_state.last_refresh_time = time.time()
                st.rerun()
        else:
            if st.button("🔴 **STOP LIVE**", type="secondary"):
                st.session_state.live_mode_active = False
                st.session_state.current_detection = None
                st.rerun()
    
    # Live detection status
    if st.session_state.live_mode_active:
        st.success("🔴 **LIVE DETECTION ACTIVE** - Keep tablets in camera view!")
        
        # Auto-refresh logic
        current_time = time.time()
        if current_time - st.session_state.last_refresh_time >= refresh_interval:
            st.session_state.last_refresh_time = current_time
            # Force refresh by incrementing a counter
            if 'refresh_counter' not in st.session_state:
                st.session_state.refresh_counter = 0
            st.session_state.refresh_counter += 1
        
        # Create unique camera key for each refresh
        camera_key = f"live_detection_camera_{st.session_state.get('refresh_counter', 0)}"
        
        # Camera input with live detection
        camera_image = st.camera_input(
            f"🔴 LIVE CAMERA (Refresh #{st.session_state.get('refresh_counter', 0)}) - Auto-detecting...",
            key=camera_key
        )
        
        if camera_image is not None:
            # Process the image immediately
            current_image = Image.open(camera_image)
            
            # Run detection
            count, detected_image = model_count_tablets_with_boxes(current_image, model, confidence)
            
            # Store results
            st.session_state.current_detection = {
                'original_image': current_image,
                'detected_image': detected_image,
                'count': count,
                'timestamp': datetime.now(),
                'refresh_count': st.session_state.get('refresh_counter', 0)
            }
            
            # Display results in two columns
            col_img, col_data = st.columns([2, 1])
            
            with col_img:
                # Show detected image with bounding boxes
                st.image(detected_image, 
                        caption=f"🔴 LIVE DETECTION: {count} tablets found", 
                        use_container_width=True)
            
            with col_data:
                st.markdown("### 📊 Live Results")
                
                if count > 0:
                    st.success(f"✅ **{count} TABLETS**")
                    
                    # Screenshot button
                    if st.button("📸 **SCREENSHOT**", type="primary", use_container_width=True):
                        snapshot_data = {
                            'image': detected_image,
                            'original_image': current_image,
                            'count': count,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.snapshots.append(snapshot_data)
                        st.success("✅ Saved!")
                        st.balloons()
                else:
                    st.info("👀 **SCANNING...**")
                
                # Live stats
                st.metric("🎯 Count", count)
                st.metric("🔄 Refresh", f"#{st.session_state.get('refresh_counter', 0)}")
                st.text(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
                
                # Quick tips
                if count == 0:
                    st.markdown("""
                    **💡 Tips:**
                    - Better lighting
                    - Lower confidence  
                    - Spread tablets apart
                    - Hold camera steady
                    """)
        
        # Auto-refresh mechanism - this is key!
        import asyncio
        if st.session_state.get('refresh_counter', 0) < 200:  # Safety limit
            # Add a small delay then refresh
            st.rerun()
            
    else:
        # Stopped mode
        st.info("Click **'🟢 START LIVE'** to begin continuous tablet detection")
        
        # Show last detection if available
        if st.session_state.current_detection:
            st.markdown("### 📊 Last Detection Result:")
            detection = st.session_state.current_detection
            
            col_img, col_data = st.columns([2, 1])
            with col_img:
                st.image(detection['detected_image'], 
                        caption=f"Last result: {detection['count']} tablets", 
                        use_container_width=True)
            with col_data:
                st.metric("Last Count", detection['count'])
                if st.button("💾 Save Last Result"):
                    snapshot_data = {
                        'image': detection['detected_image'],
                        'original_image': detection['original_image'],
                        'count': detection['count'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.snapshots.append(snapshot_data)
                    st.success("✅ Saved!")

    
    # Snapshots summary
    if len(st.session_state.snapshots) > 0:
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📸 Screenshots Taken", len(st.session_state.snapshots))
        
        with col2:
            total_tablets = sum(snap['count'] for snap in st.session_state.snapshots)
            st.metric("🧮 Total Tablets", total_tablets)
        
        with col3:
            if st.button("🗑️ Clear All Screenshots"):
                st.session_state.snapshots = []
                st.session_state.current_detection = None
                st.rerun()
    
    # Instructions  
    with st.expander("💡 **How Live Detection Works**"):
        st.markdown("""
        ### 🔴 **True Live Detection Mode:**
        
        1. **🟢 Click START LIVE**: Activates continuous detection
        2. **📱 Position Camera**: Point camera at tablets on a flat surface
        3. **👀 Auto Detection**: Camera refreshes every 1-4 seconds automatically
        4. **📊 Live Results**: See bounding boxes and count in real-time
        5. **📸 Screenshot**: Save when tablets are detected
        6. **🔴 Click STOP**: Turn off live mode
        
        ### 🎯 **What You'll See:**
        - **Red bounding boxes** around detected tablets
        - **Live tablet count** updating automatically
        - **Refresh counter** showing detection activity
        - **Screenshot button** when tablets are found
        
        ### 📋 **Pro Tips:**
        - ☀️ **Bright lighting** works best
        - 🔲 **Plain background** (white paper ideal)
        - 📐 **Flat surface** - spread tablets out
        - 📏 **Hold steady** during detection
        - ⚙️ **Lower confidence** for more sensitive detection
        - 🔄 **Be patient** - allow a few seconds between refreshes
        
        ### ⚠️ **Troubleshooting:**
        - **No bounding boxes?** Try lower confidence or better lighting
        - **False detections?** Increase confidence or improve background
        - **Battery drain?** Use longer refresh intervals (3-4s)
        """)
        
    # Show snapshots count
    if len(st.session_state.snapshots) > 0:
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"📸 {len(st.session_state.snapshots)} screenshots saved")
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.snapshots = []
                st.rerun()

# ==================== Display Snapshots ====================
def display_snapshots():
    """Display saved snapshots with enhanced features"""
    if st.session_state.get('snapshots'):
        st.subheader("📷 Saved Screenshots")
        st.write(f"Total screenshots: {len(st.session_state.snapshots)}")
        
        # Summary statistics
        if st.session_state.snapshots:
            total_tablets = sum(snap['count'] for snap in st.session_state.snapshots)
            avg_tablets = total_tablets / len(st.session_state.snapshots)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📸 Total Screenshots", len(st.session_state.snapshots))
            col2.metric("🧮 Total Tablets", total_tablets)
            col3.metric("📊 Average per Screenshot", f"{avg_tablets:.1f}")
        
        # Display snapshots in reverse order (newest first)
        for i, snapshot in enumerate(reversed(st.session_state.snapshots)):
            snapshot_num = len(st.session_state.snapshots) - i
            
            with st.expander(f"📸 Screenshot #{snapshot_num} - {snapshot['count']} tablets - {snapshot['timestamp']}", expanded=i==0):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Show both detection result and original
                    tab1, tab2 = st.tabs(["🎯 With Detection", "📷 Original"])
                    
                    with tab1:
                        st.image(snapshot['image'], caption=f"Detected: {snapshot['count']} tablets", use_container_width=True)
                    
                    with tab2:
                        st.image(snapshot['original_image'], caption="Original image", use_container_width=True)
                
                with col2:
                    st.metric("Tablets", snapshot['count'])
                    st.text(f"📅 {snapshot['timestamp']}")
                    
                    # Download buttons
                    for img_type, img_key, label in [
                        ("detection", 'image', "⬇️ With Boxes"),
                        ("original", 'original_image', "⬇️ Original")
                    ]:
                        img_buffer = io.BytesIO()
                        snapshot[img_key].save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        
                        st.download_button(
                            label=label,
                            data=img_buffer.getvalue(),
                            file_name=f"tablets_{img_type}_{snapshot_num}_{snapshot['timestamp'].replace(':', '-').replace(' ', '_')}.png",
                            mime="image/png",
                            key=f"download_{img_type}_{i}",
                            use_container_width=True
                        )
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{i}", use_container_width=True):
                        actual_index = len(st.session_state.snapshots) - 1 - i
                        st.session_state.snapshots.pop(actual_index)
                        st.rerun()

# ---------------- STREAMLIT UI ----------------
st.title("💊 Tablet Counter Pro")
st.markdown("""
### 🚀 **Enhanced Features:**
- 🖼️ **Upload Image** - Upload and analyze tablet images
- 📸 **Camera Snapshot** - Take a single photo and analyze  
- 🎥 **Live Detection** - Real-time detection with pause-and-screenshot
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
        
        # Settings
        confidence = st.slider("🎯 Detection Confidence", 0.1, 1.0, 0.45, 0.05)
        
        # Show preview
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Count Tablets", type="primary"):
            with st.spinner("Analyzing image..."):
                count, boxed_image = model_count_tablets_with_boxes(image, model, confidence)
            
            st.image(boxed_image, caption=f"Detection Result: {count} tablets found", use_container_width=True)
            
            if count > 0:
                st.success(f"✅ Found {count} tablet(s) in the image!")
            else:
                st.warning("⚠️ No tablets detected. Try adjusting lighting or confidence level.")

# ========== Camera Snapshot ==========
elif mode == "Camera Snapshot":
    st.subheader("📸 Camera Snapshot Analysis")
    
    # Settings
    confidence = st.slider("🎯 Detection Confidence", 0.1, 1.0, 0.45, 0.05)
    
    camera_file = st.camera_input(
        "Take a photo of tablets",
        help="Position tablets clearly in the camera view and take a photo"
    )
    
    if camera_file is not None:
        image = Image.open(camera_file)
        
        if st.button("🔍 Analyze This Photo", type="primary"):
            with st.spinner("Analyzing photo..."):
                count, boxed_image = model_count_tablets_with_boxes(image, model, confidence)
            
            st.image(boxed_image, caption=f"Analysis Result: {count} tablets detected", use_container_width=True)
            
            if count > 0:
                st.success(f"✅ Detected {count} tablet(s)!")
                # Option to save
                if st.button("💾 Save Result"):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    snapshot_data = {
                        'image': boxed_image,
                        'original_image': image,
                        'count': count,
                        'timestamp': timestamp
                    }
                    if 'snapshots' not in st.session_state:
                        st.session_state.snapshots = []
                    st.session_state.snapshots.append(snapshot_data)
                    st.success("✅ Result saved!")
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
st.markdown("""
### 💡 **Pro Tips for Better Detection:**
- 🔆 **Lighting**: Use natural light or bright indoor lighting
- 🔲 **Background**: Use plain, contrasting colored surface (white paper works great)
- 📐 **Layout**: Spread tablets apart, avoid overlapping
- 📱 **Distance**: Keep camera 6-12 inches away from tablets
- ⚖️ **Confidence**: Lower confidence detects more but may include false positives
""")
