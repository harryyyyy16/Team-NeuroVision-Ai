import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import io
import cv2
import numpy as np
import tempfile
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
MODEL_PATH = "/home/harryyyyy/Downloads/files (2)/yolov8_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="YOLOv8 Detection Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .detection-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<h1 class="main-header">🎯 YOLOv8 Detection Suite</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Object Detection with Image & Video Support</p>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png", use_container_width=True)
    
    st.header("⚙️ Configuration")
    
    # Mode selection
    mode = st.radio(
        "Detection Mode",
        ["📸 Image Detection", "🎥 Video Detection", "📊 Batch Processing"],
        help="Choose your detection mode"
    )
    
    st.divider()
    
    # Model settings
    st.subheader("🔧 Model Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for Non-Maximum Suppression"
    )
    
    st.divider()
    
    # Display settings
    st.subheader("🎨 Display Settings")
    
    show_labels = st.checkbox("Show Labels", value=True)
    show_conf = st.checkbox("Show Confidence", value=True)
    box_thickness = st.slider("Box Thickness", 1, 5, 2)
    
    st.divider()
    
    # Device info
    st.subheader("💻 System Info")
    st.info(f"**Device:** {DEVICE.upper()}")
    if torch.cuda.is_available():
        st.success(f"**GPU:** {torch.cuda.get_device_name(0)}")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Get class names from model
class_names = model.names if hasattr(model, 'names') else {}

# ---------------- CLASS FILTER ----------------
with st.sidebar:
    st.divider()
    st.subheader("🎯 Class Filter")
    if class_names:
        filter_classes = st.multiselect(
            "Select classes to detect",
            options=list(class_names.values()),
            default=list(class_names.values()),
            help="Only selected classes will be detected"
        )
    else:
        filter_classes = []

# ---------------- HELPER FUNCTIONS ----------------
def process_image(image, conf_thresh, iou_thresh, classes_filter):
    """Process a single image with YOLOv8"""
    
    # Filter class indices
    if classes_filter and class_names:
        class_indices = [k for k, v in class_names.items() if v in classes_filter]
    else:
        class_indices = None
    
    # Run inference
    results = model(
        image,
        conf=conf_thresh,
        iou=iou_thresh,
        classes=class_indices
    )
    
    return results[0]

def create_annotated_image(image, result, show_labels, show_conf, thickness):
    """Create custom annotated image"""
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    boxes = result.boxes
    
    if boxes is not None and len(boxes) > 0:
        for box in boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Generate color based on class
            color = plt.cm.tab10(int(cls) % 10)
            color = tuple(int(c * 255) for c in color[:3])
            color = (color[2], color[1], color[0])  # RGB to BGR
            
            # Draw box
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            if show_labels:
                label = f"{class_names[int(cls)]}"
                if show_conf:
                    label += f" {conf:.2f}"
                
                # Draw label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_bgr, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(img_bgr, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def get_detection_stats(result):
    """Extract detection statistics"""
    boxes = result.boxes
    
    if boxes is None or len(boxes) == 0:
        return None
    
    detections = []
    for box in boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        detections.append({
            'class': class_names[int(cls)],
            'confidence': conf,
            'bbox': [x1, y1, x2, y2],
            'area': (x2 - x1) * (y2 - y1)
        })
    
    return detections

# ---------------- MAIN CONTENT ----------------

if mode == "📸 Image Detection":
    st.header("📸 Image Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported formats: JPG, JPEG, PNG, BMP, WEBP"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.caption(f"📐 Size: {image.size[0]}x{image.size[1]} pixels")
    
    with col2:
        if uploaded_file is not None:
            if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
                with st.spinner("🔄 Running detection..."):
                    result = process_image(
                        image,
                        confidence_threshold,
                        iou_threshold,
                        filter_classes
                    )
                
                # Create annotated image
                annotated_img = create_annotated_image(
                    image, result, show_labels, show_conf, box_thickness
                )
                
                st.image(annotated_img, caption="Detection Results", use_container_width=True)
                
                # Download button
                buf = io.BytesIO()
                annotated_img.save(buf, format="PNG")
                st.download_button(
                    label="💾 Download Result",
                    data=buf.getvalue(),
                    file_name="detection_result.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    # Detection Statistics
    if uploaded_file is not None and 'result' in locals():
        st.divider()
        st.subheader("📊 Detection Statistics")
        
        detections = get_detection_stats(result)
        
        if detections:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="stat-box">
                        <h2>{len(detections)}</h2>
                        <p>Total Objects</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                unique_classes = len(set([d['class'] for d in detections]))
                st.markdown(f"""
                    <div class="stat-box">
                        <h2>{unique_classes}</h2>
                        <p>Unique Classes</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_conf = np.mean([d['confidence'] for d in detections])
                st.markdown(f"""
                    <div class="stat-box">
                        <h2>{avg_conf:.2%}</h2>
                        <p>Avg Confidence</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                max_conf = max([d['confidence'] for d in detections])
                st.markdown(f"""
                    <div class="stat-box">
                        <h2>{max_conf:.2%}</h2>
                        <p>Max Confidence</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Detailed table and charts
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📋 Detection Details")
                df = pd.DataFrame(detections)
                df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                df['area'] = df['area'].apply(lambda x: f"{x:.0f}")
                st.dataframe(
                    df[['class', 'confidence', 'area']],
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.subheader("📊 Class Distribution")
                class_counts = pd.DataFrame(detections)['class'].value_counts()
                fig = px.pie(
                    values=class_counts.values,
                    names=class_counts.index,
                    title="Objects by Class",
                    hole=0.4
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No objects detected with current settings")

elif mode == "🎥 Video Detection":
    st.header("🎥 Video Detection")
    
    st.info("📹 Upload a video file for object detection processing")
    
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_video is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        
        st.video(tfile.name)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            process_every_n_frames = st.number_input(
                "Process every N frames",
                min_value=1,
                max_value=30,
                value=5,
                help="Process every Nth frame to speed up processing"
            )
        
        with col2:
            max_frames = st.number_input(
                "Max frames to process",
                min_value=10,
                max_value=1000,
                value=100,
                help="Maximum number of frames to process"
            )
        
        with col3:
            output_fps = st.number_input(
                "Output FPS",
                min_value=1,
                max_value=60,
                value=10,
                help="Frames per second for output video"
            )
        
        if st.button("🎬 Process Video", type="primary", use_container_width=True):
            cap = cv2.VideoCapture(tfile.name)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            st.info(f"📹 Video Info: {total_frames} frames @ {original_fps:.2f} FPS | {width}x{height}")
            
            # Create temp output file
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            processed_count = 0
            all_detections = []
            
            while cap.isOpened() and processed_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % process_every_n_frames == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Process frame
                    result = process_image(
                        pil_image,
                        confidence_threshold,
                        iou_threshold,
                        filter_classes
                    )
                    
                    # Get detections for this frame
                    frame_detections = get_detection_stats(result)
                    if frame_detections:
                        all_detections.extend(frame_detections)
                    
                    # Annotate frame
                    annotated = create_annotated_image(
                        pil_image, result, show_labels, show_conf, box_thickness
                    )
                    annotated_bgr = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
                    
                    out.write(annotated_bgr)
                    processed_count += 1
                    
                    progress = min(processed_count / max_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {processed_count}/{max_frames} frames")
                
                frame_count += 1
            
            cap.release()
            out.release()
            
            st.success("✅ Video processing complete!")
            
            # Display processed video
            st.video(output_path)
            
            # Download button
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="💾 Download Processed Video",
                    data=f.read(),
                    file_name="detected_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            
            # Video statistics
            if all_detections:
                st.divider()
                st.subheader("📊 Video Detection Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Detections", len(all_detections))
                
                with col2:
                    unique_classes = len(set([d['class'] for d in all_detections]))
                    st.metric("Unique Classes", unique_classes)
                
                with col3:
                    avg_conf = np.mean([d['confidence'] for d in all_detections])
                    st.metric("Average Confidence", f"{avg_conf:.2%}")
                
                # Class distribution over time
                df = pd.DataFrame(all_detections)
                class_counts = df['class'].value_counts()
                
                fig = px.bar(
                    x=class_counts.index,
                    y=class_counts.values,
                    labels={'x': 'Class', 'y': 'Count'},
                    title="Total Detections by Class"
                )
                st.plotly_chart(fig, use_container_width=True)

elif mode == "📊 Batch Processing":
    st.header("📊 Batch Processing")
    
    st.info("📁 Upload multiple images for batch processing")
    
    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        help="Upload multiple images at once"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        # Display thumbnails
        cols = st.columns(5)
        for idx, file in enumerate(uploaded_files[:10]):  # Show first 10
            with cols[idx % 5]:
                image = Image.open(file)
                st.image(image, caption=file.name, use_container_width=True)
        
        if len(uploaded_files) > 10:
            st.caption(f"... and {len(uploaded_files) - 10} more images")
        
        if st.button("🚀 Process All Images", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_data = []
            processed_images = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                image = Image.open(uploaded_file)
                result = process_image(
                    image,
                    confidence_threshold,
                    iou_threshold,
                    filter_classes
                )
                
                detections = get_detection_stats(result)
                
                # Store results
                results_data.append({
                    'filename': uploaded_file.name,
                    'total_objects': len(detections) if detections else 0,
                    'unique_classes': len(set([d['class'] for d in detections])) if detections else 0,
                    'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0
                })
                
                # Create annotated image
                annotated = create_annotated_image(
                    image, result, show_labels, show_conf, box_thickness
                )
                processed_images.append((uploaded_file.name, annotated))
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Batch processing complete!")
            
            st.divider()
            st.subheader("📊 Batch Results Summary")
            
            # Summary statistics
            df_results = pd.DataFrame(results_data)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", len(df_results))
            with col2:
                st.metric("Total Objects", df_results['total_objects'].sum())
            with col3:
                st.metric("Avg Objects/Image", f"{df_results['total_objects'].mean():.1f}")
            with col4:
                st.metric("Avg Confidence", f"{df_results['avg_confidence'].mean():.2%}")
            
            # Results table
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
            # Download results as CSV
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="batch_results.csv",
                mime="text/csv"
            )
            
            # Display processed images
            st.divider()
            st.subheader("🖼️ Processed Images")
            
            cols = st.columns(2)
            for idx, (filename, img) in enumerate(processed_images):
                with cols[idx % 2]:
                    st.image(img, caption=filename, use_container_width=True)
                    
                    # Individual download
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(
                        label=f"💾 Download {filename}",
                        data=buf.getvalue(),
                        file_name=f"detected_{filename}",
                        mime="image/png",
                        key=f"download_{idx}"
                    )

# ---------------- FOOTER ----------------
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>🎯 <strong>YOLOv8 Detection Suite</strong> | Powered by Ultralytics & Streamlit</p>
        <p style='font-size: 0.9rem;'>Adjust settings in the sidebar to customize detection parameters</p>
    </div>
""", unsafe_allow_html=True)