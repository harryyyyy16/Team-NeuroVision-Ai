#  YOLOv8 Detection Suite

An advanced, feature-rich Streamlit application for object detection using YOLOv8 models. This application supports image detection, video processing, and batch image analysis with a beautiful, interactive interface.

## ✨Features

### Image Detection
- Upload and detect objects in single images
- Real-time visualization with bounding boxes
- Adjustable confidence and IoU thresholds
- Custom annotation styling
- Download annotated results
- Detailed detection statistics and charts

### Video Detection
- Process video files with object detection
- Configurable frame sampling rate
- Custom output FPS
- Download processed videos
- Video-level statistics and analysis

###  Batch Processing
- Upload and process multiple images simultaneously
- Batch statistics and CSV export
- Individual image results
- Bulk download options

###  Advanced Controls
- **Confidence Threshold**: Filter detections by confidence score
- **IoU Threshold**: Control Non-Maximum Suppression
- **Class Filtering**: Select specific classes to detect
- **Visual Customization**: Toggle labels, confidence scores, and box thickness
- **GPU Support**: Automatic CUDA detection and usage

###  Analytics & Visualization
- Interactive charts with Plotly
- Class distribution pie charts
- Detection confidence metrics
- Detailed statistics tables
- Export results to CSV

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Update the model path in `app_enhanced.py`:
```python
MODEL_PATH = "/path/to/your/yolov8_model.pt"
```

##  Usage

### Running the Application

```bash
streamlit run app_enhanced.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Select Detection Mode** (in sidebar):
   - Image Detection: Process single images
   - Video Detection: Process video files
   - Batch Processing: Process multiple images

2. **Configure Settings** (in sidebar):
   - Adjust confidence threshold (0.0 - 1.0)
   - Set IoU threshold for NMS
   - Select classes to detect
   - Customize visualization options

3. **Upload and Process**:
   - Upload your image(s) or video
   - Click the detection button
   - View results and statistics
   - Download processed outputs

## Project Structure

```
.
├── app_enhanced.py       # Main application file
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── yolov8_model.pt      # Your trained YOLOv8 model (not included)
```

## Customization

### Changing the Model

Replace the `MODEL_PATH` variable with your custom YOLOv8 model:

```python
MODEL_PATH = "/path/to/your/custom_model.pt"
```

### Adjusting Default Settings

Modify default values in the sidebar section:

```python
confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,  # Change default here
    step=0.05
)
```

### Custom Styling

Edit the CSS in the markdown section at the top of the file to customize colors and appearance.

##  Configuration Options

### Image Detection
- **Supported Formats**: JPG, JPEG, PNG, BMP, WEBP
- **Max File Size**: Depends on your Streamlit config (default 200MB)

### Video Detection
- **Supported Formats**: MP4, AVI, MOV, MKV
- **Frame Sampling**: Process every N frames (1-30)
- **Max Frames**: Limit processing for long videos
- **Output FPS**: Control output video frame rate

### Batch Processing
- **Multiple Upload**: Upload up to 200 images simultaneously
- **Parallel Processing**: Sequential processing with progress tracking
- **Export Options**: CSV summary and individual image downloads

## Output Information

### Detection Statistics
- Total objects detected
- Unique classes found
- Average and maximum confidence scores
- Per-class detection counts
- Object area measurements

### Export Formats
- **Images**: PNG format with annotations
- **Videos**: MP4 format with embedded detections
- **Data**: CSV files with detection metadata

## Troubleshooting

### Model Loading Issues
```
 Failed to load model: [Errno 2] No such file or directory
```
**Solution**: Check that `MODEL_PATH` points to a valid YOLOv8 model file.

### CUDA/GPU Issues
```
Device: CPU (CUDA not available)
```
**Solution**: 
- Install CUDA toolkit and compatible PyTorch version
- Check GPU compatibility: `torch.cuda.is_available()`

### Memory Issues
```
Out of memory error
```
**Solution**: 
- Reduce image resolution before uploading
- For videos, increase "Process every N frames" value
- Reduce "Max frames to process"

### Slow Processing
**Solutions**:
- Enable GPU processing (CUDA)
- Increase confidence threshold to filter more detections
- For videos, process fewer frames
- Reduce image resolution

##  Technical Details

### Model Architecture
- **Framework**: Ultralytics YOLOv8
- **Inference**: PyTorch backend
- **Acceleration**: CUDA support for GPU inference

### Processing Pipeline
1. Image/video upload and validation
2. Preprocessing and format conversion
3. YOLOv8 inference with configured parameters
4. Post-processing (NMS, filtering)
5. Visualization and annotation
6. Statistics computation and export

### Performance
- **Image**: ~0.1-0.5s per image (GPU) / 1-3s (CPU)
- **Video**: Depends on resolution, frame rate, and sampling
- **Batch**: Parallel-ready, limited by hardware

##  Tips for Best Results

1. **Adjust Confidence Threshold**: Lower for more detections, higher for precision
2. **Use Class Filtering**: Focus on relevant objects
3. **Optimize Video Processing**: Sample frames strategically
4. **GPU Acceleration**: Use CUDA for 10-50x speedup
5. **Image Quality**: Higher resolution = better detection (but slower)

##  Contributing

Feel free to customize and extend this application for your specific use case!

##  License

This application uses:
- **Streamlit**: Apache 2.0 License
- **Ultralytics YOLOv8**: AGPL-3.0 License
- **PyTorch**: BSD-style License

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Ultralytics YOLOv8 documentation
3. Check Streamlit documentation

##  Future Enhancements

Potential features to add:
- [ ] Real-time webcam detection
- [ ] Multi-model comparison
- [ ] Custom training interface
- [ ] API endpoint for programmatic access
- [ ] Database integration for result storage
- [ ] Advanced analytics dashboard
- [ ] Export to COCO/YOLO formats
- [ ] Object tracking in videos
- [ ] Cloud storage integration

---

**Built with ❤️ using YOLOv8 and Streamlit**
