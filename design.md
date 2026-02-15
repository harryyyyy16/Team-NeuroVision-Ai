# Design Document: YOLOv8 Detection Suite

## Overview

The YOLOv8 Detection Suite is a web-based object detection application built with Streamlit that provides an intuitive interface for detecting objects in images and videos. The system leverages the Ultralytics YOLOv8 deep learning model for state-of-the-art object detection, offering three primary modes: single image detection, video detection with frame sampling, and batch image processing.

The application architecture follows a modular design with clear separation between the UI layer (Streamlit), detection logic (YOLOv8 model integration), image/video processing (OpenCV/PIL), and analytics (Plotly/Pandas). The system supports GPU acceleration via CUDA when available, with automatic fallback to CPU processing.

Key design principles:
- **Simplicity**: Single-file application with clear functional boundaries
- **Performance**: Model caching, GPU acceleration, and configurable frame sampling
- **Flexibility**: Adjustable detection parameters and annotation styling
- **User Experience**: Real-time feedback, interactive visualizations, and easy export

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Mode Selector│  │ Configuration│  │ File Uploader│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Detection Engine                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Model Manager (Cached YOLOv8 Model)                 │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Detection Processor (Inference + Filtering)         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Media Processing Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │Image Processor│  │Video Processor│  │Batch Processor│    │
│  │  (PIL/OpenCV)│  │   (OpenCV)   │  │   (Parallel) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Analytics & Visualization Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │Stats Calculator│  │Chart Generator│  │Export Manager│    │
│  │   (Pandas)   │  │   (Plotly)   │  │  (CSV/Files) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

**Image Detection Flow:**
1. User uploads image via Streamlit file uploader
2. Image loaded as PIL Image object
3. Detection processor runs YOLOv8 inference with configured parameters
4. Results filtered by confidence threshold and class filter
5. Annotation engine draws bounding boxes and labels on image copy
6. Statistics calculator extracts detection metrics
7. UI displays annotated image, statistics, and charts
8. User can download annotated image and CSV data

**Video Detection Flow:**
1. User uploads video file, saved to temporary location
2. OpenCV VideoCapture opens video stream
3. Frame sampler selects every Nth frame based on configuration
4. Each sampled frame processed through detection pipeline
5. Annotated frames written to output video via VideoWriter
6. Detections aggregated across all processed frames
7. UI displays processed video and aggregate statistics
8. User can download annotated video and CSV data

**Batch Processing Flow:**
1. User uploads multiple images simultaneously
2. System iterates through all uploaded files
3. Each image processed independently with same configuration
4. Results collected in list structure
5. Statistics aggregated across all images
6. UI displays summary table and individual annotated images
7. User can download individual images and aggregate CSV

## Components and Interfaces

### 1. Model Manager

**Responsibility:** Load, cache, and provide access to the YOLOv8 model.

**Interface:**
```python
@st.cache_resource
def load_model() -> YOLO | None:
    """
    Load YOLOv8 model from disk and move to appropriate device.
    
    Returns:
        YOLO model instance or None if loading fails
    
    Caching:
        Uses Streamlit's cache_resource to load model once per session
    """
```

**Implementation Details:**
- Uses Streamlit's `@st.cache_resource` decorator for singleton pattern
- Automatically detects CUDA availability and moves model to GPU/CPU
- Returns None on failure with error message to UI
- Model path configured via global constant `MODEL_PATH`

### 2. Detection Processor

**Responsibility:** Execute object detection inference with filtering.

**Interface:**
```python
def process_image(
    image: PIL.Image,
    conf_thresh: float,
    iou_thresh: float,
    classes_filter: List[str]
) -> ultralytics.engine.results.Results:
    """
    Process single image through YOLOv8 model with filtering.
    
    Args:
        image: PIL Image object to process
        conf_thresh: Minimum confidence threshold (0.0-1.0)
        iou_thresh: IoU threshold for NMS (0.0-1.0)
        classes_filter: List of class names to detect (empty = all classes)
    
    Returns:
        YOLOv8 Results object containing detections
    """
```

**Implementation Details:**
- Converts class name filter to class indices using model's class mapping
- Passes parameters directly to YOLOv8 model's `__call__` method
- Returns first result from results list (single image inference)
- Model handles NMS internally based on IoU threshold

### 3. Annotation Engine

**Responsibility:** Create visually annotated images with detection overlays.

**Interface:**
```python
def create_annotated_image(
    image: PIL.Image,
    result: ultralytics.engine.results.Results,
    show_labels: bool,
    show_conf: bool,
    thickness: int
) -> PIL.Image:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: Original PIL Image
        result: YOLOv8 detection results
        show_labels: Whether to display class labels
        show_conf: Whether to display confidence scores
        thickness: Bounding box line thickness (1-5)
    
    Returns:
        New PIL Image with annotations drawn
    """
```

**Implementation Details:**
- Converts PIL Image to NumPy array for OpenCV operations
- Iterates through detection boxes from results
- Generates distinct colors per class using matplotlib colormap
- Draws rectangles with OpenCV `cv2.rectangle`
- Draws text labels with background rectangles for readability
- Converts back to PIL Image for Streamlit display
- Does not modify original image (creates copy)

### 4. Statistics Calculator

**Responsibility:** Extract and compute detection metrics.

**Interface:**
```python
def get_detection_stats(
    result: ultralytics.engine.results.Results
) -> List[Dict[str, Any]] | None:
    """
    Extract detection statistics from results.
    
    Args:
        result: YOLOv8 detection results
    
    Returns:
        List of detection dictionaries with keys:
        - class: str (class name)
        - confidence: float (0.0-1.0)
        - bbox: List[float] (x1, y1, x2, y2)
        - area: float (bounding box area in pixels)
        
        Returns None if no detections found
    """
```

**Implementation Details:**
- Accesses `result.boxes.data` tensor containing detection information
- Each box contains: [x1, y1, x2, y2, confidence, class_id]
- Maps class IDs to class names using model's class dictionary
- Calculates bounding box area as (width × height)
- Returns structured data suitable for Pandas DataFrame conversion

### 5. Video Processor

**Responsibility:** Process video files frame-by-frame with detection.

**Implementation Details:**
- Uses OpenCV `VideoCapture` to read input video
- Extracts video metadata: frame count, FPS, resolution
- Implements frame sampling: processes every Nth frame
- Uses OpenCV `VideoWriter` to create output video
- Codec: MP4V for web compatibility
- Maintains original resolution in output
- Converts frames between BGR (OpenCV) and RGB (PIL) color spaces
- Aggregates detections across all processed frames
- Cleans up temporary files after processing

### 6. Batch Processor

**Responsibility:** Process multiple images with same configuration.

**Implementation Details:**
- Iterates through list of uploaded files sequentially
- Applies identical detection configuration to all images
- Collects results in list structure for aggregation
- Displays progress bar updated after each image
- Generates summary statistics: total objects, average confidence
- Creates Pandas DataFrame for tabular results display
- Provides individual download buttons for each processed image

### 7. Export Manager

**Responsibility:** Generate downloadable outputs.

**Implementation Details:**
- **Image Export:** Saves PIL Image to BytesIO buffer as PNG
- **Video Export:** Reads processed video file as bytes
- **CSV Export:** Converts detection data to Pandas DataFrame, exports as CSV
- CSV format includes columns: filename, class, confidence, bbox coordinates
- Uses Streamlit's `download_button` for file delivery
- Generates appropriate MIME types for each file format

### 8. Analytics Visualizer

**Responsibility:** Generate interactive charts and statistics displays.

**Implementation Details:**
- Uses Plotly Express for chart generation
- **Pie Chart:** Class distribution with donut hole style
- **Bar Chart:** Detection counts by class for video/batch modes
- Calculates aggregate statistics: mean, min, max confidence
- Formats statistics in styled HTML boxes with gradients
- Updates visualizations reactively when detections change

## Data Models

### Detection Result

```python
{
    'class': str,           # Object class name (e.g., "person", "car")
    'confidence': float,    # Detection confidence score (0.0-1.0)
    'bbox': [float, float, float, float],  # [x1, y1, x2, y2] coordinates
    'area': float          # Bounding box area in pixels
}
```

### Configuration State

```python
{
    'mode': str,                    # "Image", "Video", or "Batch"
    'confidence_threshold': float,  # 0.0-1.0
    'iou_threshold': float,         # 0.0-1.0
    'classes_filter': List[str],    # Selected class names
    'show_labels': bool,            # Display class labels
    'show_conf': bool,              # Display confidence scores
    'box_thickness': int,           # Line thickness (1-5)
    'frame_sampling': int,          # Process every Nth frame (video mode)
    'max_frames': int,              # Maximum frames to process (video mode)
    'output_fps': int              # Output video frame rate (video mode)
}
```

### Batch Results Summary

```python
{
    'filename': str,           # Original filename
    'total_objects': int,      # Number of detections
    'unique_classes': int,     # Number of distinct classes
    'avg_confidence': float    # Mean confidence score
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Confidence Threshold Filtering

*For any* detection result and any confidence threshold value, all returned detections should have confidence scores greater than or equal to the specified threshold.

**Validates: Requirements 1.4, 2.4**

### Property 2: Class Filter Enforcement

*For any* detection result and any selected class filter, all returned detections should only contain objects whose class names are in the selected filter list.

**Validates: Requirements 1.6, 2.5**

### Property 3: Frame Sampling Correctness

*For any* video and any sampling rate N, the system should process frames at indices that are multiples of N (0, N, 2N, 3N, ...).

**Validates: Requirements 2.2**

### Property 4: Batch Processing Completeness

*For any* set of input images in batch mode, the number of output annotated images should equal the number of input images.

**Validates: Requirements 3.1, 3.3**

### Property 5: Batch Configuration Consistency

*For any* batch of images and any detection configuration, all images in the batch should be processed with identical confidence threshold, IoU threshold, and class filter settings.

**Validates: Requirements 3.2**

### Property 6: Batch Statistics Aggregation

*For any* batch processing result, the aggregate total detection count should equal the sum of detection counts from all individual images.

**Validates: Requirements 3.4**

### Property 7: Configuration Range Validation

*For any* confidence threshold or IoU threshold value, the system should only accept values in the range [0.0, 1.0] and reject values outside this range.

**Validates: Requirements 4.1, 4.2**

### Property 8: Annotation Styling Consistency

*For any* detection result and annotation settings, the same styling (label visibility, confidence visibility, box thickness) should be applied consistently across all bounding boxes in the result.

**Validates: Requirements 5.5**

### Property 9: Device Independence

*For any* image and detection configuration, the detection results should be equivalent whether processed on CPU or GPU (same classes detected with same confidence scores within floating-point tolerance).

**Validates: Requirements 6.4**

### Property 10: Detection Statistics Accuracy

*For any* detection result, the calculated statistics (total count, class distribution, mean/min/max confidence) should accurately reflect the detection data:
- Total count equals number of detections
- Class distribution sums to total count
- Mean confidence equals average of all confidence scores
- Min/max confidence are the actual minimum and maximum values

**Validates: Requirements 7.1, 7.2, 7.3**

### Property 11: CSV Export Completeness

*For any* detection result exported to CSV, the CSV should contain one row per detection with all required fields: class name, confidence score, and bounding box coordinates (x1, y1, x2, y2).

**Validates: Requirements 8.4**

### Property 12: CSV Format Correctness

*For any* CSV export, the file should include a header row with column names and all data rows should have the same number of columns as the header.

**Validates: Requirements 8.5**

### Property 13: Image Format Acceptance

*For any* valid image file in JPG, PNG, or JPEG format, the system should successfully load and process the image without errors.

**Validates: Requirements 1.1, 11.1**

### Property 14: Video Format Acceptance

*For any* valid video file in MP4, AVI, or MOV format, the system should successfully load and process the video without errors.

**Validates: Requirements 2.1, 11.2**

### Property 15: Invalid Format Rejection

*For any* file with an unsupported format (not in the accepted image/video formats), the system should reject the file and not attempt processing.

**Validates: Requirements 11.3**

### Property 16: File Integrity Validation

*For any* uploaded file, the system should verify the file can be opened and read before attempting detection processing.

**Validates: Requirements 11.5**

### Property 17: Annotation Thickness Application

*For any* image annotation and thickness setting, all bounding boxes should be drawn with the specified line thickness value.

**Validates: Requirements 12.3**

### Property 18: Aspect Ratio Preservation

*For any* input image, the output annotated image should maintain the same aspect ratio (width/height ratio) as the input.

**Validates: Requirements 12.5**

### Property 19: Video Resolution Preservation

*For any* input video, the output annotated video should maintain the same frame resolution (width × height) as the input.

**Validates: Requirements 13.3**

### Property 20: Video Frame Detection Coverage

*For any* video and sampling configuration, every sampled frame (at indices 0, N, 2N, ...) should have detection processing applied.

**Validates: Requirements 13.4**

## Error Handling

### Model Loading Errors

**Scenario:** Model file missing or corrupted
- **Detection:** Try-except block in `load_model()` function
- **Response:** Display error message via `st.error()`, return None
- **Recovery:** Application stops execution via `st.stop()`, preventing detection operations
- **User Guidance:** Error message includes file path and suggests checking model file

### File Upload Errors

**Scenario:** Invalid file format or corrupted file
- **Detection:** File extension validation and PIL/OpenCV loading attempts
- **Response:** Display error message indicating invalid format
- **Recovery:** Allow user to upload different file
- **User Guidance:** List supported formats in error message

### Processing Errors

**Scenario:** Detection inference fails (OOM, model error)
- **Detection:** Try-except blocks around model inference calls
- **Response:** Display error message with exception details
- **Recovery:** Allow user to adjust settings (lower resolution, reduce batch size)
- **User Guidance:** Suggest reducing image size or using CPU if GPU OOM

### Video Processing Errors

**Scenario:** Video codec not supported or video file corrupted
- **Detection:** OpenCV VideoCapture returns False on read
- **Response:** Display error message indicating video processing failure
- **Recovery:** Allow user to upload different video or try different format
- **User Guidance:** Suggest converting video to MP4 format

### Configuration Errors

**Scenario:** Invalid threshold values (outside 0.0-1.0 range)
- **Detection:** Streamlit slider constraints prevent invalid values
- **Response:** UI prevents setting invalid values
- **Recovery:** Not needed (prevented by UI constraints)
- **User Guidance:** Slider shows valid range

### Memory Errors

**Scenario:** Large video or batch processing exceeds available memory
- **Detection:** System-level memory errors during processing
- **Response:** Display error message suggesting reducing workload
- **Recovery:** User can reduce max_frames, increase frame_sampling, or reduce batch size
- **User Guidance:** Provide specific suggestions based on mode

## Testing Strategy

The testing strategy employs a dual approach combining unit tests for specific scenarios and property-based tests for comprehensive validation across input spaces.

### Unit Testing

Unit tests focus on specific examples, edge cases, and integration points:

**Model Management:**
- Test successful model loading from valid file path
- Test error handling when model file is missing
- Test model caching behavior (same instance returned on multiple calls)
- Test device selection (GPU when available, CPU fallback)

**Detection Processing:**
- Test detection on sample images with known objects
- Test empty image (no detections)
- Test confidence threshold filtering with specific values
- Test class filter with specific class selections
- Test IoU threshold effects on overlapping detections

**Annotation:**
- Test annotation with labels enabled/disabled
- Test annotation with confidence scores enabled/disabled
- Test different box thickness values (1, 3, 5)
- Test color generation for different classes
- Test annotation on image with no detections

**Statistics:**
- Test statistics calculation on sample detection results
- Test statistics with zero detections (returns None)
- Test class distribution calculation
- Test confidence statistics (mean, min, max)

**Video Processing:**
- Test frame sampling with specific sampling rates (1, 5, 10)
- Test video processing with max frame limit
- Test output video creation and format
- Test frame count in output video

**Batch Processing:**
- Test batch with 1, 5, 10 images
- Test batch statistics aggregation
- Test batch with mixed image sizes

**Export:**
- Test CSV export format and content
- Test image export to PNG
- Test video export to MP4

**Error Handling:**
- Test invalid image format upload
- Test invalid video format upload
- Test corrupted file handling
- Test missing model file scenario

### Property-Based Testing

Property-based tests validate universal properties across randomly generated inputs. Each test should run a minimum of 100 iterations to ensure comprehensive coverage.

**Test Configuration:**
- Use `pytest` with `hypothesis` library for Python property-based testing
- Configure each test with `@given` decorators for input generation
- Set `max_examples=100` minimum for each property test
- Tag each test with feature name and property number

**Property Test Implementation:**

Each correctness property from the design document should be implemented as a property-based test:

**Property 1: Confidence Threshold Filtering**
- **Feature: yolov8-detection-suite, Property 1:** Confidence threshold filtering
- Generate: random images, random threshold values (0.0-1.0)
- Assert: all detections have confidence >= threshold

**Property 2: Class Filter Enforcement**
- **Feature: yolov8-detection-suite, Property 2:** Class filter enforcement
- Generate: random images, random class filter subsets
- Assert: all detections are in selected classes

**Property 3: Frame Sampling Correctness**
- **Feature: yolov8-detection-suite, Property 3:** Frame sampling correctness
- Generate: random videos, random sampling rates (1-30)
- Assert: processed frame indices are multiples of sampling rate

**Property 4: Batch Processing Completeness**
- **Feature: yolov8-detection-suite, Property 4:** Batch processing completeness
- Generate: random lists of images (1-20 images)
- Assert: output count equals input count

**Property 5: Batch Configuration Consistency**
- **Feature: yolov8-detection-suite, Property 5:** Batch configuration consistency
- Generate: random image batches, random configurations
- Assert: all images processed with same config

**Property 6: Batch Statistics Aggregation**
- **Feature: yolov8-detection-suite, Property 6:** Batch statistics aggregation
- Generate: random image batches with detections
- Assert: aggregate count equals sum of individual counts

**Property 7: Configuration Range Validation**
- **Feature: yolov8-detection-suite, Property 7:** Configuration range validation
- Generate: random float values including out-of-range values
- Assert: only values in [0.0, 1.0] accepted

**Property 8: Annotation Styling Consistency**
- **Feature: yolov8-detection-suite, Property 8:** Annotation styling consistency
- Generate: random detection results, random styling settings
- Assert: all boxes in result use same styling

**Property 9: Device Independence**
- **Feature: yolov8-detection-suite, Property 9:** Device independence
- Generate: random images
- Assert: CPU and GPU results are equivalent (within tolerance)

**Property 10: Detection Statistics Accuracy**
- **Feature: yolov8-detection-suite, Property 10:** Detection statistics accuracy
- Generate: random detection results
- Assert: calculated stats match actual data

**Property 11: CSV Export Completeness**
- **Feature: yolov8-detection-suite, Property 11:** CSV export completeness
- Generate: random detection results
- Assert: CSV has one row per detection with all fields

**Property 12: CSV Format Correctness**
- **Feature: yolov8-detection-suite, Property 12:** CSV format correctness
- Generate: random detection results
- Assert: CSV has header and consistent column count

**Property 13: Image Format Acceptance**
- **Feature: yolov8-detection-suite, Property 13:** Image format acceptance
- Generate: valid images in JPG, PNG, JPEG formats
- Assert: all load successfully

**Property 14: Video Format Acceptance**
- **Feature: yolov8-detection-suite, Property 14:** Video format acceptance
- Generate: valid videos in MP4, AVI, MOV formats
- Assert: all load successfully

**Property 15: Invalid Format Rejection**
- **Feature: yolov8-detection-suite, Property 15:** Invalid format rejection
- Generate: files with invalid extensions
- Assert: all rejected without processing

**Property 16: File Integrity Validation**
- **Feature: yolov8-detection-suite, Property 16:** File integrity validation
- Generate: valid and corrupted files
- Assert: corrupted files detected before processing

**Property 17: Annotation Thickness Application**
- **Feature: yolov8-detection-suite, Property 17:** Annotation thickness application
- Generate: random images, random thickness values (1-5)
- Assert: all boxes drawn with specified thickness

**Property 18: Aspect Ratio Preservation**
- **Feature: yolov8-detection-suite, Property 18:** Aspect ratio preservation
- Generate: random images with various aspect ratios
- Assert: output aspect ratio equals input aspect ratio

**Property 19: Video Resolution Preservation**
- **Feature: yolov8-detection-suite, Property 19:** Video resolution preservation
- Generate: random videos with various resolutions
- Assert: output resolution equals input resolution

**Property 20: Video Frame Detection Coverage**
- **Feature: yolov8-detection-suite, Property 20:** Video frame detection coverage
- Generate: random videos, random sampling rates
- Assert: all sampled frames have detections applied

### Test Data Generation

**Image Generators:**
- Solid color images (various sizes)
- Random noise images
- Synthetic images with geometric shapes
- Real images from test dataset (COCO validation set subset)

**Video Generators:**
- Synthetic videos with moving objects
- Various resolutions (480p, 720p, 1080p)
- Various frame rates (24, 30, 60 FPS)
- Various durations (1s, 5s, 10s)

**Detection Result Generators:**
- Empty results (no detections)
- Single detection
- Multiple detections (1-50)
- Various confidence scores (0.0-1.0)
- Various classes from COCO dataset

### Integration Testing

**End-to-End Workflows:**
- Complete image detection workflow (upload → detect → view → download)
- Complete video detection workflow (upload → configure → process → download)
- Complete batch processing workflow (upload multiple → process → view results → download)

**UI Integration:**
- Test Streamlit session state management
- Test configuration changes trigger re-detection
- Test file uploader integration
- Test download button functionality

### Performance Testing

**Benchmarks:**
- Model loading time
- Single image detection time (CPU vs GPU)
- Video processing throughput (frames per second)
- Batch processing time scaling with batch size
- Memory usage for various workloads

**Optimization Validation:**
- Verify model caching reduces loading time
- Verify GPU acceleration improves throughput
- Verify frame sampling reduces processing time proportionally

### Testing Balance

The testing strategy balances unit tests and property tests:
- **Unit tests** provide concrete examples and catch specific bugs
- **Property tests** provide comprehensive coverage and catch edge cases
- Together they ensure both correctness and robustness

Avoid writing excessive unit tests for scenarios already covered by property tests. Focus unit tests on:
- Integration points between components
- Specific error conditions
- UI behavior and user workflows
- Performance benchmarks

Property tests handle:
- Input validation across ranges
- Algorithmic correctness across input spaces
- Consistency properties
- Data integrity properties
