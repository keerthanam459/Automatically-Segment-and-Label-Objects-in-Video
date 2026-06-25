

# Solution to MATLAB and Simulink Challenge Project 203 – Automatically Segment and Label Objects in Video
## Program Link

https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Automatically%20Segment%20and%20Label%20Objects%20in%20Video

## Project Description Link

https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub

## Project Details
### Objective

The objective of this project is to automatically identify and label objects present in a video sequence using deep learning-based object detection techniques.

The system processes an input video, detects target objects, annotates them with class labels and confidence scores, and generates a labeled output video.

The solution aims to reduce manual annotation effort and support video analysis applications in intelligent transportation, surveillance, and computer vision systems.

## Implementation Approach
### Methodology

#### The implemented pipeline consists of:

1. Video loading and frame extraction.
2. Frame resizing to match the detector input size.
3. Object detection using a pretrained YOLOv4 detector.
4. Confidence threshold filtering.
5. Non-Maximum Suppression (NMS).
6. Object labeling and visualization.
7. Statistical analysis of detections.
8. Generation of an annotated output video.
   
## MATLAB Scripts Developed

The project is implemented using:

```text
main.m
```

The script performs:

1. Video reading
2. Frame extraction
3. YOLOv4 object detection
4. Label generation
5. Result visualization
6. Video export
7. Performance analysis
   
## Algorithms Implemented
### YOLOv4 Object Detection

A pretrained YOLOv4 detector with CSP-Darknet53 backbone is used to detect objects in each frame.

### Confidence Thresholding

Only detections above a confidence threshold are retained.

### Non-Maximum Suppression (NMS)

Duplicate bounding boxes are removed using MATLAB's built-in strongest selection mechanism.

### Object Annotation

Detected objects are annotated with:

- Bounding boxes
- Class labels
- Confidence scores

### Target Classes

- Person
- Car
- Truck
- Bus
- Motorbike
  
## Design Decisions
1. YOLOv4 was selected due to its high detection accuracy and real-time performance.
2. Frame size was fixed at 416 × 416 pixels.
3. Only transportation-related and human classes were considered.
4. Detection results are stored in MAT files.
5. Color-coded annotations improve visualization.

## Limitations
1. Performs object detection and labeling rather than pixel-level segmentation.
2. Object tracking is not implemented.
3. Detection quality depends on video quality and lighting conditions.
4. This implementation focuses on object detection and labeling using YOLOv4 bounding boxes. Pixel-level semantic or instance segmentation is not currently implemented.
   
## Repository Structure
```text
Automatically-Segment-and-Label-Objects-in-Video/
│
├── README.md
├── LICENSE
├── data/
│   └── sample/
│       └── New.mp4
├── matlab/
│   └── main.m
├── results/
│   ├── frames/
│   ├── annotated/
│   ├── labeled_output.avi
│   ├── detection_summary.png
│   └── detection_results.mat
└── docs/
    └── screenshots/
```
    
## Requirements

### MATLAB Version
MATLAB R2025b

### Required Toolboxes
1. Computer Vision Toolbox
2. Deep Learning Toolbox
3. Image Processing Toolbox
   
## How to Run
1. Clone the repository.
2. Open MATLAB.
3. Navigate to the project folder.
4. Open:

```text
matlab/main.m
```

5. Update the video path if necessary:

```matlab
cfg.videoPath = fullfile('data','sample','New.mp4');
```

6. Run the script.

The program will:

1. Extract frames
2. Detect objects
3. Generate annotated frames
4. Create detection statistics
5. Produce an annotated video
   
## Input Data

The sample input video can be downloaded from:

https://drive.google.com/drive/folders/1caavfQ1jB3pULgvL6vLerRduDDI8Erzn?usp=drive_link

After downloading, place the file in:

```text
data/sample/New.mp4
```

## Demo / Results
### Detection Summary

The project automatically generates:

1. Detection count per frame
2. Class distribution across all frames

### Output
```text
results/detection_summary.png
```

### Annotated Frames

```text
results/annotated/
```

### Output Video

https://drive.google.com/drive/folders/1nLeDii1cnra_o6WyOqxe8jV8Omiaseyv?usp=drive_link

```text
results/labeled_output.avi
```

### Detection Results

```text
results/detection_results.mat
```

## Testing and Verification

Run:

```matlab
main
```

## Expected outputs:

1. Extracted video frames
2. Annotated image frames
3. Detection summary plot
4. Detection results MAT file
5. Annotated output video

# References
1. MATLAB Computer Vision Toolbox Documentation
2. MATLAB Deep Learning Toolbox Documentation
3. COCO Dataset
4. MathWorks YOLOv4 Object Detector Documentation
